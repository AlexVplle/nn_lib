use crate::{
    activation::Activation,
    cost::CostFunction,
    error::NeuralNetworkError,
    layers::{ActivationLayer, DenseLayer, Layer, Trainable},
    metrics::{Benchmark, History, Metrics},
    optimizer::Optimizer,
    tensor::{Device, Tensor},
};
use log::debug;

#[derive(Default)]
pub struct SequentialBuilder {
    layers: Vec<Box<dyn Layer>>,
    metrics: Option<Metrics>,
}

impl SequentialBuilder {
    pub fn new() -> SequentialBuilder {
        Self {
            layers: vec![],
            metrics: None,
        }
    }

    /// Add a layer to the sequential neural network
    /// in a sequential neural network, layers are added left to right (input -> hidden -> output)
    pub fn push(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Build the neural network.
    /// Returns a `NeuralNetworkError` if the network is wrongly defined.
    /// See `NeuralNetworkError` for information on what can fail.
    pub fn compile(
        self,
        optimizer: impl Optimizer + 'static,
        cost_function: CostFunction,
    ) -> Result<Sequential, NeuralNetworkError> {
        if cost_function.is_output_dependant() {
            self.validate_last_layer_activation(&cost_function)?;
        }

        Ok(Sequential {
            layers: self.layers,
            cost_function,
            optimizer: Box::new(optimizer),
            metrics: self.metrics,
        })
    }

    /// Validates that the last layer's activation function is compatible with the given cost function.
    fn validate_last_layer_activation(
        &self,
        cost_function: &CostFunction,
    ) -> Result<(), NeuralNetworkError> {
        self.layers
            .last()
            .and_then(|layer: &Box<dyn Layer + 'static>| {
                layer.as_any().downcast_ref::<ActivationLayer>()
            })
            .map_or(
                Err(NeuralNetworkError::MissingActivationLayer),
                |activation_layer: &ActivationLayer| match cost_function {
                    CostFunction::Mse => Ok(()),
                    CostFunction::CrossEntropy
                        if activation_layer.activation == Activation::Softmax =>
                    {
                        Ok(())
                    }
                    CostFunction::BinaryCrossEntropy
                        if activation_layer.activation == Activation::Sigmoid =>
                    {
                        Ok(())
                    }
                    _ => Err(NeuralNetworkError::WrongOutputActivationLayer),
                },
            )
    }
}

pub struct Sequential {
    pub(crate) layers: Vec<Box<dyn Layer>>,
    pub(crate) cost_function: CostFunction,
    optimizer: Box<dyn Optimizer>,
    metrics: Option<Metrics>,
}

impl Sequential {
    /// Get the device of the network (from first trainable layer)
    fn get_device(&self) -> Device {
        for layer in &self.layers {
            if let Some(trainable) = layer.as_any().downcast_ref::<crate::layers::DenseLayer>() {
                return trainable.get_parameters()[0].device().clone();
            }
        }
        Device::CPU
    }

    /// Predict a value from the neural network
    ///
    /// # Arguments
    /// * `input` - Batched input tensor, shape: [batch_size, input_features]
    ///
    /// # Returns
    /// * `Result<Tensor, NeuralNetworkError>` - Output tensor, shape: [batch_size, output_features]
    pub fn predict(&self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        let mut current_output = input.clone();

        for layer in &self.layers {
            current_output = layer.feed_forward(&current_output)?;
        }

        Ok(current_output)
    }

    /// Evaluate the trained neural network on test data
    ///
    /// # Arguments
    /// * `test_data` - Test data tensors (features, labels), shapes: [n_samples, n_features] and [n_samples, n_classes]
    /// * `batch_size` - Batch size for evaluation
    ///
    /// # Returns
    /// * `Result<Benchmark, NeuralNetworkError>` - Evaluation metrics and loss
    pub fn evaluate(
        &self,
        test_data: (&Tensor, &Tensor),
        batch_size: usize,
    ) -> Result<Benchmark, NeuralNetworkError> {
        let metrics = self
            .metrics
            .clone()
            .expect("Metrics must be set before evaluation");
        let mut bench = Benchmark::new(metrics);
        let (x, y) = test_data;

        assert_eq!(x.shape()[0], y.shape()[0]);
        let batches = Self::create_batches(x, y, batch_size)?;

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        let network_device = self.get_device();

        for (batched_x, batched_y) in batches.into_iter() {
            let batch_x_device = batched_x.to_device(network_device.clone())?;
            let batch_y_device = batched_y.to_device(network_device.clone())?;

            let output = self.predict(&batch_x_device)?;
            let batch_loss = self.cost_function.cost(&output, &batch_y_device)?;
            bench.metrics.accumulate(&output, &batch_y_device)?;
            total_loss += batch_loss;
            batch_count += 1;
        }

        bench.metrics.finalize();
        bench.loss = total_loss / batch_count as f64;
        Ok(bench)
    }

    /// Train the neural network with gradient descent
    ///
    /// # Arguments
    /// * `train_data` - Training data tensors (features, labels)
    /// * `validation_data` - Optional validation data tensors
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Size of mini-batches
    ///
    /// # Returns
    /// * `Result<(History, Option<History>), NeuralNetworkError>` - Training and validation history
    pub fn train(
        &mut self,
        train_data: (&Tensor, &Tensor),
        validation_data: Option<(&Tensor, &Tensor)>,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(History, Option<History>), NeuralNetworkError> {
        let (x_train, y_train) = train_data;

        if x_train.shape()[0] != y_train.shape()[0] {
            return Err(NeuralNetworkError::DimensionMismatch);
        }

        let mut train_history = History::new();
        let mut validation_history = validation_data.map(|_| History::new());
        let batches = Self::create_batches(x_train, y_train, batch_size)?;

        for e in 0..epochs {
            debug!("Training epoch: {}", e);
            let epoch_result = self.process_epoch(&batches)?;
            train_history.history.push(epoch_result);

            if let Some((x_val, y_val)) = validation_data {
                let validation_bench = self.evaluate((x_val, y_val), batch_size)?;
                validation_history
                    .as_mut()
                    .unwrap()
                    .history
                    .push(validation_bench);
            }
        }

        Ok((train_history, validation_history))
    }

    /// Process one training epoch
    fn process_epoch(
        &mut self,
        batches: &[(Tensor, Tensor)],
    ) -> Result<Benchmark, NeuralNetworkError> {
        let metrics = self
            .metrics
            .clone()
            .expect("Metrics must be set before training");
        let mut bench = Benchmark::new(metrics);
        let mut total_loss = 0.0;

        let network_device = self.get_device();

        for (_idx, (batched_x, batched_y)) in batches.iter().enumerate() {
            let batch_x_device = batched_x.to_device(network_device.clone())?;
            let batch_y_device = batched_y.to_device(network_device.clone())?;

            let output = self.feed_forward(&batch_x_device)?;
            let batch_loss = self.cost_function.cost(&output, &batch_y_device)?;
            total_loss += batch_loss;
            bench.metrics.accumulate(&output, &batch_y_device)?;

            self.backpropagation(&output, &batch_y_device)?;
        }

        bench.metrics.finalize();
        bench.loss = total_loss / batches.len() as f64;
        Ok(bench)
    }

    /// Create batches from training data with shuffling
    ///
    /// # Arguments
    /// * `x_train` - Training features tensor, shape: [n_samples, n_features]
    /// * `y_train` - Training labels tensor, shape: [n_samples, n_classes]
    /// * `batch_size` - Size of each batch
    fn create_batches(
        x_train: &Tensor,
        y_train: &Tensor,
        batch_size: usize,
    ) -> Result<Vec<(Tensor, Tensor)>, NeuralNetworkError> {
        let n_samples = x_train.shape()[0];

        let mut batches = Vec::new();

        for start_idx in (0..n_samples).step_by(batch_size) {
            let end_idx = (start_idx + batch_size).min(n_samples);
            let batch_x = x_train.slice(0, start_idx..end_idx)?;
            let batch_y = y_train.slice(0, start_idx..end_idx)?;
            batches.push((batch_x, batch_y));
        }

        Ok(batches)
    }

    /// Forward pass through all layers with gradient tracking
    pub fn feed_forward(&mut self, input: &Tensor) -> Result<Tensor, NeuralNetworkError> {
        let mut current_output = input.clone();

        for layer in &mut self.layers {
            current_output = layer.feed_forward_save(&current_output)?;
        }

        Ok(current_output)
    }

    /// Backpropagation algorithm
    ///
    /// # Arguments
    /// * `net_output` - Network output tensor, shape: [batch_size, n_classes]
    /// * `observed` - One-hot encoded labels tensor, shape: [batch_size, n_classes]
    fn backpropagation(
        &mut self,
        net_output: &Tensor,
        observed: &Tensor,
    ) -> Result<(), NeuralNetworkError> {
        let mut current_gradient = self
            .cost_function
            .cost_output_gradient(net_output, observed)?;

        for layer in self.layers.iter_mut().rev() {
            current_gradient = layer.propagate_backward(&current_gradient)?;

            if let Some(trainable_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
                self.optimizer.step(trainable_layer);
            }
        }

        Ok(())
    }
}
