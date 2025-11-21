use crate::{
    activation::Activation,
    cost::CostFunction,
    layers::{ActivationLayer, ConvolutionalLayer, DenseLayer, Layer, LayerError},
    metrics::{Benchmark, History, Metrics},
    optimizer::Optimizer,
};
use log::debug;
use candle_core::Tensor;
use rand::seq::SliceRandom;
use rand::thread_rng;
use thiserror::Error;

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
        // Check if the cost function is compatible with the last layer's activation function
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

/// a trainable `NeuralNetwork`
/// # Fields
/// * `layers` - A vector of layers (could be activation, convolutional, dense, etc..) in
/// sequential order
/// note that this crate dont use autodiff, so if you are planning to use a neural net architecture
/// with cross entropy, or binary cross entropy, the network make and use the assumption of
/// softmax, and sigmoid activation function respectively just before the cost function.
/// Thus you don't need to include it in the layers. However if you use any kind of independent
/// cost function (like mse) you can include whatever activation function you want after the
/// output because the gradient calculation is independent of the last layer you choose.
/// * cost_function - TODO
/// * optimoizer - TODO
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    cost_function: CostFunction,
    optimizer: Box<dyn Optimizer>,
    metrics: Option<Metrics>,
}

impl Sequential {
    /// predict a value from the neural network
    /// the shape of the prediction is (n, dim o) where **dim o** is the dimension of the network
    /// last layer and **n** is the number of point in the batch.
    ///
    /// # Arguments
    /// * `input` : batched input, of size (n, dim i) where **dim i** is the dimension of the
    /// network first layer and **n** is the number of point in the batch.
    pub fn predict(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.feed_forward(&output)?;
        }
        Ok(output)
    }

    /// Evaluate the **trained** neural network on a test input and observed values.
    /// returning a `Benchmark` containing the error on the test set, along with the metrics
    /// provided
    ///
    /// # Arguments
    /// * `test_data` test data set, the outer dimension must contain the data
    /// * `metrics` optional metrics struct
    /// * `batch_size` the batch size, ie: number of data point treated simultaneously
    pub fn evaluate(
        &self,
        test_data: (&Tensor, &Tensor),
        batch_size: usize,
    ) -> Benchmark {
        let metrics: Metrics = self
            .metrics
            .clone()
            .expect("Metrics must be set before calling evaluate");
        let mut bench: Benchmark = Benchmark::new(metrics);
        let (x, y) = test_data;
        assert_eq!(x.dims()[0], y.dims()[0]);
        let batches = Self::create_batches(x, y, batch_size);

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (batched_x, batched_y) in batches.into_iter() {
            let output = self.predict(&batched_x).unwrap();

            let batch_loss = self.cost_function.cost(&output, &batched_y);

            bench.metrics.accumulate(&output, &batched_y);

            total_loss += batch_loss;
            batch_count += 1;
        }

        bench.metrics.finalize();
        bench.loss = total_loss / batch_count as f64;
        bench
    }

    /// Train the neural network with Gradient descent Algorithm
    /// # Arguments
    /// * `train_data`
    pub fn train(
        &mut self,
        train_data: (&Tensor, &Tensor),
        validation_data: Option<(&Tensor, &Tensor)>,
        epochs: usize,
        batch_size: usize,
    ) -> Result<(History, Option<History>), LayerError> {
        let (x_train, y_train) = train_data;

        if x_train.dims()[0] != y_train.dims()[0] {
            return Err(LayerError::DimensionMismatch);
        }

        let mut train_history = History::new();
        let mut validation_history = validation_data.map(|_| History::new());

        let batches = Self::create_batches(x_train, y_train, batch_size);

        for e in 0..epochs {
            debug!("Training epochs : {}", e);
            let epoch_result = self.process_epoch(&batches)?;
            train_history.history.push(epoch_result);

            if let Some((x_val, y_val)) = validation_data {
                let validation_bench = self.evaluate((x_val, y_val), batch_size);
                validation_history
                    .as_mut()
                    .unwrap()
                    .history
                    .push(validation_bench);
            }
        }

        Ok((train_history, validation_history))
    }

    fn process_epoch(
        &mut self,
        batches: &[(Tensor, Tensor)],
    ) -> Result<Benchmark, LayerError> {
        let metrics = self
            .metrics
            .clone()
            .expect("Metrics must be set before calling train");
        let mut bench = Benchmark::new(metrics);
        let mut total_loss = 0.0;

        for (batched_x, batched_y) in batches.iter() {
            let output = self.feed_forward(batched_x)?;
            let batch_loss = self.cost_function.cost(&output, batched_y);

            total_loss += batch_loss;

            bench.metrics.accumulate(&output, batched_y);
            self.backpropagation(&output, batched_y)?;
        }

        bench.metrics.finalize();
        bench.loss = total_loss / batches.len() as f64;

        Ok(bench)
    }

    fn create_batches(
        x_train: &Tensor,
        y_train: &Tensor,
        batch_size: usize,
    ) -> Vec<(Tensor, Tensor)> {
        let num_samples = x_train.dims()[0];
        let mut indices: Vec<usize> = (0..num_samples).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        indices
            .chunks(batch_size)
            .map(|batch_indices| {
                let indices_u32: Vec<u32> = batch_indices.iter().map(|&i| i as u32).collect();
                let index_tensor = Tensor::from_slice(&indices_u32, batch_indices.len(), x_train.device()).expect("Failed to create index tensor");
                let x_batch = x_train.index_select(&index_tensor, 0).expect("Failed to select batch");
                let y_batch = y_train.index_select(&index_tensor, 0).expect("Failed to select batch");
                (x_batch, y_batch)
            })
            .collect()
    }

    pub fn feed_forward(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.feed_forward_save(&output)?;
        }
        Ok(output)
    }

    fn backpropagation(
        &mut self,
        net_output: &Tensor,
        observed: &Tensor,
    ) -> Result<(), LayerError> {
        let mut grad = self
            .cost_function
            .cost_output_gradient(net_output, observed);

        // if the cost function is dependant of the last layer, the gradient calculation
        // have been done with respect to the net logits directly, thus skip the last layer
        // in the gradients backpropagation
        let skip_layer = if self.cost_function.is_output_dependant() {
            1
        } else {
            0
        };

        for layer in self.layers.iter_mut().rev().skip(skip_layer) {
            grad = layer.propagate_backward(&grad)?;

            // Downcast to Trainable and call optimizes step method if possible
            // is other layers (like convolutional implement trainable, need to downcast
            // explicitly)
            if let Some(trainable_layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
                self.optimizer.step(trainable_layer);
            }

            if let Some(trainable_layer) = layer.as_any_mut().downcast_mut::<ConvolutionalLayer>() {
                self.optimizer.step(trainable_layer);
            }
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum NeuralNetworkError {
    #[error("Missing a last activation layer before the output")]
    MissingActivationLayer,

    #[error(
        "Invalid output activation layer,
        see CostFunction::output_dependant for detailed explanation"
    )]
    WrongOutputActivationLayer,
}
