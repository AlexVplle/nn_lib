// MNIST training example with MLP
//
// CPU backend (default): cargo run --example mlp_mnist --release
// Metal backend:         cargo run --example mlp_mnist --release --features metal

use ndarray::{Array2, ArrayD};
use nn_lib::{
    activation::Activation,
    cost::CostFunction,
    initialization::InitializerType,
    layers::{ActivationLayer, DenseLayer},
    metrics::{Metrics, MulticlassMetricType},
    optimizer::GradientDescent,
    sequential::SequentialBuilder,
    tensor::{Device, Tensor},
};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
struct MnistData {
    pub training: (ArrayD<u8>, ArrayD<u8>),
    pub test: (ArrayD<u8>, ArrayD<u8>),
}

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    #[cfg(feature = "metal")]
    let backend_name = "Metal";
    #[cfg(not(feature = "metal"))]
    let backend_name = "CPU";

    println!("MNIST MLP Training - {} Backend", backend_name);
    println!("=====================================\n");

    println!("Loading MNIST dataset...");
    let dataset = load_dataset()?;

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::CPU;

    println!("Preparing data...");
    let (x_train, y_train) = prepare_data(dataset.training, device.clone())?;
    let (x_test, y_test) = prepare_data(dataset.test, device.clone())?;

    let x_validation = x_train.slice(0, 48000..60000)?;
    let y_validation = y_train.slice(0, 48000..60000)?;
    let x_train = x_train.slice(0, 0..48000)?;
    let y_train = y_train.slice(0, 0..48000)?;

    println!("Building MLP network (784 -> 256 -> 128 -> 10)...");
    let metrics = Metrics::multiclass_classification(&vec![MulticlassMetricType::Accuracy]);

    let mut net = SequentialBuilder::new()
        .push(DenseLayer::new(784, 256, InitializerType::He, device.clone())?)
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(256, 128, InitializerType::He, device.clone())?)
        .push(ActivationLayer::from(Activation::ReLU))
        .push(DenseLayer::new(128, 10, InitializerType::He, device)?)
        .push(ActivationLayer::from(Activation::Softmax))
        .with_metrics(metrics)
        .compile(GradientDescent::new(0.1), CostFunction::CrossEntropy)?;

    println!("\nTraining for 10 epochs (batch size: 128)...\n");
    let start = std::time::Instant::now();

    let (train_hist, validation_hist) = net.train(
        (&x_train, &y_train),
        Some((&x_validation, &y_validation)),
        10,
        128,
    )?;

    let elapsed = start.elapsed();
    println!("\nTraining completed in {:.2}s\n", elapsed.as_secs_f64());

    println!("Results:");
    for (i, (train, validation)) in train_hist
        .history
        .iter()
        .zip(validation_hist.unwrap().history.iter())
        .enumerate()
    {
        let train_acc = train
            .metrics
            .get_metric(MulticlassMetricType::Accuracy)
            .unwrap_or(0.0);
        let val_acc = validation
            .metrics
            .get_metric(MulticlassMetricType::Accuracy)
            .unwrap_or(0.0);

        println!(
            "Epoch {}: train_loss={:.4}, val_loss={:.4}, train_acc={:.2}%, val_acc={:.2}%",
            i,
            train.loss,
            validation.loss,
            train_acc * 100.0,
            val_acc * 100.0
        );
    }

    println!("\nEvaluating on test set...");
    let bench = net.evaluate((&x_test, &y_test), 10)?;

    if let Some(accuracy) = bench.metrics.get_metric(MulticlassMetricType::Accuracy) {
        println!("Test accuracy: {:.2}%", accuracy * 100.0);
    }
    println!("Test loss: {:.4}", bench.loss);

    println!("\nSaving model to mnist_model.json...");
    net.save("mnist_model.json")?;
    println!("Model saved!");

    Ok(())
}

fn load_dataset() -> anyhow::Result<MnistData> {
    let (training_images, training_labels) = (
        load_file("train-images-idx3-ubyte.gz")?,
        load_file("train-labels-idx1-ubyte.gz")?,
    );
    let (test_images, test_labels) = (
        load_file("t10k-images-idx3-ubyte.gz")?,
        load_file("t10k-labels-idx1-ubyte.gz")?,
    );

    Ok(MnistData {
        training: (training_images, training_labels),
        test: (test_images, test_labels),
    })
}

fn load_file(file_name: &str) -> anyhow::Result<ArrayD<u8>> {
    let possible_paths = vec![
        PathBuf::from("mnist/resources/compressed").join(file_name),
        PathBuf::from("../mnist/resources/compressed").join(file_name),
        PathBuf::from("resources/compressed").join(file_name),
        PathBuf::from("data/compressed").join(file_name),
    ];

    let compressed_path = possible_paths
        .iter()
        .find(|p| p.exists())
        .ok_or_else(|| anyhow::anyhow!("Could not find MNIST file: {}", file_name))?;

    let file_stem = Path::new(file_name)
        .file_stem()
        .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;

    let raw_dir = compressed_path
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("raw");
    fs::create_dir_all(&raw_dir)?;

    let raw_path = raw_dir.join(file_stem);

    decompress_gz(&compressed_path, &raw_path)?;
    read_idx_file(&raw_path)
}

fn decompress_gz(src: &Path, dst: &Path) -> io::Result<()> {
    use flate2::bufread::GzDecoder;
    use std::io::{BufReader, BufWriter};

    if dst.exists() {
        return Ok(());
    }

    let file = File::open(src)?;
    let buf_reader = BufReader::new(file);
    let mut gz = GzDecoder::new(buf_reader);
    let output_file = File::create(dst)?;
    let mut buf_writer = BufWriter::new(output_file);

    io::copy(&mut gz, &mut buf_writer)?;
    Ok(())
}

fn read_idx_file(path: &Path) -> anyhow::Result<ArrayD<u8>> {
    use byteorder::{BigEndian, ReadBytesExt};
    use std::io::BufReader;

    let mut f = BufReader::new(File::open(path)?);
    let magic_number = f.read_u32::<BigEndian>()?;
    let num_dimension = magic_number & 0xFF;

    let mut shape = Vec::new();
    for _ in 0..num_dimension {
        shape.push(f.read_u32::<BigEndian>()? as usize);
    }

    let total_size: usize = shape.iter().product();
    let mut data = vec![0u8; total_size];
    f.read_exact(&mut data)?;

    Ok(ArrayD::from_shape_vec(shape, data)?)
}

fn prepare_data(
    data: (ArrayD<u8>, ArrayD<u8>),
    device: Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x = data.0.mapv(|e| e as f32 / 255.0);
    let outer = x.shape()[0];
    let x = x.into_shape((outer, 28 * 28))?;

    let y = one_hot_encode(&data.1, 10);

    let x_vec: Vec<f32> = x.iter().copied().collect();
    let y_vec: Vec<f32> = y.iter().copied().collect();

    let x_tensor = Tensor::new(x_vec, vec![outer, 28 * 28], device.clone())?;
    let y_tensor = Tensor::new(y_vec, vec![outer, 10], device)?;

    Ok((x_tensor, y_tensor))
}

fn one_hot_encode(labels: &ArrayD<u8>, num_classes: usize) -> Array2<f32> {
    let num_labels = labels.len();
    let mut one_hot = Array2::<f32>::zeros((num_labels, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }
    one_hot
}
