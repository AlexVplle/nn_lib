// Load trained model and run inference on CPU
//
// Usage: cargo run --example load_and_inference --release

use ndarray::{Array2, ArrayD};
use nn_lib::{
    metrics::{Metrics, MulticlassMetricType},
    sequential::Sequential,
    tensor::{Device, Tensor},
};

fn main() -> anyhow::Result<()> {
    println!("Loading model from mnist_model.json...");

    let device = Device::CPU;
    let net = Sequential::load("mnist_model.json", device.clone())?;

    println!("Model loaded on CPU device!");

    println!("\nLoading test dataset...");
    let mnist_data = load_dataset()?;

    let x_test: Tensor = Tensor::new(
        mnist_data
            .x_test
            .iter()
            .copied()
            .map(|x| x as f32)
            .collect(),
        vec![mnist_data.x_test.nrows(), mnist_data.x_test.ncols()],
        device.clone(),
    )?;

    let y_test: Tensor = Tensor::new(
        mnist_data
            .y_test
            .iter()
            .copied()
            .map(|x| x as f32)
            .collect(),
        vec![mnist_data.y_test.nrows(), mnist_data.y_test.ncols()],
        device.clone(),
    )?;

    println!("Evaluating on test set...");
    let bench = net.evaluate((&x_test, &y_test), 10)?;

    if let Some(accuracy) = bench.metrics.get_metric(MulticlassMetricType::Accuracy) {
        println!("Test accuracy: {:.2}%", accuracy * 100.0);
    }
    println!("Test loss: {:.4}", bench.loss);

    println!("\nTesting single prediction...");
    let single_sample = x_test.slice(0, 0..1)?;
    let prediction = net.predict(&single_sample)?;
    let pred_vec = prediction.to_vec()?;

    let predicted_digit = pred_vec
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    println!("Predicted digit: {}", predicted_digit);
    println!("Confidence: {:.2}%", pred_vec[predicted_digit] * 100.0);

    Ok(())
}

struct MnistData {
    x_test: Array2<f64>,
    y_test: Array2<f64>,
}

fn load_dataset() -> anyhow::Result<MnistData> {
    let (test_images, test_labels) = (
        load_file("t10k-images-idx3-ubyte.gz")?,
        load_file("t10k-labels-idx1-ubyte.gz")?,
    );

    let n_test = test_images.len() / (28 * 28);

    let x_test = Array2::from_shape_vec((n_test, 28 * 28), test_images)?
        .mapv(|x| x as f64 / 255.0);

    let mut y_test = Array2::<f64>::zeros((n_test, 10));
    for (i, &label) in test_labels.iter().enumerate() {
        y_test[[i, label as usize]] = 1.0;
    }

    Ok(MnistData { x_test, y_test })
}

fn load_file(path: &str) -> anyhow::Result<Vec<u8>> {
    use byteorder::{BigEndian, ReadBytesExt};
    use flate2::read::GzDecoder;
    use std::fs::File;
    use std::io::Read;

    let file = File::open(path)?;
    let mut decoder = GzDecoder::new(file);
    let mut buffer = Vec::new();
    decoder.read_to_end(&mut buffer)?;

    let mut cursor = std::io::Cursor::new(&buffer);

    let magic = cursor.read_u32::<BigEndian>()?;
    let num_items = cursor.read_u32::<BigEndian>()? as usize;

    if magic == 2051 {
        let _rows = cursor.read_u32::<BigEndian>()?;
        let _cols = cursor.read_u32::<BigEndian>()?;
    }

    let mut data = vec![0u8; buffer.len() - cursor.position() as usize];
    cursor.read_exact(&mut data)?;

    Ok(data)
}
