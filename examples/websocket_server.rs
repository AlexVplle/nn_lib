use nn_lib::{sequential::Sequential, tensor::{Device, Tensor}};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

#[derive(Deserialize)]
struct CanvasData {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

#[derive(Deserialize)]
struct InferenceRequest {
    canvas_data: CanvasData,
    #[serde(default)]
    model_type: String,
}

#[derive(Serialize)]
struct InferenceResponse {
    predictions: Vec<f32>,
    predicted_digit: usize,
    confidence: f32,
}

struct ModelServer {
    model: Mutex<Sequential>,
    device: Device,
}

impl ModelServer {
    fn new(model_path: &str) -> anyhow::Result<Self> {
        println!("Loading model from {}...", model_path);
        let device = Device::CPU;
        let model = Sequential::load(model_path, device.clone())?;
        println!("Model loaded on CPU!");

        Ok(Self {
            model: Mutex::new(model),
            device
        })
    }

    fn preprocess_canvas(&self, canvas_data: &CanvasData) -> anyhow::Result<Tensor> {
        let mut input = vec![0.0f32; 28 * 28];

        for y in 0..28 {
            for x in 0..28 {
                let idx = (y * 28 + x) * 4;
                if idx + 3 < canvas_data.pixels.len() {
                    // Use RGB luminance instead of alpha
                    let r = canvas_data.pixels[idx] as f32;
                    let g = canvas_data.pixels[idx + 1] as f32;
                    let b = canvas_data.pixels[idx + 2] as f32;
                    let luminance = (r + g + b) / (3.0 * 255.0);
                    input[y * 28 + x] = luminance;
                }
            }
        }

        Ok(Tensor::new(input, vec![1, 784], self.device.clone())?)
    }

    fn predict(&self, canvas_data: &CanvasData) -> anyhow::Result<InferenceResponse> {
        let input = self.preprocess_canvas(canvas_data)?;
        let model = self.model.lock().unwrap();
        let output = model.predict(&input)?;
        let predictions = output.to_vec()?;

        let (predicted_digit, &confidence) = predictions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok(InferenceResponse {
            predictions,
            predicted_digit,
            confidence,
        })
    }
}

async fn handle_connection(
    stream: TcpStream,
    server: Arc<ModelServer>,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut write, mut read) = ws_stream.split();

    println!("New WebSocket connection established");

    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<InferenceRequest>(&text) {
                    Ok(request) => {
                        match server.predict(&request.canvas_data) {
                            Ok(response) => {
                                let response_json = serde_json::to_string(&response)?;
                                write.send(Message::Text(response_json)).await?;
                            }
                            Err(e) => {
                                eprintln!("Prediction error: {}", e);
                                let error = serde_json::json!({"error": e.to_string()});
                                write.send(Message::Text(error.to_string())).await?;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to parse request: {}", e);
                    }
                }
            }
            Ok(Message::Close(_)) => break,
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    println!("WebSocket connection closed");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("MNIST WebSocket Inference Server");
    println!("=================================\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/mnist_model.json".to_string());

    let server = Arc::new(ModelServer::new(&model_path)?);

    let addr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let addr: SocketAddr = addr.parse()?;

    let listener = TcpListener::bind(&addr).await?;
    println!("WebSocket server listening on {}", addr);
    println!("Ready to accept connections!\n");

    while let Ok((stream, peer_addr)) = listener.accept().await {
        println!("Connection from: {}", peer_addr);
        let server = Arc::clone(&server);

        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, server).await {
                eprintln!("Error handling connection from {}: {}", peer_addr, e);
            }
        });
    }

    Ok(())
}
