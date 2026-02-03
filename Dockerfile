FROM rustlang/rust:nightly as builder

WORKDIR /app

COPY Cargo.toml ./
COPY src ./src
COPY examples ./examples
COPY models ./models

RUN cargo build --example websocket_server --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/examples/websocket_server /app/websocket_server
COPY --from=builder /app/models /app/models

ENV MODEL_PATH=/app/models/mnist_model.json
ENV LISTEN_ADDR=0.0.0.0:8080

EXPOSE 8080

CMD ["/app/websocket_server"]
