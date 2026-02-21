FROM rust:1.93-slim AS builder

WORKDIR /app

# Install build dependencies (native-tls for websocket, openssl for ormlite/sqlite, curl for utoipa-swagger-ui)
RUN apt-get update && apt-get install -y pkg-config libssl-dev libsqlite3-dev curl && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy actual source and build
COPY src ./src

# Force recompilation by touching source files
RUN find src -name "*.rs" -exec touch {} \; && cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates curl libssl3 libsqlite3-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/ololon /usr/local/bin/ololon

ENTRYPOINT ["ololon"]
