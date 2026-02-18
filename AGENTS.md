# AGENTS.md - AI Coding Agent Guide

## Project Overview

Ololon is a Rust-based BTC/USDT 5-minute prediction trading bot that collects real-time price data from Binance, trains an LSTM model using the Burn framework, and executes automated trades on Polymarket.

## Build & Run Commands

```bash
cargo build                 # Build the project
cargo build --release       # Build optimized
cargo check                 # Check for compilation errors
cargo test                  # Run all tests
cargo test test_name        # Run a single test
cargo test --lib db::tests  # Run tests in a module
cargo test -- --nocapture   # Run with verbose output
cargo run -- collect        # Run the collector
cargo run -- train          # Run training
cargo run -- trade          # Run trading
cargo fmt                   # Format code
cargo clippy                # Lint with clippy
cargo clippy --fix          # Fix clippy warnings
cargo audit                 # Check for vulnerabilities
```

## Code Style Guidelines

### Module Structure
- Each module in its own file (`src/db.rs`, `src/train.rs`, etc.)
- Module-level docs use `//!`, public functions use `///`

### Imports
Group in order with blank lines between:
1. Standard library (`std::...`)
2. External crates (`anyhow::`, `tokio::`, `serde::`)
3. Local crate (`crate::db`)

```rust
use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::db;
```

### Naming Conventions
- **Types/Structs/Enums**: `PascalCase` (e.g., `TrainConfig`, `PricePredictor`)
- **Functions/Variables**: `snake_case` (e.g., `fetch_labeled_features`)
- **Type Aliases**: `PascalCase` (e.g., `type MyBackend = NdArray<f32>;`)

### Error Handling
Use `anyhow::Result<T>` for fallible functions with `.context()` for context:

```rust
pub fn load_data(path: &str) -> Result<Vec<Sample>> {
    let data = std::fs::read_to_string(path)
        .context("Failed to read data file")?;
    if data.is_empty() {
        anyhow::bail!("Data file is empty");
    }
    Ok(parse_data(&data))
}
```

### Logging
Use `tracing` for structured logging:

```rust
use tracing::{debug, error, info, warn};
info!(count = samples.len(), "Loaded training data");
error!(error = ?err, "Failed to connect");
```

### Async Code
- Use `tokio` as async runtime
- Use `#[tokio::main]` for main function
- Use `tokio::sync::RwLock` for async-safe shared state

### Dead Code
Mark intentionally unused items: `#[allow(dead_code)]`

### Tests
Place tests in `#[cfg(test)] mod tests` block at file end. Use `tempfile` for temp files:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_init_db() {
        let temp_file = NamedTempFile::new().unwrap();
        let conn = init_db(temp_file.path().to_str().unwrap()).unwrap();
    }
}
```

### Configuration Structs
Use dedicated config structs for module parameters:

```rust
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub db_path: String,
    pub model_path: String,
    pub epochs: usize,
    pub batch_size: usize,
}
```

### CLI with Clap
Use derive macros:

```rust
#[derive(Parser)]
#[command(name = "ololon", about = "BTC/USDT trading bot")]
struct Cli {
    #[arg(short, long, global = true)]
    database: String,
    #[command(subcommand)]
    command: Commands,
}
```

### Database Patterns
- Use `rusqlite::params![]` for query parameters
- Handle `QueryReturnedNoRows` explicitly:

```rust
match conn.query_row("SELECT ...", [], |row| Ok(row.get(0)?)) {
    Ok(val) => Ok(Some(val)),
    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
    Err(e) => Err(e.into()),
}
```

## Project Structure

```
src/
├── main.rs       # CLI entry point with subcommands
├── collect.rs    # Binance WebSocket data collection
├── db.rs         # SQLite database operations
├── train.rs      # LSTM model training (Burn framework)
└── trade.rs      # Live trading with Polymarket API
```

## Dependencies

- **Deep Learning**: `burn` with `ndarray` backend
- **Database**: `rusqlite` with bundled feature
- **Async**: `tokio`, `futures-util`, `tokio-stream`
- **WebSocket**: `tokio-tungstenite`
- **HTTP**: `reqwest` with JSON and rustls
- **Crypto**: `k256`, `hmac`, `sha2`, `sha3` for Polymarket signing
- **Logging**: `tracing`, `tracing-subscriber`
- **Error Handling**: `anyhow`, `thiserror`
- **CLI**: `clap` with derive feature
- **Testing**: `tempfile` (dev dependency)