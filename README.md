# Ololon

BTC/USDT 5-minute prediction trading bot using deep learning and Polymarket.

## Overview

Ololon is a Rust-based trading system that:
1. **Collects** real-time price data from Binance WebSocket
2. **Trains** an LSTM neural network to predict 5-minute price direction
3. **Trades** automatically on Polymarket's btc-updown-5m prediction markets

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Binance   │────▶│  Training   │────▶│  Polymarket │
│  WebSocket  │     │    LSTM     │     │   Trading   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Price Data        Model Weights        Trade Execution
       │                   │                   │
       └───────────────────┴───────────────────┘
                           │
                    ┌──────▼──────┐
                    │   SQLite    │
                    │  Database   │
                    └─────────────┘
```

## Features

- **Real-time Data Collection**: Streams aggregate trade data from Binance
- **5-Minute Time Windows**: Collects price sequences for LSTM input
- **Automatic Labeling**: Labels samples based on price direction
- **Deep Learning**: LSTM model with Burn framework (ndarray backend)
- **Live Inference**: Predicts up/down probability in real-time
- **Polymarket Integration**: Executes trades when probability edge exceeds threshold

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ololon.git
cd ololon

# Build
cargo build --release
```

## Usage

### 1. Collect Training Data

```bash
cargo run -- collect
```

Options:
- `--ws-url`: Binance WebSocket URL (default: `wss://stream.binance.com:9443/ws`)
- `--symbol`: Trading pair (default: `btcusdt`)
- `--window-size`: Number of price points per window (default: `60`)
- `--label-delay-secs`: Delay before labeling (default: `300` = 5 minutes)

Data is stored in SQLite (`ololon.db` by default).

### 2. Train the Model

```bash
cargo run -- train
```

Options:
- `--epochs`: Training epochs (default: `100`)
- `--batch-size`: Batch size (default: `32`)
- `--learning-rate`: Learning rate (default: `0.001`)
- `--hidden-size`: LSTM hidden size (default: `64`)
- `--num-layers`: Number of LSTM layers (default: `2`)
- `--validation-split`: Validation split ratio (default: `0.2`)

Model weights are saved to `model.json.gz`.

### 3. Live Trading

```bash
# Set environment variables
export POLY_PRIVATE_KEY="your_ethereum_private_key"
export POLY_API_KEY="your_polymarket_api_key"
export POLY_API_SECRET="your_polymarket_api_secret"

cargo run -- trade
```

Options:
- `--min-edge`: Minimum probability edge to trade (default: `0.05`)
- `--trade-size`: Trade size in USDC (default: `10.0`)
- `--gamma-api-url`: Polymarket Gamma API URL
- `--clob-api-url`: Polymarket CLOB API URL

## Database Schema

### features table
| Column | Description |
|--------|-------------|
| `time_range_start` | Start of 5-min window (Unix timestamp) |
| `time_range_end` | End of 5-min window |
| `start_price` | Price at window start |
| `end_price` | Price at window end |
| `feature_vector` | JSON array of normalized log returns |
| `target` | 1 = price went up, 0 = down |

### trades table
| Column | Description |
|--------|-------------|
| `timestamp` | Trade timestamp |
| `market_id` | Polymarket market ID |
| `outcome` | Prediction (YES/NO) |
| `predicted_prob` | Model's predicted probability |
| `market_prob` | Market's implied probability |
| `edge` | Probability difference |
| `trade_size` | Size in USDC |
| `status` | Trade status |

## Model Architecture

```
Input (sequence of log returns)
    │
    ▼
┌─────────────┐
│    LSTM     │  (hidden_size=64, num_layers=2)
└─────────────┘
    │
    ▼
┌─────────────┐
│   Linear    │  (hidden_size → 1)
└─────────────┘
    │
    ▼
┌─────────────┐
│   Sigmoid   │  → Probability [0, 1]
└─────────────┘
```

**Loss Function**: Binary Cross-Entropy

**Training**: Adam optimizer with configurable learning rate

## Configuration

Global options (all commands):
- `-d, --database`: SQLite database path (default: `ololon.db`)
- `-m, --model`: Model weights path (default: `model.json.gz`)

## Development

```bash
# Run tests
cargo test

# Lint
cargo clippy

# Format
cargo fmt

# Security audit
cargo audit
```

## Dependencies

| Category | Crates |
|----------|--------|
| Deep Learning | `burn`, `burn-ndarray` |
| Database | `rusqlite` |
| Async | `tokio`, `futures-util`, `tokio-stream` |
| WebSocket | `tokio-tungstenite` |
| HTTP | `reqwest` |
| Crypto | `k256`, `hmac`, `sha2`, `sha3` |
| Logging | `tracing`, `tracing-subscriber` |
| Error Handling | `anyhow`, `thiserror` |
| CLI | `clap` |

## License

Proprietary - All rights reserved. See [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves significant risk. Use at your own risk.