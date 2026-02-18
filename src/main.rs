//! Ololon - BTC/USDT 5-minute prediction trading bot
//!
//! A Rust-based trading bot that predicts Bitcoin price direction and places
//! automated bets on Polymarket's btc-updown-5m events using deep learning.

mod collect;
mod db;
mod train;
mod trade;

use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "ololon")]
#[command(about = "BTC/USDT 5-minute prediction trading bot", long_about = None)]
#[command(version)]
struct Cli {
    /// Path to SQLite database file
    #[arg(short, long, global = true, default_value = "ololon.db")]
    database: String,

    /// Path to model weights file (for train/trade commands)
    #[arg(short, long, global = true, default_value = "model.json.gz")]
    model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Collect real-time price data from Binance and create labeled training samples
    Collect {
        /// Binance WebSocket URL
        #[arg(long, default_value = "wss://stream.binance.com:9443/ws")]
        ws_url: String,

        /// Trading pair symbol
        #[arg(long, default_value = "btcusdt")]
        symbol: String,

        /// Window size for feature computation (number of price points)
        #[arg(long, default_value = "60")]
        window_size: usize,

        /// Labeling delay in seconds (default: 300 = 5 minutes)
        #[arg(long, default_value = "300")]
        label_delay_secs: u64,
    },

    /// Train the LSTM model on collected labeled data
    Train {
        /// Number of training epochs
        #[arg(long, default_value = "100")]
        epochs: usize,

        /// Batch size for training
        #[arg(long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "0.001")]
        learning_rate: f64,

        /// Hidden size for LSTM layer
        #[arg(long, default_value = "64")]
        hidden_size: usize,

        /// Number of LSTM layers
        #[arg(long, default_value = "2")]
        num_layers: usize,

        /// Validation split ratio (0.0 - 1.0)
        #[arg(long, default_value = "0.2")]
        validation_split: f64,
    },

    /// Run live trading with model inference and Polymarket execution
    Trade {
        /// Binance WebSocket URL
        #[arg(long, default_value = "wss://stream.binance.com:9443/ws")]
        ws_url: String,

        /// Trading pair symbol
        #[arg(long, default_value = "btcusdt")]
        symbol: String,

        /// Polymarket Gamma API base URL
        #[arg(long, default_value = "https://gamma-api.polymarket.com")]
        gamma_api_url: String,

        /// Polymarket CLOB API base URL
        #[arg(long, default_value = "https://clob.polymarket.com")]
        clob_api_url: String,

        /// Minimum probability edge required to place a trade
        #[arg(long, default_value = "0.05")]
        min_edge: f64,

        /// Trade size in USDC
        #[arg(long, default_value = "10.0")]
        trade_size: f64,

        /// Ethereum private key for L1 signing (or set POLY_PRIVATE_KEY env var)
        #[arg(long, env = "POLY_PRIVATE_KEY")]
        private_key: Option<String>,

        /// Polymarket L2 API key (or set POLY_API_KEY env var)
        #[arg(long, env = "POLY_API_KEY")]
        api_key: Option<String>,

        /// Polymarket L2 API secret (or set POLY_API_SECRET env var)
        #[arg(long, env = "POLY_API_SECRET")]
        api_secret: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load .env file if present
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    match cli.command {
        Commands::Collect {
            ws_url,
            symbol,
            window_size,
            label_delay_secs,
        } => {
            collect::run(collect::Config {
                db_path: cli.database,
                ws_url,
                symbol,
                window_size,
                label_delay_secs,
            })
            .await?;
        }
        Commands::Train {
            epochs,
            batch_size,
            learning_rate,
            hidden_size,
            num_layers,
            validation_split,
        } => {
            train::run(train::TrainConfig {
                db_path: cli.database,
                model_path: cli.model,
                epochs,
                batch_size,
                learning_rate,
                hidden_size,
                num_layers,
                validation_split,
            })
            .await?;
        }
        Commands::Trade {
            ws_url,
            symbol,
            gamma_api_url,
            clob_api_url,
            min_edge,
            trade_size,
            private_key,
            api_key,
            api_secret,
        } => {
            trade::run(trade::TradeConfig {
                db_path: cli.database,
                model_path: cli.model,
                ws_url,
                symbol,
                gamma_api_url,
                clob_api_url,
                min_edge,
                trade_size,
                hidden_size: 64,      // Default hidden size for inference
                num_layers: 2,         // Default num layers for inference
                private_key,
                api_key,
                api_secret,
            })
            .await?;
        }
    }

    Ok(())
}