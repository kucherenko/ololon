//! Ololon - BTC/USDT 5-minute prediction trading bot
//!
//! A Rust-based trading bot that predicts Bitcoin price direction and places
//! automated bets on Polymarket's btc-updown-5m events using deep learning.

mod collect;
mod db;
mod server;
mod train;
mod trade;
mod validate;

use clap::{Parser, Subcommand};
use tracing::{Level, Metadata};
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt, Layer};

/// Filter that only allows logs marked for web visibility or WARN/ERROR level
/// 
/// Logs are marked for web by including `web = true` field:
/// ```ignore
/// info!(web = true, "This appears in web interface");
/// info!("This only appears in console");
/// ```
struct WebVisibleFilter;

/// Helper to check if an event has web=true field
struct WebFieldChecker {
    has_web_field: bool,
}

impl tracing::field::Visit for WebFieldChecker {
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if field.name() == "web" && value {
            self.has_web_field = true;
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
    fn record_u64(&mut self, _field: &tracing::field::Field, _value: u64) {}
    fn record_i64(&mut self, _field: &tracing::field::Field, _value: i64) {}
    fn record_f64(&mut self, _field: &tracing::field::Field, _value: f64) {}
    fn record_str(&mut self, _field: &tracing::field::Field, _value: &str) {}
    fn record_error(&mut self, _field: &tracing::field::Field, _value: &(dyn std::error::Error + 'static)) {}
}

impl<S: tracing::Subscriber> tracing_subscriber::layer::Filter<S> for WebVisibleFilter {
    fn enabled(&self, metadata: &Metadata<'_>, _ctx: &tracing_subscriber::layer::Context<'_, S>) -> bool {
        // Always allow WARN and ERROR to web
        if metadata.level() >= &Level::WARN {
            return true;
        }
        // For INFO and below, we need to check fields - return true here
        // and rely on the fact that events without web=true won't be recorded
        // The actual filtering happens at the event level
        true
    }
    
    fn event_enabled(&self, event: &tracing::Event<'_>, _ctx: &tracing_subscriber::layer::Context<'_, S>) -> bool {
        let metadata = event.metadata();
        
        // Always allow WARN and ERROR
        if metadata.level() >= &Level::WARN {
            return true;
        }
        
        // Check for web=true field
        let mut checker = WebFieldChecker { has_web_field: false };
        event.record(&mut checker);
        
        checker.has_web_field
    }
}

#[derive(Parser)]
#[command(name = "ololon")]
#[command(about = "BTC/USDT 5-minute prediction trading bot", long_about = None)]
#[command(version)]
struct Cli {
    /// Path to SQLite database file
    #[arg(short, long, global = true, env = "OLOLON_DATABASE", default_value = "ololon.db")]
    database: String,

    /// Path to model weights file (for train/trade commands)
    #[arg(short, long, global = true, env = "OLOLON_MODEL", default_value = "model.json.gz")]
    model: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Collect real-time price data from Binance and create labeled training samples
    Collect {
        /// Binance WebSocket URL
        #[arg(long, env = "OLOLON_WS_URL", default_value = "wss://stream.binance.com:9443/ws")]
        ws_url: String,

        /// Trading pair symbol
        #[arg(long, env = "OLOLON_SYMBOL", default_value = "btcusdt")]
        symbol: String,

        /// Window size for feature computation (number of price points)
        #[arg(long, env = "OLOLON_WINDOW_SIZE", default_value = "60")]
        window_size: usize,

        /// Labeling delay in seconds (default: 300 = 5 minutes)
        #[arg(long, env = "OLOLON_LABEL_DELAY_SECS", default_value = "300")]
        label_delay_secs: u64,

        /// Order book sampling interval in seconds
        #[arg(long, env = "OLOLON_OB_SAMPLE_INTERVAL", default_value = "5")]
        ob_sample_interval_secs: u64,
    },

    /// Train the LSTM model on collected labeled data
    Train {
        /// Number of training epochs
        #[arg(long, env = "OLOLON_EPOCHS", default_value = "100")]
        epochs: usize,

        /// Batch size for training
        #[arg(long, env = "OLOLON_BATCH_SIZE", default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, env = "OLOLON_LEARNING_RATE", default_value = "0.001")]
        learning_rate: f64,

        /// Hidden size for LSTM layer
        #[arg(long, env = "OLOLON_HIDDEN_SIZE", default_value = "64")]
        hidden_size: usize,

        /// Number of LSTM layers
        #[arg(long, env = "OLOLON_NUM_LAYERS", default_value = "2")]
        num_layers: usize,

        /// Validation split ratio (0.0 - 1.0)
        #[arg(long, env = "OLOLON_VALIDATION_SPLIT", default_value = "0.2")]
        validation_split: f64,
    },

    /// Run live trading with model inference and Polymarket execution
    Trade {
        /// Binance WebSocket URL
        #[arg(long, env = "OLOLON_WS_URL", default_value = "wss://stream.binance.com:9443/ws")]
        ws_url: String,

        /// Trading pair symbol
        #[arg(long, env = "OLOLON_SYMBOL", default_value = "btcusdt")]
        symbol: String,

        /// Polymarket Gamma API base URL
        #[arg(long, env = "OLOLON_GAMMA_API_URL", default_value = "https://gamma-api.polymarket.com")]
        gamma_api_url: String,

        /// Polymarket CLOB API base URL
        #[arg(long, env = "OLOLON_CLOB_API_URL", default_value = "https://clob.polymarket.com")]
        clob_api_url: String,

        /// Minimum probability edge required to place a trade
        #[arg(long, env = "OLOLON_MIN_EDGE", default_value = "0.05")]
        min_edge: f64,

        /// Trade size in USDC
        #[arg(long, env = "OLOLON_TRADE_SIZE", default_value = "10.0")]
        trade_size: f64,

        /// Hidden size for LSTM inference (must match trained model)
        #[arg(long, env = "OLOLON_HIDDEN_SIZE", default_value = "64")]
        hidden_size: usize,

        /// Number of LSTM layers for inference (must match trained model)
        #[arg(long, env = "OLOLON_NUM_LAYERS", default_value = "2")]
        num_layers: usize,

        /// Ethereum private key for L1 signing (or set POLY_PRIVATE_KEY env var)
        #[arg(long, env = "POLY_PRIVATE_KEY")]
        private_key: Option<String>,

        /// Polymarket L2 API key (or set POLY_API_KEY env var)
        #[arg(long, env = "POLY_API_KEY")]
        api_key: Option<String>,

        /// Polymarket L2 API secret (or set POLY_API_SECRET env var)
        #[arg(long, env = "POLY_API_SECRET")]
        api_secret: Option<String>,

        /// Run in dry-run mode without placing real trades (or set DRY_RUN=1)
        #[arg(long, env = "DRY_RUN")]
        dry_run: bool,
    },

    /// Start the REST API server with authentication
    Server {
        /// Authentication token for API access (required, or set OLOLON_AUTH_TOKEN env var)
        #[arg(long, env = "OLOLON_AUTH_TOKEN")]
        auth_token: String,

        /// Address to bind the server to
        #[arg(long, env = "OLOLON_BIND_ADDRESS", default_value = "0.0.0.0:3000")]
        bind_address: String,

        /// Path to the log file to serve
        #[arg(long, env = "OLOLON_LOG_PATH", default_value = "logs/ololon.log")]
        log_path: String,
    },

    /// Validate trained model against all labeled data in database
    Validate {
        /// Hidden size for LSTM inference (must match trained model)
        #[arg(long, env = "OLOLON_HIDDEN_SIZE", default_value = "64")]
        hidden_size: usize,

        /// Only validate on new data not used for training
        #[arg(long)]
        new_only: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs").ok();

    // Set up file appender with daily rotation
    // IMPORTANT: Keep the guard alive for the duration of the program
    let file_appender = tracing_appender::rolling::daily("logs", "ololon.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Initialize tracing with both stdout and JSON file logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        // Human-readable stdout - shows all INFO and above
        .with(tracing_subscriber::fmt::layer())
        // JSON file logging - only shows web-visible logs (web=true) or WARN/ERROR
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_writer(non_blocking)
                .with_span_events(FmtSpan::CLOSE)
                .with_target(true)
                .with_thread_ids(false)
                .with_thread_names(false)
                .with_filter(WebVisibleFilter),
        )
        .init();

    // Keep the guard alive - dropping it would stop log writing
    let _guard = guard;

    // Load .env file if present
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // Get command name for logging
    let command_name = match &cli.command {
        Commands::Collect { .. } => "collect",
        Commands::Train { .. } => "train",
        Commands::Trade { .. } => "trade",
        Commands::Server { .. } => "server",
        Commands::Validate { .. } => "validate",
    };

    // Create a span that will add "command" field to all logs within this context
    let command_span = tracing::info_span!("command", %command_name);
    let _enter = command_span.enter();

    match cli.command {
        Commands::Collect {
            ws_url,
            symbol,
            window_size,
            label_delay_secs,
            ob_sample_interval_secs,
        } => {
            collect::run(collect::Config {
                db_path: cli.database,
                ws_url,
                symbol,
                window_size,
                label_delay_secs,
                ob_sample_interval_secs,
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
            hidden_size,
            num_layers,
            private_key,
            api_key,
            api_secret,
            dry_run,
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
                hidden_size,
                num_layers,
                private_key,
                api_key,
                api_secret,
                dry_run,
            })
            .await?;
        }
        Commands::Server {
            auth_token,
            bind_address,
            log_path,
        } => {
            server::run(server::Config {
                db_path: cli.database,
                auth_token,
                bind_address,
                log_path,
            })
            .await?;
        }
        Commands::Validate { hidden_size, new_only } => {
            validate::run(validate::ValidateConfig {
                db_path: cli.database,
                model_path: cli.model,
                hidden_size,
                new_only,
            })
            .await?;
        }
    }

    Ok(())
}