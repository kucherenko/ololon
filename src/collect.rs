//! Data collection module - streams Binance WebSocket data and creates 5-minute time-range samples
//!
//! Architecture:
//! 1. Collect prices during explicit 5-minute time windows (e.g., 11:00:00 - 11:05:00)
//! 2. When a window ends, immediately finalize it with the end price (first price of next window)
//! 3. Label the previous window based on price direction: 1=up, -1=down, 0=no change
//! 4. Store complete, labeled samples in the database

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use futures_util::StreamExt;
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use crate::db;

/// Configuration for the collect subcommand
#[derive(Debug, Clone)]
pub struct Config {
    pub db_path: String,
    pub ws_url: String,
    pub symbol: String,
    pub window_size: usize,
    #[allow(dead_code)]
    pub label_delay_secs: u64,  // Not used - kept for CLI compatibility
}

/// Binance aggregate trade message
#[derive(Debug, Deserialize)]
struct BinanceAggTradeMessage {
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "T")]
    trade_time: i64,
}

/// Price point collected within a time window
#[derive(Debug, Clone, Copy)]
struct PricePoint {
    price: f64,
    timestamp_ms: i64,
}

/// Completed window ready to be stored
#[derive(Debug, Clone)]
struct CompletedWindow {
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    end_price: f64,
    prices: Vec<PricePoint>,
}

/// Active time window being collected
#[derive(Debug, Clone)]
struct ActiveWindow {
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    prices: VecDeque<PricePoint>,
    first_timestamp_ms: i64,
    last_price: f64,
}

impl ActiveWindow {
    fn new(time_range_start: i64, time_range_end: i64, first_price: f64, first_timestamp_ms: i64) -> Self {
        let mut prices = VecDeque::new();
        prices.push_back(PricePoint { price: first_price, timestamp_ms: first_timestamp_ms });

        Self {
            time_range_start,
            time_range_end,
            start_price: first_price,
            prices,
            first_timestamp_ms,
            last_price: first_price,
        }
    }

    fn add_price(&mut self, price: f64, timestamp_ms: i64) {
        self.prices.push_back(PricePoint { price, timestamp_ms });
        self.last_price = price;
    }

    fn finalize_with_end_price(self, end_price: f64) -> CompletedWindow {
        CompletedWindow {
            time_range_start: self.time_range_start,
            time_range_end: self.time_range_end,
            start_price: self.start_price,
            end_price,
            prices: self.prices.into_iter().collect(),
        }
    }

    fn num_points(&self) -> usize {
        self.prices.len()
    }

    fn elapsed_secs(&self) -> i64 {
        if self.prices.is_empty() {
            return 0;
        }
        let last_ts = self.prices.back().map(|p| p.timestamp_ms).unwrap_or(self.first_timestamp_ms);
        (last_ts - self.first_timestamp_ms) / 1000
    }
}

impl CompletedWindow {
    /// Compute normalized log returns from all prices in the window
    fn compute_log_returns(&self) -> Option<Vec<f64>> {
        if self.prices.len() < 2 {
            return None;
        }

        let prices: Vec<f64> = self.prices.iter().map(|p| p.price).collect();

        // Compute consecutive log returns
        let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
        if returns.is_empty() {
            return None;
        }

        // Normalize to zero mean and unit variance
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            Some(vec![0.0; returns.len()])
        } else {
            Some(returns.iter().map(|r| (r - mean) / std).collect())
        }
    }

    /// Compute price direction label
    /// Returns: 1 = up, -1 = down, 0 = no change (within 0.01%)
    fn compute_direction(&self) -> i32 {
        let change_pct = self.end_price - self.start_price;
        if change_pct > 0.0 {
            1  // Up
        } else if change_pct < 0.0 {
            -1  // Down
        } else {
            0  // No significant change
        }
    }
}

/// Get the current 5-minute window boundaries aligned to clock time
fn get_current_window(timestamp_secs: i64, window_duration_secs: i64) -> (i64, i64) {
    let window_start = (timestamp_secs / window_duration_secs) * window_duration_secs;
    let window_end = window_start + window_duration_secs;
    (window_start, window_end)
}

/// Format timestamp as human-readable time range
fn format_time_range(start: i64, end: i64) -> String {
    let start_dt = Utc.timestamp_opt(start, 0).single().unwrap_or_else(|| Utc::now());
    let end_dt = Utc.timestamp_opt(end, 0).single().unwrap_or_else(|| Utc::now());
    format!("{} - {}", start_dt.format("%H:%M:%S"), end_dt.format("%H:%M:%S"))
}

/// Database operation commands
enum DbCommand {
    InsertWindow {
        time_range_start: i64,
        time_range_end: i64,
        start_price: f64,
        end_price: f64,
        feature_vector: Vec<f64>,
        num_points: usize,
        direction: i32,
    },
}

/// Main entry point for the collect subcommand
pub async fn run(config: Config) -> Result<()> {
    let window_duration_secs = 300i64; // 5 minutes fixed

    info!("=== BTC/USDT 5-Minute Window Data Collection ===");
    info!(
        symbol = %config.symbol,
        window_duration = "5 minutes",
        min_samples_per_window = config.window_size,
        database = %config.db_path,
        "Configuration"
    );

    // Initialize database
    let db_path = PathBuf::from(&config.db_path);

    // Check existing data
    if let Ok(conn) = rusqlite::Connection::open(&db_path) {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM features", [], |row| row.get(0)
        ).unwrap_or(0);
        if count > 0 {
            info!(existing_windows = count, "Found existing data in database");
        }
    }

    db::init_db(&db_path.to_string_lossy())?;
    info!("Database initialized");

    // Shared state
    let active_window: Arc<RwLock<Option<ActiveWindow>>> = Arc::new(RwLock::new(None));
    let previous_window: Arc<RwLock<Option<CompletedWindow>>> = Arc::new(RwLock::new(None));
    let db_path_for_writer = db_path.clone();

    // Statistics
    let total_windows_stored = Arc::new(AtomicU64::new(0));
    let total_prices_collected = Arc::new(AtomicU64::new(0));

    // Channel for database operations
    let (db_tx, mut db_rx) = mpsc::unbounded_channel::<DbCommand>();

    // Database writer task
    let total_stored_clone = Arc::clone(&total_windows_stored);
    let db_writer = tokio::spawn(async move {
        while let Some(cmd) = db_rx.recv().await {
            match cmd {
                DbCommand::InsertWindow {
                    time_range_start,
                    time_range_end,
                    start_price,
                    end_price,
                    feature_vector,
                    num_points,
                    direction,
                } => {
                    match rusqlite::Connection::open(&db_path_for_writer) {
                        Ok(conn) => {
                            // Check if this window already exists
                            let exists: bool = conn.query_row(
                                "SELECT COUNT(*) > 0 FROM features WHERE time_range_start = ?1",
                                rusqlite::params![time_range_start],
                                |row| row.get(0),
                            ).unwrap_or(false);

                            if !exists {
                                // Insert with direction label already set
                                let vector_json = serde_json::to_string(&feature_vector).unwrap_or_default();
                                let labeled_at = Utc::now().timestamp();
                                let feature_len = feature_vector.len();

                                match conn.execute(
                                    "INSERT INTO features
                                     (time_range_start, time_range_end, start_price, end_price,
                                      feature_vector, target, num_points, labeled_at)
                                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                                    rusqlite::params![
                                        time_range_start,
                                        time_range_end,
                                        start_price,
                                        end_price,
                                        vector_json,
                                        direction,
                                        num_points as i32,
                                        labeled_at,
                                    ],
                                ) {
                                    Ok(_) => {
                                        let count = total_stored_clone.fetch_add(1, Ordering::Relaxed) + 1;
                                        let dir_str = match direction {
                                            1 => "UP ↑",
                                            -1 => "DOWN ↓",
                                            _ => "FLAT →",
                                        };
                                        let change_pct = (end_price - start_price) / start_price * 100.0;

                                        info!(
                                            "✓ STORED WINDOW #{}: {} | {} -> {} | {} | {:.4}% | {} prices, {} features",
                                            count,
                                            format_time_range(time_range_start, time_range_end),
                                            format!("{:.2}", start_price),
                                            format!("{:.2}", end_price),
                                            dir_str,
                                            change_pct,
                                            num_points,
                                            feature_len
                                        );
                                    }
                                    Err(e) => error!(error = ?e, "Failed to insert window"),
                                }
                            } else {
                                debug!(
                                    time_range = %format_time_range(time_range_start, time_range_end),
                                    "Window already exists, skipping"
                                );
                            }
                        }
                        Err(e) => error!(error = ?e, "Failed to open database"),
                    }
                }
            }
        }
    });

    // Periodic status logger
    let active_window_clone = Arc::clone(&active_window);
    let prices_clone = Arc::clone(&total_prices_collected);
    let windows_clone = Arc::clone(&total_windows_stored);

    let status_logger = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(30));
        loop {
            ticker.tick().await;

            let window = active_window_clone.read().await;
            let total_prices = prices_clone.load(Ordering::Relaxed);
            let total_windows = windows_clone.load(Ordering::Relaxed);

            if let Some(ref w) = *window {
                let elapsed = w.elapsed_secs();
                let remaining = window_duration_secs - elapsed;
                let price_change = ((w.last_price - w.start_price) / w.start_price) * 100.0;
                let change_symbol = if price_change >= 0.0 { "↑" } else { "↓" };

                info!(
                    "📊 STATUS: {} | {} prices collected | {:2}m{:02}s elapsed, {}s remaining | {} {:.3}% | Total: {} windows stored, {} prices",
                    format_time_range(w.time_range_start, w.time_range_end),
                    w.num_points(),
                    elapsed / 60,
                    elapsed % 60,
                    remaining.max(0),
                    change_symbol,
                    price_change,
                    total_windows,
                    total_prices
                );
            } else {
                info!(
                    "📊 STATUS: Waiting for first trade... | Total: {} windows stored, {} prices",
                    total_windows,
                    total_prices
                );
            }
        }
    });

    // Connect to Binance WebSocket
    let stream_name = format!("{}@aggTrade", config.symbol.to_lowercase());
    let ws_url = format!("{}/{}", config.ws_url, stream_name);

    info!("Connecting to Binance WebSocket at {}", ws_url);

    let (ws_stream, _) = connect_async(&ws_url)
        .await
        .context("Failed to connect to Binance WebSocket")?;

    info!("✓ Connected to Binance WebSocket");
    info!("⏱  Windows are aligned to 5-minute boundaries (e.g., 11:00:00 - 11:05:00 UTC)");
    info!("📋 Each window will be labeled immediately when it closes using the next window's first price");
    info!("🏷  Direction labels: UP (1) if price up >0.01%, DOWN (-1) if price down >0.01%, FLAT (0) otherwise");
    info!("---");

    let (_, mut read) = ws_stream.split();
    let min_samples = config.window_size.max(2);

    // Main message loop
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(trade) = serde_json::from_str::<BinanceAggTradeMessage>(&text) {
                    let price: f64 = match trade.price.parse() {
                        Ok(p) if p > 0.0 => p,
                        _ => continue,
                    };
                    let timestamp_ms = trade.trade_time;
                    let timestamp_secs = timestamp_ms / 1000;

                    // Count price
                    let price_count = total_prices_collected.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    // Log first few prices
                    if price_count <= 3 {
                        info!(
                            "💰 Received trade #{}: price={:.2}, time={}",
                            price_count,
                            price,
                            Utc.timestamp_opt(timestamp_secs, 0)
                                .single()
                                .map(|dt| dt.format("%H:%M:%S").to_string())
                                .unwrap_or_else(|| timestamp_secs.to_string())
                        );
                    }

                    // Determine current 5-minute window
                    let (window_start, window_end) = get_current_window(timestamp_secs, window_duration_secs);

                    let mut window_guard = active_window.write().await;

                    // Check if we need to finalize current window and start a new one
                    let should_finalize = match window_guard.as_ref() {
                        None => false,
                        Some(w) => timestamp_secs >= w.time_range_end,
                    };

                    if should_finalize {
                        // Take the current window
                        let old_window = window_guard.take().unwrap();
                        let num_points = old_window.num_points();
                        let old_start = old_window.start_price;

                        // Log window completion
                        let window_elapsed = old_window.elapsed_secs();
                        info!(
                            "⏰ WINDOW COMPLETE: {} | {} prices in {}m{:02}s | Start: {:.2} | Last: {:.2} | Change: {:.4}%",
                            format_time_range(old_window.time_range_start, old_window.time_range_end),
                            num_points,
                            window_elapsed / 60,
                            window_elapsed % 60,
                            old_start,
                            old_window.last_price,
                            ((old_window.last_price - old_start) / old_start) * 100.0
                        );

                        // The first price of the new window becomes the end_price of the old window
                        let end_price = price;

                        // Finalize the old window
                        let completed = old_window.finalize_with_end_price(end_price);
                        let direction = completed.compute_direction();

                        // Process previous window (if exists) - store it now
                        {
                            let mut prev = previous_window.write().await;
                            if let Some(prev_completed) = prev.take() {
                                if prev_completed.prices.len() >= min_samples {
                                    if let Some(features) = prev_completed.compute_log_returns() {
                                        info!(
                                            "📦 STORING PREVIOUS: {} | {} features computed",
                                            format_time_range(prev_completed.time_range_start, prev_completed.time_range_end),
                                            features.len()
                                        );
                                        let _ = db_tx.send(DbCommand::InsertWindow {
                                            time_range_start: prev_completed.time_range_start,
                                            time_range_end: prev_completed.time_range_end,
                                            start_price: prev_completed.start_price,
                                            end_price: prev_completed.end_price,
                                            feature_vector: features,
                                            num_points: prev_completed.prices.len(),
                                            direction: prev_completed.compute_direction(),
                                        });
                                    }
                                } else {
                                    warn!(
                                        "⚠ SKIPPED (insufficient samples): {} | only {} prices (need {})",
                                        format_time_range(prev_completed.time_range_start, prev_completed.time_range_end),
                                        prev_completed.prices.len(),
                                        min_samples
                                    );
                                }
                            }

                            // Store current completed window as previous
                            if num_points >= min_samples {
                                let dir_str = match direction {
                                    1 => "UP ↑",
                                    -1 => "DOWN ↓",
                                    _ => "FLAT →",
                                };
                                info!(
                                    "📝 QUEUING: {} | End price: {:.2} | Direction: {}",
                                    format_time_range(completed.time_range_start, completed.time_range_end),
                                    end_price,
                                    dir_str
                                );
                                *prev = Some(completed);
                            } else {
                                warn!(
                                    "⚠ SKIPPED (insufficient samples): {} | only {} prices (need {})",
                                    format_time_range(completed.time_range_start, completed.time_range_end),
                                    num_points,
                                    min_samples
                                );
                            }
                        }

                        // Start new window
                        info!(
                            "🆕 NEW WINDOW: {} | First price: {:.2}",
                            format_time_range(window_start, window_end),
                            price
                        );

                        *window_guard = Some(ActiveWindow::new(window_start, window_end, price, timestamp_ms));
                    } else if let Some(ref mut window) = *window_guard {
                        // Add to existing window
                        window.add_price(price, timestamp_ms);

                        // Log every 1000th price at debug level
                        if window.num_points() % 1000 == 0 {
                            debug!(
                                time_range = %format_time_range(window.time_range_start, window.time_range_end),
                                prices = window.num_points(),
                                current_price = price,
                                "Collecting prices"
                            );
                        }
                    } else {
                        // First price ever - create initial window
                        info!(
                            "🆕 INITIAL WINDOW: {} | First price: {:.2}",
                            format_time_range(window_start, window_end),
                            price
                        );
                        *window_guard = Some(ActiveWindow::new(window_start, window_end, price, timestamp_ms));
                    }
                }
            }
            Ok(Message::Close(_)) => {
                warn!("WebSocket closed by server");
                break;
            }
            Err(e) => {
                error!(error = ?e, "WebSocket error");
                break;
            }
            _ => {}
        }
    }

    // Finalize any remaining windows
    info!("Shutting down, finalizing remaining windows...");

    {
        let mut window_guard = active_window.write().await;
        let mut prev_guard = previous_window.write().await;

        // Store previous window if exists
        if let Some(prev_completed) = prev_guard.take() {
            if prev_completed.prices.len() >= min_samples {
                if let Some(features) = prev_completed.compute_log_returns() {
                    info!(
                        "📦 FINALIZING PREVIOUS: {}",
                        format_time_range(prev_completed.time_range_start, prev_completed.time_range_end)
                    );
                    let _ = db_tx.send(DbCommand::InsertWindow {
                        time_range_start: prev_completed.time_range_start,
                        time_range_end: prev_completed.time_range_end,
                        start_price: prev_completed.start_price,
                        end_price: prev_completed.end_price,
                        feature_vector: features,
                        num_points: prev_completed.prices.len(),
                        direction: prev_completed.compute_direction(),
                    });
                }
            }
        }

        // Current window cannot be fully labeled (no next window), but store it with available data
        if let Some(current) = window_guard.take() {
            if current.num_points() >= min_samples {
                let end_price = current.prices.back().map(|p| p.price).unwrap_or(current.start_price);
                let completed = current.finalize_with_end_price(end_price);
                if let Some(features) = completed.compute_log_returns() {
                    info!(
                        "📦 FINALIZING CURRENT (partial): {}",
                        format_time_range(completed.time_range_start, completed.time_range_end)
                    );
                    let _ = db_tx.send(DbCommand::InsertWindow {
                        time_range_start: completed.time_range_start,
                        time_range_end: completed.time_range_end,
                        start_price: completed.start_price,
                        end_price: completed.end_price,
                        feature_vector: features,
                        num_points: completed.prices.len(),
                        direction: completed.compute_direction(),
                    });
                }
            }
        }
    }

    // Cleanup
    status_logger.abort();
    drop(db_tx);
    let _ = db_writer.await;

    let final_windows = total_windows_stored.load(Ordering::Relaxed);
    let final_prices = total_prices_collected.load(Ordering::Relaxed);

    info!("=== Collection stopped ===");
    info!(total_windows_stored = final_windows, total_prices_collected = final_prices, "Summary");

    Ok(())
}
