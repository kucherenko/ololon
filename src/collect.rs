//! Data collection module - streams Binance WebSocket data and creates 5-minute time-range samples
//!
//! Architecture:
//! 1. Collect prices during explicit 5-minute time windows (e.g., 11:00:00 - 11:05:00)
//! 2. Collect order book snapshots every N seconds for market depth features
//! 3. When a window ends, immediately save its feature vector to DB (with NULL target)
//! 4. When the next window completes, update the previous window's target based on price direction
//! 5. Direction labels: 1=up, -1=down, 0=no change
//!
//! Order book features (aggregated per window):
//! - mean/std imbalance: buying vs selling pressure
//! - mean/max spread: liquidity tightness
//! - depth ratio: bid vs ask volume

use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use futures_util::StreamExt;
use ormlite::sqlite::SqliteConnection;
use ormlite::Connection;
use serde::Deserialize;
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn, Instrument};

use crate::db;
use ordered_float::OrderedFloat;

/// Configuration for the collect subcommand
#[derive(Debug, Clone)]
pub struct Config {
    pub db_path: String,
    pub ws_url: String,
    pub symbol: String,
    pub window_size: usize,
    #[allow(dead_code)]
    pub label_delay_secs: u64,  // Not used - kept for CLI compatibility
    pub ob_sample_interval_secs: u64,  // Order book sampling interval
}

/// Order book level (price, quantity)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct ObLevel {
    price: f64,
    quantity: f64,
}

/// Local order book state maintained from depth stream updates
#[derive(Debug, Clone, Default)]
struct OrderBook {
    bids: BTreeMap<OrderedFloat<f64>, f64>,  // Sorted ascending, we want highest bid
    asks: BTreeMap<OrderedFloat<f64>, f64>,  // Sorted ascending, we want lowest ask
    last_update_id: u64,
}

impl OrderBook {
    /// Update from Binance depth event
    fn update(&mut self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>, update_id: u64) {
        // Update bids (quantity 0 means remove)
        for (price, qty) in bids {
            let key = OrderedFloat(price);
            if qty == 0.0 {
                self.bids.remove(&key);
            } else {
                self.bids.insert(key, qty);
            }
        }
        // Update asks
        for (price, qty) in asks {
            let key = OrderedFloat(price);
            if qty == 0.0 {
                self.asks.remove(&key);
            } else {
                self.asks.insert(key, qty);
            }
        }
        self.last_update_id = update_id;
    }

    /// Get best bid (highest buy price)
    fn best_bid(&self) -> Option<f64> {
        self.bids.iter().next_back().map(|(p, _)| p.into_inner())
    }

    /// Get best ask (lowest sell price)
    fn best_ask(&self) -> Option<f64> {
        self.asks.iter().next().map(|(p, _)| p.into_inner())
    }

    /// Get top N bid levels (highest first)
    fn top_bids(&self, n: usize) -> Vec<ObLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(p, q)| ObLevel { price: p.into_inner(), quantity: *q })
            .collect()
    }

    /// Get top N ask levels (lowest first)
    fn top_asks(&self, n: usize) -> Vec<ObLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(p, q)| ObLevel { price: p.into_inner(), quantity: *q })
            .collect()
    }
}

/// Order book snapshot features computed at a point in time
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)]
struct ObSnapshot {
    timestamp_ms: i64,
    imbalance: f64,      // (bid_vol - ask_vol) / (bid_vol + ask_vol), range [-1, 1]
    spread_bps: f64,     // (ask - bid) / mid_price * 10000
    bid_depth: f64,      // Sum of top 5 bid quantities
    ask_depth: f64,      // Sum of top 5 ask quantities
    depth_ratio: f64,    // bid_depth / ask_depth
    mid_price: f64,      // (bid + ask) / 2
}

impl ObSnapshot {
    /// Compute snapshot from order book state
    fn from_order_book(ob: &OrderBook, timestamp_ms: i64) -> Option<Self> {
        let best_bid = ob.best_bid()?;
        let best_ask = ob.best_ask()?;
        
        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        let spread_bps = if mid_price > 0.0 {
            (spread / mid_price) * 10000.0
        } else {
            0.0
        };

        let top_bids = ob.top_bids(5);
        let top_asks = ob.top_asks(5);
        
        let bid_depth: f64 = top_bids.iter().map(|l| l.quantity).sum();
        let ask_depth: f64 = top_asks.iter().map(|l| l.quantity).sum();
        
        let total_depth = bid_depth + ask_depth;
        let imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };
        
        let depth_ratio = if ask_depth > 0.0 {
            bid_depth / ask_depth
        } else if bid_depth > 0.0 {
            100.0  // Very bid-heavy
        } else {
            1.0  // No depth
        };

        Some(Self {
            timestamp_ms,
            imbalance,
            spread_bps,
            bid_depth,
            ask_depth,
            depth_ratio,
            mid_price,
        })
    }
}

/// Aggregated order book features for a time window
#[derive(Debug, Clone, Default)]
struct ObFeatures {
    mean_imbalance: f64,
    std_imbalance: f64,
    mean_spread_bps: f64,
    max_spread_bps: f64,
    mean_depth_ratio: f64,
    last_imbalance: f64,  // Final state of the window
}

impl ObFeatures {
    /// Aggregate snapshots into window-level features
    fn from_snapshots(snapshots: &[ObSnapshot]) -> Self {
        if snapshots.is_empty() {
            return Self::default();
        }

        let n = snapshots.len() as f64;
        
        let imbalances: Vec<f64> = snapshots.iter().map(|s| s.imbalance).collect();
        let spreads: Vec<f64> = snapshots.iter().map(|s| s.spread_bps).collect();
        let depth_ratios: Vec<f64> = snapshots.iter().map(|s| s.depth_ratio).collect();

        let mean_imbalance = imbalances.iter().sum::<f64>() / n;
        let mean_spread_bps = spreads.iter().sum::<f64>() / n;
        let mean_depth_ratio = depth_ratios.iter().sum::<f64>() / n;

        let variance_imbalance: f64 = imbalances
            .iter()
            .map(|x| (x - mean_imbalance).powi(2))
            .sum::<f64>() / n;
        let std_imbalance = variance_imbalance.sqrt();

        let max_spread_bps = spreads.iter().cloned().fold(0.0, f64::max);
        let last_imbalance = *imbalances.last().unwrap_or(&0.0);

        Self {
            mean_imbalance,
            std_imbalance,
            mean_spread_bps,
            max_spread_bps,
            mean_depth_ratio,
            last_imbalance,
        }
    }

    /// Convert to feature vector (normalized)
    fn to_feature_vector(&self) -> Vec<f64> {
        // Standardize features to reasonable ranges
        vec![
            self.mean_imbalance,      // Already in [-1, 1]
            self.std_imbalance,       // In [0, 1] typically
            self.mean_spread_bps / 10.0,  // Normalize by 10 bps
            self.max_spread_bps / 10.0,   // Normalize by 10 bps
            (self.mean_depth_ratio - 1.0).tanh(),  // Center around 1, bound to [-1, 1]
            self.last_imbalance,      // Already in [-1, 1]
        ]
    }
}

/// Binance aggregate trade message
#[derive(Debug, Deserialize)]
struct BinanceAggTradeMessage {
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "T")]
    trade_time: i64,
}

/// Binance depth stream message (partial book update)
#[derive(Debug, Deserialize)]
struct BinanceDepthMessage {
    #[serde(rename = "U")]
    #[allow(dead_code)]
    first_update_id: u64,
    #[serde(rename = "u")]
    final_update_id: u64,
    #[serde(rename = "b")]
    bids: Vec<[String; 2]>,  // [price, qty]
    #[serde(rename = "a")]
    asks: Vec<[String; 2]>,  // [price, qty]
}

/// Binance combined stream message wrapper
#[derive(Debug, Deserialize)]
struct BinanceCombinedMessage {
    stream: String,
    data: serde_json::Value,
}

impl BinanceDepthMessage {
    /// Parse bids/asks into (price, qty) tuples
    fn parse_levels(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids: Vec<(f64, f64)> = self.bids
            .iter()
            .filter_map(|[p, q]| {
                let price = p.parse().ok()?;
                let qty = q.parse().ok()?;
                if price > 0.0 && qty >= 0.0 { Some((price, qty)) } else { None }
            })
            .collect();
        
        let asks: Vec<(f64, f64)> = self.asks
            .iter()
            .filter_map(|[p, q]| {
                let price = p.parse().ok()?;
                let qty = q.parse().ok()?;
                if price > 0.0 && qty >= 0.0 { Some((price, qty)) } else { None }
            })
            .collect();
        
        (bids, asks)
    }
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
    ob_snapshots: Vec<ObSnapshot>,  // Order book snapshots collected during window
}

/// Active time window being collected
#[derive(Debug, Clone)]
struct ActiveWindow {
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    prices: VecDeque<PricePoint>,
    ob_snapshots: Vec<ObSnapshot>,  // Order book snapshots
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
            ob_snapshots: Vec::new(),
            first_timestamp_ms,
            last_price: first_price,
        }
    }

    fn add_price(&mut self, price: f64, timestamp_ms: i64) {
        self.prices.push_back(PricePoint { price, timestamp_ms });
        self.last_price = price;
    }

    fn add_ob_snapshot(&mut self, snapshot: ObSnapshot) {
        self.ob_snapshots.push(snapshot);
    }

    fn finalize_with_end_price(self, end_price: f64) -> CompletedWindow {
        CompletedWindow {
            time_range_start: self.time_range_start,
            time_range_end: self.time_range_end,
            start_price: self.start_price,
            end_price,
            prices: self.prices.into_iter().collect(),
            ob_snapshots: self.ob_snapshots,
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

    /// Compute full feature vector: log returns + order book features
    fn compute_feature_vector(&self) -> Option<Vec<f64>> {
        let log_returns = self.compute_log_returns()?;
        
        // Compute order book features
        let ob_features = ObFeatures::from_snapshots(&self.ob_snapshots);
        let ob_vector = ob_features.to_feature_vector();
        
        // Combine: [log_returns..., ob_features...]
        let mut features = log_returns;
        features.extend(ob_vector);
        
        Some(features)
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
    let start_dt = Utc.timestamp_opt(start, 0).single().unwrap_or_else(Utc::now);
    let end_dt = Utc.timestamp_opt(end, 0).single().unwrap_or_else(Utc::now);
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
    },
    UpdateTarget {
        time_range_start: i64,
        target: i32,
    },
    Shutdown,
}

/// Main entry point for the collect subcommand
pub async fn run(config: Config) -> Result<()> {
    let window_duration_secs = 300i64; // 5 minutes fixed

    info!("=== BTC/USDT 5-Minute Window Data Collection ===");
    info!(
        symbol = %config.symbol,
        window_duration = "5 minutes",
        min_samples_per_window = config.window_size,
        ob_sample_interval = format!("{}s", config.ob_sample_interval_secs),
        database = %config.db_path,
        "Configuration"
    );

    // Initialize database
    let db_path = PathBuf::from(&config.db_path);

    // Check existing data (async)
    {
        let mut conn = db::init_db(&db_path.to_string_lossy()).await?;
        let count = db::count_total_features(&mut conn).await.unwrap_or(0);
        if count > 0 {
            info!(existing_windows = count, "Found existing data in database");
        }
    }
    info!("Database initialized");

    // Shared state
    let active_window: Arc<RwLock<Option<ActiveWindow>>> = Arc::new(RwLock::new(None));
    // Track the last saved window's time_range_start so we can update its target when next window completes
    let last_saved_window_start: Arc<RwLock<Option<i64>>> = Arc::new(RwLock::new(None));
    let db_path_for_writer = db_path.clone();

    // Order book state
    let order_book: Arc<RwLock<OrderBook>> = Arc::new(RwLock::new(OrderBook::default()));

    // Statistics
    let total_windows_stored = Arc::new(AtomicU64::new(0));
    let total_prices_collected = Arc::new(AtomicU64::new(0));
    let total_ob_updates = Arc::new(AtomicU64::new(0));
    let total_ob_snapshots = Arc::new(AtomicU64::new(0));

    // Channel for database operations
    let (db_tx, mut db_rx) = mpsc::unbounded_channel::<DbCommand>();

    // Database writer task (async with ormlite)
    let total_stored_clone = Arc::clone(&total_windows_stored);
    let db_writer = tokio::spawn(
        async move {
            // Open connection once for the writer task
            let mut conn = match SqliteConnection::connect(&db_path_for_writer.to_string_lossy()).await {
                Ok(c) => c,
                Err(e) => {
                    error!(error = ?e, "Failed to open database connection for writer");
                    return;
                }
            };

            while let Some(cmd) = db_rx.recv().await {
            match cmd {
                DbCommand::InsertWindow {
                    time_range_start,
                    time_range_end,
                    start_price,
                    end_price,
                    feature_vector,
                    num_points,
                } => {
                    // Check if this window already exists
                    let exists_result: Result<Option<(i32,)>, _> = ormlite::query_as(
                        "SELECT 1 FROM features WHERE time_range_start = ?1 LIMIT 1",
                    )
                    .bind(time_range_start)
                    .fetch_optional(&mut conn)
                    .await;

                    let exists = matches!(exists_result, Ok(Some(_)));

                    if !exists {
                        // Insert WITH end_price but WITHOUT target - target will be set when next period completes
                        let vector_json = serde_json::to_string(&feature_vector).unwrap_or_default();
                        let feature_len = feature_vector.len();

                        let result = ormlite::query(
                            "INSERT INTO features
                             (time_range_start, time_range_end, start_price, end_price, feature_vector, num_points, created_at)
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%s', 'now'))",
                        )
                        .bind(time_range_start)
                        .bind(time_range_end)
                        .bind(start_price)
                        .bind(end_price)
                        .bind(&vector_json)
                        .bind(num_points as i32)
                        .execute(&mut conn)
                        .await;

                        match result {
                            Ok(_) => {
                                let count = total_stored_clone.fetch_add(1, Ordering::Relaxed) + 1;
                                let change_pct = (end_price - start_price) / start_price * 100.0;
                                info!(
                                    "💾 STORED #{}: {} | {:.2} -> {:.2} ({:.4}%) | {} prices, {} features | target: pending",
                                    count,
                                    format_time_range(time_range_start, time_range_end),
                                    start_price,
                                    end_price,
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
                DbCommand::UpdateTarget {
                    time_range_start,
                    target,
                } => {
                    let labeled_at = Utc::now().timestamp();
                    let result = ormlite::query(
                        "UPDATE features SET target = ?1, labeled_at = ?2 WHERE time_range_start = ?3",
                    )
                    .bind(target)
                    .bind(labeled_at)
                    .bind(time_range_start)
                    .execute(&mut conn)
                    .await;

                    match result {
                        Ok(_) => {
                            let dir_str = match target {
                                1 => "UP ↑",
                                -1 => "DOWN ↓",
                                _ => "FLAT →",
                            };
                            info!(
                                "🏷  LABELED: {} | Target: {}",
                                format_time_range(time_range_start, time_range_start + 300),
                                dir_str
                            );
                        }
                        Err(e) => error!(error = ?e, "Failed to update target"),
                    }
                }
                DbCommand::Shutdown => break,
            }
        }
    }.instrument(tracing::info_span!("db_writer", command_name = "collect")));

    // Order book snapshot sampler task
    let ob_sampler_interval = Duration::from_secs(config.ob_sample_interval_secs);
    let ob_book_clone = Arc::clone(&order_book);
    let active_window_clone = Arc::clone(&active_window);
    let ob_snapshots_clone = Arc::clone(&total_ob_snapshots);
    
    let ob_sampler = tokio::spawn(async move {
        let mut ticker = interval(ob_sampler_interval);
        loop {
            ticker.tick().await;
            
            let now_ms = Utc::now().timestamp_millis();
            
            // Get OB snapshot
            let ob = ob_book_clone.read().await;
            if let Some(snapshot) = ObSnapshot::from_order_book(&ob, now_ms) {
                drop(ob); // Release read lock before acquiring write
                
                // Add to active window if exists
                let mut window = active_window_clone.write().await;
                if let Some(ref mut w) = *window {
                    w.add_ob_snapshot(snapshot);
                    let count = ob_snapshots_clone.fetch_add(1, Ordering::Relaxed) + 1;
                    
                    // Log every 10th snapshot
                    if count.is_multiple_of(10) {
                        debug!(
                            imbalance = format!("{:.3}", snapshot.imbalance),
                            spread_bps = format!("{:.1}", snapshot.spread_bps),
                            depth_ratio = format!("{:.2}", snapshot.depth_ratio),
                            "OB snapshot #{}",
                            count
                        );
                    }
                }
            }
        }
    }.instrument(tracing::info_span!("ob_sampler", command_name = "collect")));

    // Periodic status logger
    let active_window_clone2 = Arc::clone(&active_window);
    let prices_clone = Arc::clone(&total_prices_collected);
    let windows_clone = Arc::clone(&total_windows_stored);
    let ob_snapshots_status = Arc::clone(&total_ob_snapshots);
    let ob_updates_status = Arc::clone(&total_ob_updates);

    let status_logger = tokio::spawn(async move {
        let mut ticker = interval(Duration::from_secs(30));
        loop {
            ticker.tick().await;

            let window = active_window_clone2.read().await;
            let total_prices = prices_clone.load(Ordering::Relaxed);
            let total_windows = windows_clone.load(Ordering::Relaxed);
            let total_ob_snapshots = ob_snapshots_status.load(Ordering::Relaxed);
            let total_ob_updates = ob_updates_status.load(Ordering::Relaxed);

            if let Some(ref w) = *window {
                let elapsed = w.elapsed_secs();
                let remaining = window_duration_secs - elapsed;
                let price_change = ((w.last_price - w.start_price) / w.start_price) * 100.0;
                let change_symbol = if price_change >= 0.0 { "↑" } else { "↓" };

                info!(
                    "📊 STATUS: {} | {} prices, {} OB samples | {:2}m{:02}s elapsed, {}s remaining | {} {:.3}% | Total: {} windows, {} prices, {} OB snapshots, {} OB updates",
                    format_time_range(w.time_range_start, w.time_range_end),
                    w.num_points(),
                    w.ob_snapshots.len(),
                    elapsed / 60,
                    elapsed % 60,
                    remaining.max(0),
                    change_symbol,
                    price_change,
                    total_windows,
                    total_prices,
                    total_ob_snapshots,
                    total_ob_updates
                );
            } else {
                info!(
                    "📊 STATUS: Waiting for first trade... | Total: {} windows, {} prices, {} OB snapshots, {} OB updates",
                    total_windows,
                    total_prices,
                    total_ob_snapshots,
                    total_ob_updates
                );
            }
        }
    }.instrument(tracing::info_span!("status_logger", command_name = "collect")));

    // Connect to Binance combined WebSocket stream (aggTrade + depth)
    // Using combined stream format: /stream?streams=btcusdt@aggTrade/btcusdt@depth@100ms
    // Base URL typically ends with /ws, so we strip it for combined streams
    let agg_trade_stream = format!("{}@aggTrade", config.symbol.to_lowercase());
    let depth_stream = format!("{}@depth@100ms", config.symbol.to_lowercase());
    let base_url = config.ws_url.trim_end_matches("/ws");
    let ws_url = format!("{}/stream?streams={}/{}", base_url, agg_trade_stream, depth_stream);

    info!("Connecting to Binance combined WebSocket at {}", ws_url);

    let (ws_stream, _) = connect_async(&ws_url)
        .await
        .context("Failed to connect to Binance WebSocket")?;

    info!("✓ Connected to Binance WebSocket (aggTrade + depth streams)");
    info!("⏱  Windows are aligned to 5-minute boundaries (e.g., 11:00:00 - 11:05:00 UTC)");
    info!("📋 Vectors are saved immediately when window ends, targets updated when next window completes");
    info!("📊 Order book snapshots collected every {}s for market depth features", config.ob_sample_interval_secs);
    info!("🏷  Direction labels: UP (1) if price up, DOWN (-1) if price down, FLAT (0) if unchanged");
    info!("---");

    let (_, mut read) = ws_stream.split();
    let min_samples = config.window_size.max(2);

    // Main message loop - handles both trade and depth messages
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse combined stream message
                if let Ok(combined) = serde_json::from_str::<BinanceCombinedMessage>(&text) {
                    let stream_name = combined.stream;
                    
                    // Handle depth stream updates
                    if stream_name.ends_with("@depth@100ms") {
                        if let Ok(depth) = serde_json::from_value::<BinanceDepthMessage>(combined.data) {
                            let (bids, asks) = depth.parse_levels();
                            
                            // Update order book
                            {
                                let mut ob = order_book.write().await;
                                ob.update(bids, asks, depth.final_update_id);
                            }
                            
                            total_ob_updates.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // Handle trade stream updates
                    else if stream_name.ends_with("@aggTrade") {
                        if let Ok(trade) = serde_json::from_value::<BinanceAggTradeMessage>(combined.data) {
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
                                let num_ob_snapshots = old_window.ob_snapshots.len();
                                let old_start = old_window.start_price;

                                // Log window completion
                                let window_elapsed = old_window.elapsed_secs();
                                info!(
                                    "⏰ WINDOW COMPLETE: {} | {} prices, {} OB samples in {}m{:02}s | Start: {:.2} | Last: {:.2} | Change: {:.4}%",
                                    format_time_range(old_window.time_range_start, old_window.time_range_end),
                                    num_points,
                                    num_ob_snapshots,
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

                                // Update target of previously saved window using THIS window's direction
                                {
                                    let mut last_start = last_saved_window_start.write().await;
                                    if let Some(prev_time_range_start) = last_start.take() {
                                        // The target for previous window is the direction of THIS window
                                        let dir_str = match direction {
                                            1 => "UP ↑",
                                            -1 => "DOWN ↓",
                                            _ => "FLAT →",
                                        };
                                        info!(
                                            "🔄 Updating target for previous window: direction={}",
                                            dir_str
                                        );
                                        let _ = db_tx.send(DbCommand::UpdateTarget {
                                            time_range_start: prev_time_range_start,
                                            target: direction,
                                        });
                                    }
                                }

                                // Save current window immediately (with end_price but without target)
                                if num_points >= min_samples {
                                    // Use combined feature vector (log returns + OB features)
                                    if let Some(features) = completed.compute_feature_vector() {
                                        let ob_features = ObFeatures::from_snapshots(&completed.ob_snapshots);
                                        let change_pct = (end_price - completed.start_price) / completed.start_price * 100.0;
                                        info!(
                                            "💾 SAVING: {} | {:.2} -> {:.2} ({:.4}%) | {} prices, {} OB samples, {} features | imbalance={:.3}, spread={:.1}bps",
                                            format_time_range(completed.time_range_start, completed.time_range_end),
                                            completed.start_price,
                                            end_price,
                                            change_pct,
                                            num_points,
                                            num_ob_snapshots,
                                            features.len(),
                                            ob_features.mean_imbalance,
                                            ob_features.mean_spread_bps
                                        );
                                        let _ = db_tx.send(DbCommand::InsertWindow {
                                            time_range_start: completed.time_range_start,
                                            time_range_end: completed.time_range_end,
                                            start_price: completed.start_price,
                                            end_price,
                                            feature_vector: features,
                                            num_points,
                                        });
                                        // Remember this window for target update
                                        *last_saved_window_start.write().await = Some(completed.time_range_start);
                                    }
                                } else {
                                    warn!(
                                        "⚠ SKIPPED (insufficient samples): {} | only {} prices (need {})",
                                        format_time_range(completed.time_range_start, completed.time_range_end),
                                        num_points,
                                        min_samples
                                    );
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
                                        ob_samples = window.ob_snapshots.len(),
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

        // Finalize current window (it will remain unlabeled since no future window)
        if let Some(current) = window_guard.take() {
            if current.num_points() >= min_samples {
                let end_price = current.prices.back().map(|p| p.price).unwrap_or(current.start_price);
                let completed = current.finalize_with_end_price(end_price);
                
                // Use combined feature vector
                if let Some(features) = completed.compute_feature_vector() {
                    let ob_features = ObFeatures::from_snapshots(&completed.ob_snapshots);
                    let change_pct = (end_price - completed.start_price) / completed.start_price * 100.0;
                    info!(
                        "💾 SAVING (unlabeled - no future window): {} | {:.2} -> {:.2} ({:.4}%) | {} prices, {} OB samples, {} features | imbalance={:.3}",
                        format_time_range(completed.time_range_start, completed.time_range_end),
                        completed.start_price,
                        end_price,
                        change_pct,
                        completed.prices.len(),
                        completed.ob_snapshots.len(),
                        features.len(),
                        ob_features.mean_imbalance
                    );
                    let _ = db_tx.send(DbCommand::InsertWindow {
                        time_range_start: completed.time_range_start,
                        time_range_end: completed.time_range_end,
                        start_price: completed.start_price,
                        end_price,
                        feature_vector: features,
                        num_points: completed.prices.len(),
                    });
                    // Note: This window remains unlabeled (no target) since there's no future window
                }
            }
        }
    }

    // Cleanup
    ob_sampler.abort();
    status_logger.abort();
    let _ = db_tx.send(DbCommand::Shutdown);
    drop(db_tx);
    let _ = db_writer.await;

    let final_windows = total_windows_stored.load(Ordering::Relaxed);
    let final_prices = total_prices_collected.load(Ordering::Relaxed);
    let final_obs = total_ob_snapshots.load(Ordering::Relaxed);

    info!("=== Collection stopped ===");
    info!(
        total_windows_stored = final_windows,
        total_prices_collected = final_prices,
        total_ob_snapshots = final_obs,
        "Summary"
    );

    Ok(())
}
