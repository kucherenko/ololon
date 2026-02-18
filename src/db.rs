//! Database module for SQLite operations using ormlite ORM
//!
//! Schema:
//! - `features`: Stores feature vectors computed from 5-minute time ranges
//! - `trades`: Trade history log
//! - `model_metadata`: Model training metadata

use anyhow::Result;
use ormlite::model::*;
use ormlite::sqlite::SqliteConnection;
use ormlite::{Connection, Executor};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Initialize the database connection and create schema
pub async fn init_db(db_path: &str) -> Result<SqliteConnection> {
    // Ensure parent directory exists (sqlx doesn't auto-create like rusqlite)
    let path = Path::new(db_path);
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    
    // Create empty file if it doesn't exist (required for sqlx)
    if !path.exists() {
        std::fs::File::create(db_path)?;
    }

    let mut conn = SqliteConnection::connect(db_path).await?;

    // Create schema with raw SQL (ormlite doesn't provide schema creation API)
    conn.execute(
        r#"
        -- Features table: stores vectors from 5-minute time ranges with delayed labels
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_range_start INTEGER NOT NULL,
            time_range_end INTEGER NOT NULL,
            start_price REAL NOT NULL,
            end_price REAL,
            feature_vector TEXT NOT NULL,
            target INTEGER,
            num_points INTEGER NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            labeled_at INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_features_time_range_start ON features(time_range_start);
        CREATE INDEX IF NOT EXISTS idx_features_time_range_end ON features(time_range_end);
        CREATE INDEX IF NOT EXISTS idx_features_target ON features(target);

        -- Trade history table
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            market_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            market_prob REAL NOT NULL,
            edge REAL NOT NULL,
            trade_size REAL NOT NULL,
            avg_price REAL NOT NULL,
            status TEXT NOT NULL,
            order_id TEXT,
            tx_hash TEXT,
            error_message TEXT,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        -- Model metadata table
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            model_path TEXT NOT NULL,
            trained_at INTEGER NOT NULL,
            epochs INTEGER NOT NULL,
            final_train_loss REAL NOT NULL,
            final_val_loss REAL,
            hidden_size INTEGER NOT NULL,
            num_layers INTEGER NOT NULL,
            window_duration_secs INTEGER NOT NULL
        );
        "#,
    )
    .await?;

    Ok(conn)
}

/// Feature record for database operations
#[derive(Debug, Clone, Model, Serialize, Deserialize)]
#[ormlite(table = "features", insert = "InsertFeature")]
pub struct Feature {
    #[ormlite(primary_key)]
    pub id: i64,
    pub time_range_start: i64,
    pub time_range_end: i64,
    pub start_price: f64,
    pub end_price: Option<f64>,
    #[ormlite(json)]
    pub feature_vector: Vec<f64>,
    pub target: Option<i32>,
    pub num_points: i32,
    #[ormlite(default)]
    pub created_at: i64,
    pub labeled_at: Option<i64>,
}

/// Trade record for database operations (main struct for reads)
#[allow(dead_code)]
#[derive(Debug, Clone, Model, Serialize, Deserialize)]
#[ormlite(table = "trades", insert = "InsertTrade")]
pub struct Trade {
    #[ormlite(primary_key)]
    pub id: i64,
    pub timestamp: i64,
    pub market_id: String,
    pub outcome: String,
    pub predicted_prob: f64,
    pub market_prob: f64,
    pub edge: f64,
    pub trade_size: f64,
    pub avg_price: f64,
    pub status: String,
    pub order_id: Option<String>,
    pub tx_hash: Option<String>,
    pub error_message: Option<String>,
    #[ormlite(default)]
    pub created_at: i64,
}

/// Model training metadata
#[allow(dead_code)]
#[derive(Debug, Clone, Model, Serialize, Deserialize)]
#[ormlite(table = "model_metadata")]
pub struct ModelMetadata {
    #[ormlite(primary_key)]
    pub id: i64,
    pub model_path: String,
    pub trained_at: i64,
    pub epochs: i32,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub hidden_size: i32,
    pub num_layers: i32,
    pub window_duration_secs: i64,
}

/// Insert a new feature vector (without label - end_price is unknown)
#[allow(dead_code)]
pub async fn insert_feature(
    conn: &mut SqliteConnection,
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    feature_vector: Vec<f64>,
    num_points: usize,
) -> Result<Feature> {
    let result = ormlite::query(
        r#"INSERT INTO features (time_range_start, time_range_end, start_price, feature_vector, num_points) 
           VALUES (?1, ?2, ?3, ?4, ?5)"#,
    )
    .bind(time_range_start)
    .bind(time_range_end)
    .bind(start_price)
    .bind(serde_json::to_string(&feature_vector)?)
    .bind(num_points as i32)
    .execute(&mut *conn)
    .await?;
    
    let id = result.last_insert_rowid();
    
    // Fetch the inserted row to return full Feature
    let feature = Feature::select()
        .where_("id = ?")
        .bind(id)
        .fetch_one(&mut *conn)
        .await?;
    
    Ok(feature)
}

/// Apply delayed label to a feature record using the end price
#[allow(dead_code)]
pub async fn apply_label_with_price(
    conn: &mut SqliteConnection,
    id: i64,
    end_price: f64,
    labeled_at: i64,
) -> Result<i32> {
    // Get the start price to determine the target
    let feature = Feature::select()
        .where_("id = ?")
        .bind(id)
        .fetch_one(&mut *conn)
        .await?;

    let target = if end_price >= feature.start_price { 1 } else { 0 };

    // Update using builder for partial update
    feature
        .update_partial()
        .target(Some(target))
        .end_price(Some(end_price))
        .labeled_at(Some(labeled_at))
        .update(&mut *conn)
        .await?;

    Ok(target)
}

/// Fetch features ready for labeling (time_range_end has passed)
#[allow(dead_code)]
pub async fn fetch_features_ready_for_labeling(
    conn: &mut SqliteConnection,
    current_time: i64,
) -> Result<Vec<Feature>> {
    let features = Feature::select()
        .where_("target IS NULL AND time_range_end <= ?")
        .bind(current_time)
        .fetch_all(conn)
        .await?;
    Ok(features)
}

/// Fetch all labeled features for training
pub async fn fetch_labeled_features(conn: &mut SqliteConnection) -> Result<Vec<Feature>> {
    let features = Feature::select()
        .where_("target IS NOT NULL")
        .order_asc("time_range_start")
        .fetch_all(conn)
        .await?;
    Ok(features)
}

/// Count labeled features
#[allow(dead_code)]
pub async fn count_labeled_features(conn: &mut SqliteConnection) -> Result<i64> {
    let count: (i64,) = ormlite::query_as(
        "SELECT COUNT(*) FROM features WHERE target IS NOT NULL",
    )
    .fetch_one(conn)
    .await?;
    Ok(count.0)
}

/// Count total features (labeled + unlabeled)
pub async fn count_total_features(conn: &mut SqliteConnection) -> Result<i64> {
    let count: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM features")
        .fetch_one(conn)
        .await?;
    Ok(count.0)
}

/// Get the latest time_range_end from features table
#[allow(dead_code)]
pub async fn get_latest_window_end(conn: &mut SqliteConnection) -> Result<Option<i64>> {
    let result: Option<(Option<i64>,)> = ormlite::query_as("SELECT MAX(time_range_end) FROM features")
        .fetch_optional(conn)
        .await?;
    Ok(result.and_then(|r| r.0))
}

/// Legacy struct for compatibility - TradeRecord
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub id: Option<i64>,
    pub timestamp: i64,
    pub market_id: String,
    pub outcome: String,
    pub predicted_prob: f64,
    pub market_prob: f64,
    pub edge: f64,
    pub trade_size: f64,
    pub avg_price: f64,
    pub status: String,
    pub order_id: Option<String>,
    pub tx_hash: Option<String>,
    pub error_message: Option<String>,
}

/// Insert a trade record
#[allow(dead_code)]
pub async fn insert_trade(conn: &mut SqliteConnection, trade: &TradeRecord) -> Result<i64> {
    let result = ormlite::query(
        r#"INSERT INTO trades (timestamp, market_id, outcome, predicted_prob, market_prob, 
           edge, trade_size, avg_price, status, order_id, tx_hash, error_message)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"#,
    )
    .bind(trade.timestamp)
    .bind(&trade.market_id)
    .bind(&trade.outcome)
    .bind(trade.predicted_prob)
    .bind(trade.market_prob)
    .bind(trade.edge)
    .bind(trade.trade_size)
    .bind(trade.avg_price)
    .bind(&trade.status)
    .bind(&trade.order_id)
    .bind(&trade.tx_hash)
    .bind(&trade.error_message)
    .execute(conn)
    .await?;
    
    Ok(result.last_insert_rowid())
}

/// Legacy struct for compatibility - ModelMetadataRecord
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelMetadataRecord {
    pub model_path: String,
    pub trained_at: i64,
    pub epochs: usize,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub window_duration_secs: i64,
}

/// Save model metadata after training
#[allow(clippy::too_many_arguments)]
pub async fn save_model_metadata(
    conn: &mut SqliteConnection,
    model_path: &str,
    epochs: usize,
    final_train_loss: f64,
    final_val_loss: Option<f64>,
    hidden_size: usize,
    num_layers: usize,
    window_duration_secs: i64,
) -> Result<()> {
    let trained_at = chrono::Utc::now().timestamp();

    // Use raw query for upsert since ormlite's OnConflict might not work with custom constraint
    ormlite::query(
        r#"INSERT OR REPLACE INTO model_metadata 
           (id, model_path, trained_at, epochs, final_train_loss, final_val_loss, hidden_size, num_layers, window_duration_secs)
           VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
    )
    .bind(model_path)
    .bind(trained_at)
    .bind(epochs as i32)
    .bind(final_train_loss)
    .bind(final_val_loss)
    .bind(hidden_size as i32)
    .bind(num_layers as i32)
    .bind(window_duration_secs)
    .execute(conn)
    .await?;

    Ok(())
}

/// Get model metadata
pub async fn get_model_metadata(conn: &mut SqliteConnection) -> Result<Option<(String, i64)>> {
    let result: Option<(String, i64)> = ormlite::query_as(
        "SELECT model_path, window_duration_secs FROM model_metadata WHERE id = 1",
    )
    .fetch_optional(conn)
    .await?;
    Ok(result)
}

// Re-export for backward compatibility
#[allow(dead_code)]
pub type FeatureRecord = Feature;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_db() {
        let mut conn = init_db(":memory:").await.unwrap();

        let count: (i64,) = ormlite::query_as(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('features', 'trades', 'model_metadata')",
        )
        .fetch_one(&mut conn)
        .await
        .unwrap();
        assert_eq!(count.0, 3);
    }

    #[tokio::test]
    async fn test_insert_and_label_feature() {
        let mut conn = init_db(":memory:").await.unwrap();

        // Insert feature for time range 11:00 - 11:05
        let feature = insert_feature(
            &mut conn,
            1704067200,
            1704067500,
            50000.0,
            vec![0.01, 0.02, -0.01],
            3,
        )
        .await
        .unwrap();
        assert!(feature.id > 0);

        // Label with end price (price went up)
        let target = apply_label_with_price(&mut conn, feature.id, 50100.0, 1704067600)
            .await
            .unwrap();
        assert_eq!(target, 1); // Price went up

        let features = fetch_labeled_features(&mut conn).await.unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].target, Some(1));
        assert_eq!(features[0].end_price, Some(50100.0));
    }
}