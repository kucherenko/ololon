//! Database module for SQLite operations
//!
//! Schema:
//! - `features`: Stores feature vectors computed from 5-minute time ranges
//! - `trades`: Trade history log

use anyhow::Result;
use rusqlite::Connection;

/// Initialize the database schema
pub fn init_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;

    conn.execute_batch(
        r#"
        -- Features table: stores vectors from 5-minute time ranges with delayed labels
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time_range_start INTEGER NOT NULL,    -- Start of 5-min window (Unix timestamp)
            time_range_end INTEGER NOT NULL,      -- End of 5-min window (Unix timestamp)
            start_price REAL NOT NULL,            -- Price at window start
            end_price REAL,                       -- Price at window end (NULL until labeled)
            feature_vector TEXT NOT NULL,         -- JSON array of normalized log returns
            target INTEGER,                       -- NULL until labeled: 1 = price went up, 0 = down
            num_points INTEGER NOT NULL,          -- Number of price points in window
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            labeled_at INTEGER                    -- Timestamp when label was applied
        );

        -- Indexes for efficient queries
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
    )?;

    Ok(conn)
}

/// Feature record for database operations
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FeatureRecord {
    pub id: i64,
    pub time_range_start: i64,
    #[allow(dead_code)]
    pub time_range_end: i64,
    #[allow(dead_code)]
    pub start_price: f64,
    #[allow(dead_code)]
    pub end_price: Option<f64>,
    pub feature_vector: Vec<f64>,
    pub target: Option<i32>,
    pub num_points: i32,
    #[allow(dead_code)]
    pub created_at: i64,
    #[allow(dead_code)]
    pub labeled_at: Option<i64>,
}

/// Trade record for database operations
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

/// Insert a new feature vector (without label - end_price is unknown)
#[allow(dead_code)]
pub fn insert_feature(
    conn: &Connection,
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    feature_vector: &[f64],
    num_points: usize,
) -> Result<i64> {
    let vector_json = serde_json::to_string(feature_vector)?;

    conn.execute(
        "INSERT INTO features (time_range_start, time_range_end, start_price, feature_vector, num_points) 
         VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![time_range_start, time_range_end, start_price, vector_json, num_points as i32],
    )?;

    Ok(conn.last_insert_rowid())
}

/// Apply delayed label to a feature record using the end price
#[allow(dead_code)]
pub fn apply_label_with_price(
    conn: &Connection,
    id: i64,
    end_price: f64,
    labeled_at: i64,
) -> Result<i32> {
    // Get the start price to determine the target
    let start_price: f64 = conn.query_row(
        "SELECT start_price FROM features WHERE id = ?1",
        rusqlite::params![id],
        |row| row.get(0),
    )?;

    let target = if end_price >= start_price { 1 } else { 0 };

    conn.execute(
        "UPDATE features SET target = ?1, end_price = ?2, labeled_at = ?3 WHERE id = ?4",
        rusqlite::params![target, end_price, labeled_at, id],
    )?;

    Ok(target)
}

/// Fetch features ready for labeling (time_range_end has passed)
#[allow(dead_code)]
pub fn fetch_features_ready_for_labeling(
    conn: &Connection,
    current_time: i64,
) -> Result<Vec<FeatureRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, time_range_start, time_range_end, start_price, end_price, feature_vector, 
                target, num_points, created_at, labeled_at
         FROM features 
         WHERE target IS NULL AND time_range_end <= ?1
         ORDER BY time_range_start ASC",
    )?;

    let records = stmt
        .query_map([current_time], |row| {
            let vector_json: String = row.get(5)?;
            let feature_vector: Vec<f64> = serde_json::from_str(&vector_json).unwrap_or_default();
            Ok(FeatureRecord {
                id: row.get(0)?,
                time_range_start: row.get(1)?,
                time_range_end: row.get(2)?,
                start_price: row.get(3)?,
                end_price: row.get(4)?,
                feature_vector,
                target: row.get(6)?,
                num_points: row.get(7)?,
                created_at: row.get(8)?,
                labeled_at: row.get(9)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(records)
}

/// Fetch all labeled features for training
pub fn fetch_labeled_features(conn: &Connection) -> Result<Vec<FeatureRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, time_range_start, time_range_end, start_price, end_price, feature_vector, 
                target, num_points, created_at, labeled_at
         FROM features 
         WHERE target IS NOT NULL
         ORDER BY time_range_start ASC",
    )?;

    let records = stmt
        .query_map([], |row| {
            let vector_json: String = row.get(5)?;
            let feature_vector: Vec<f64> = serde_json::from_str(&vector_json).unwrap_or_default();
            Ok(FeatureRecord {
                id: row.get(0)?,
                time_range_start: row.get(1)?,
                time_range_end: row.get(2)?,
                start_price: row.get(3)?,
                end_price: row.get(4)?,
                feature_vector,
                target: row.get(6)?,
                num_points: row.get(7)?,
                created_at: row.get(8)?,
                labeled_at: row.get(9)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

    Ok(records)
}

/// Count labeled features
#[allow(dead_code)]
pub fn count_labeled_features(conn: &Connection) -> Result<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM features WHERE target IS NOT NULL",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Count total features (labeled + unlabeled)
#[allow(dead_code)]
pub fn count_total_features(conn: &Connection) -> Result<i64> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM features", [], |row| row.get(0))?;
    Ok(count)
}

/// Get the latest time_range_end from features table
#[allow(dead_code)]
pub fn get_latest_window_end(conn: &Connection) -> Result<Option<i64>> {
    let result: Option<i64> =
        conn.query_row("SELECT MAX(time_range_end) FROM features", [], |row| {
            row.get(0)
        })?;
    Ok(result)
}

/// Insert a trade record
#[allow(dead_code)]
pub fn insert_trade(conn: &Connection, trade: &TradeRecord) -> Result<i64> {
    conn.execute(
        "INSERT INTO trades (timestamp, market_id, outcome, predicted_prob, market_prob, 
         edge, trade_size, avg_price, status, order_id, tx_hash, error_message)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            trade.timestamp,
            trade.market_id,
            trade.outcome,
            trade.predicted_prob,
            trade.market_prob,
            trade.edge,
            trade.trade_size,
            trade.avg_price,
            trade.status,
            trade.order_id,
            trade.tx_hash,
            trade.error_message,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Save model metadata after training
pub fn save_model_metadata(
    conn: &Connection,
    model_path: &str,
    epochs: usize,
    final_train_loss: f64,
    final_val_loss: Option<f64>,
    hidden_size: usize,
    num_layers: usize,
    window_duration_secs: i64,
) -> Result<()> {
    let trained_at = chrono::Utc::now().timestamp();
    conn.execute(
        "INSERT OR REPLACE INTO model_metadata 
         (id, model_path, trained_at, epochs, final_train_loss, final_val_loss, hidden_size, num_layers, window_duration_secs)
         VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            model_path,
            trained_at,
            epochs as i32,
            final_train_loss,
            final_val_loss,
            hidden_size as i32,
            num_layers as i32,
            window_duration_secs,
        ],
    )?;
    Ok(())
}

/// Get model metadata
#[allow(dead_code)]
pub fn get_model_metadata(conn: &Connection) -> Result<Option<(String, i64)>> {
    let result = conn.query_row(
        "SELECT model_path, window_duration_secs FROM model_metadata WHERE id = 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    );

    match result {
        Ok(val) => Ok(Some(val)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_init_db() {
        let temp_file = NamedTempFile::new().unwrap();
        let conn = init_db(temp_file.path().to_str().unwrap()).unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('features', 'trades', 'model_metadata')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_insert_and_label_feature() {
        let temp_file = NamedTempFile::new().unwrap();
        let conn = init_db(temp_file.path().to_str().unwrap()).unwrap();

        // Insert feature for time range 11:00 - 11:05
        let id = insert_feature(
            &conn,
            1704067200,
            1704067500,
            50000.0,
            &[0.01, 0.02, -0.01],
            3,
        )
        .unwrap();
        assert!(id > 0);

        // Label with end price (price went up)
        let target = apply_label_with_price(&conn, id, 50100.0, 1704067600).unwrap();
        assert_eq!(target, 1); // Price went up

        let features = fetch_labeled_features(&conn).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].target, Some(1));
        assert_eq!(features[0].end_price, Some(50100.0));
    }
}
