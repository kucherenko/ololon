//! Web server module providing authenticated REST API access to database data
//!
//! Endpoints (all require Bearer token authentication):
//! - GET /api/features - List features (optional ?limit=N&offset=M&labeled=true/false)
//! - GET /api/features/:id - Get single feature
//! - GET /api/features/stats - Get feature statistics
//! - GET /api/trades - List trades (optional ?limit=N&offset=M)
//! - GET /api/trades/:id - Get single trade
//! - GET /api/model - Get model metadata
//! - GET /api/logs/stream - Server-Sent Events stream of bot logs
//! - GET /api/logs - Get recent log entries (optional ?lines=N)
//! - GET /health - Health check (no auth required)

use axum::{
    extract::{rejection::QueryRejection, Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response, sse::Event},
    routing::{get, post},
    Json, Router,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use futures_util::StreamExt;
use ormlite::sqlite::SqliteConnection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::db;

/// Server configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub db_path: String,
    pub auth_token: String,
    pub bind_address: String,
    /// Path to the log file (default: logs/ololon.log)
    pub log_path: String,
}

/// Shared application state
#[derive(Debug, Clone)]
pub struct AppState {
    pub db: Arc<RwLock<SqliteConnection>>,
    pub auth_token: String,
    pub log_path: String,
}

/// Query parameters for feature listing
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct FeatureQuery {
    /// Maximum number of results to return (default: 100, max: 1000)
    pub limit: Option<i64>,
    /// Number of results to skip for pagination
    pub offset: Option<i64>,
    /// Filter by labeled status: true = labeled, false = unlabeled, null = all
    pub labeled: Option<bool>,
}

/// Query parameters for trade listing
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct TradeQuery {
    /// Maximum number of results to return (default: 100, max: 1000)
    pub limit: Option<i64>,
    /// Number of results to skip for pagination
    pub offset: Option<i64>,
    /// Filter by dry-run status: true = dry-run only, false = real trades only, null = all
    pub dry_run: Option<bool>,
}

/// Query parameters for settling a trade
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct SettleQuery {
    /// Settlement price (the actual outcome price at settlement time)
    pub settlement_price: f64,
}

/// API error response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct ApiError {
    /// Error message
    pub error: String,
}

/// Feature statistics response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct FeatureStats {
    /// Total number of features
    pub total_count: i64,
    /// Number of labeled features
    pub labeled_count: i64,
    /// Number of unlabeled features
    pub unlabeled_count: i64,
}

/// Health check response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct HealthResponse {
    /// Server status
    pub status: String,
}

/// Trade statistics response
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct TradeStats {
    /// Total number of trades
    pub total_count: i64,
    /// Number of dry-run trades
    pub dry_run_count: i64,
    /// Number of real trades
    pub real_trade_count: i64,
    /// Number of settled dry-run trades
    pub settled_count: i64,
    /// Total profit/loss from settled dry-run trades
    pub total_profit_loss: f64,
}

/// OpenAPI specification
#[derive(OpenApi)]
#[openapi(
    paths(
        health_check,
        list_features,
        get_feature,
        get_feature_stats,
        list_trades,
        get_trade,
        settle_trade,
        get_trade_stats,
        get_model_metadata,
        get_logs,
        stream_logs,
    ),
    components(
        schemas(
            ApiError,
            FeatureStats,
            TradeStats,
            HealthResponse,
            LogEntry,
            db::Feature,
            db::Trade,
            db::ModelMetadata,
        )
    ),
    modifiers(&SecurityAddon)
)]
pub struct ApiDoc;

/// Security scheme for Bearer token authentication
struct SecurityAddon;

impl utoipa::Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = &mut openapi.components {
            components.add_security_scheme(
                "bearer_auth",
                utoipa::openapi::security::SecurityScheme::Http(
                    utoipa::openapi::security::Http::new(utoipa::openapi::security::HttpAuthScheme::Bearer),
                ),
            );
        }
    }
}

/// Run the web server
pub async fn run(config: Config) -> anyhow::Result<()> {
    let db = db::init_db(&config.db_path).await?;
    let state = AppState {
        db: Arc::new(RwLock::new(db)),
        auth_token: config.auth_token,
        log_path: config.log_path,
    };

    // Build API routes with auth
    let api_routes = Router::new()
        .route("/features", get(list_features))
        .route("/features/stats", get(get_feature_stats))
        .route("/features/{id}", get(get_feature))
        .route("/trades", get(list_trades))
        .route("/trades/{id}", get(get_trade))
        .route("/trades/{id}/settle", post(settle_trade))
        .route("/trades/stats", get(get_trade_stats))
        .route("/model", get(get_model_metadata))
        .route("/logs", get(get_logs))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // SSE stream with its own token validation (EventSource doesn't support headers)
    let logs_stream_route = Router::new()
        .route("/api/logs/stream", get(stream_logs))
        .with_state(state.clone());

    // Health check (no auth)
    let health_route = Router::new().route("/health", get(health_check));

    // OpenAPI spec and Swagger UI
    let openapi = ApiDoc::openapi();
    let swagger = SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", openapi.clone());

    let app = Router::new()
        .route("/api", get(|| async { "Ololon API" }))
        .nest("/api", api_routes)
        .merge(logs_stream_route)
        .merge(health_route)
        .merge(swagger)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&config.bind_address).await?;
    tracing::info!("Server listening on {}", config.bind_address);
    tracing::info!("Swagger UI available at http://{}/swagger-ui", config.bind_address);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Auth middleware - validates Bearer token
async fn auth_middleware(
    State(state): State<AppState>,
    TypedHeader(auth): TypedHeader<Authorization<Bearer>>,
    request: Request,
    next: Next,
) -> Result<Response, ApiError> {
    let token = auth.token();

    if token != state.auth_token {
        return Err(ApiError {
            error: "Invalid or missing authentication token".to_string(),
        });
    }

    Ok(next.run(request).await)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.error.as_str() {
            "Invalid or missing authentication token" => StatusCode::UNAUTHORIZED,
            "Feature not found" | "Trade not found" | "Model metadata not found" => {
                StatusCode::NOT_FOUND
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(self)).into_response()
    }
}

/// Health check endpoint
///
/// Returns server health status. No authentication required.
#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Server is healthy", body = HealthResponse)
    )
)]
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// List features with optional filtering and pagination
///
/// Returns a list of feature records. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/features",
    params(
        FeatureQuery
    ),
    responses(
        (status = 200, description = "List of features", body = Vec<db::Feature>),
        (status = 401, description = "Unauthorized", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn list_features(
    State(state): State<AppState>,
    query: Result<Query<FeatureQuery>, QueryRejection>,
) -> Result<Json<Vec<db::Feature>>, ApiError> {
    let Query(params) = query.unwrap_or(Query(FeatureQuery {
        limit: None,
        offset: None,
        labeled: None,
    }));

    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    let mut conn = state.db.write().await;

    let features = match params.labeled {
        Some(true) => {
            ormlite::query_as(
                r#"
                SELECT * FROM features 
                WHERE target IS NOT NULL 
                ORDER BY time_range_start DESC 
                LIMIT ? OFFSET ?
                "#,
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
        Some(false) => {
            ormlite::query_as(
                r#"
                SELECT * FROM features 
                WHERE target IS NULL 
                ORDER BY time_range_start DESC 
                LIMIT ? OFFSET ?
                "#,
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
        None => {
            ormlite::query_as(
                r#"
                SELECT * FROM features 
                ORDER BY time_range_start DESC 
                LIMIT ? OFFSET ?
                "#,
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
    }
    .map_err(|e| ApiError {
        error: format!("Database error: {}", e),
    })?;

    Ok(Json(features))
}

/// Get a single feature by ID
///
/// Returns a specific feature record. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/features/{id}",
    params(
        ("id" = i64, Path, description = "Feature ID")
    ),
    responses(
        (status = 200, description = "Feature found", body = db::Feature),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 404, description = "Feature not found", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_feature(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<db::Feature>, ApiError> {
    let mut conn = state.db.write().await;

    let feature: Option<db::Feature> = ormlite::query_as("SELECT * FROM features WHERE id = ?")
        .bind(id)
        .fetch_optional(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    match feature {
        Some(f) => Ok(Json(f)),
        None => Err(ApiError {
            error: "Feature not found".to_string(),
        }),
    }
}

/// Get feature statistics
///
/// Returns aggregate statistics about features. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/features/stats",
    responses(
        (status = 200, description = "Feature statistics", body = FeatureStats),
        (status = 401, description = "Unauthorized", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_feature_stats(State(state): State<AppState>) -> Result<Json<FeatureStats>, ApiError> {
    let mut conn = state.db.write().await;

    let total: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM features")
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let labeled: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM features WHERE target IS NOT NULL")
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let stats = FeatureStats {
        total_count: total.0,
        labeled_count: labeled.0,
        unlabeled_count: total.0 - labeled.0,
    };

    Ok(Json(stats))
}

/// List trades with optional pagination
///
/// Returns a list of trade records. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/trades",
    params(
        TradeQuery
    ),
    responses(
        (status = 200, description = "List of trades", body = Vec<db::Trade>),
        (status = 401, description = "Unauthorized", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn list_trades(
    State(state): State<AppState>,
    query: Result<Query<TradeQuery>, QueryRejection>,
) -> Result<Json<Vec<db::Trade>>, ApiError> {
    let Query(params) = query.unwrap_or(Query(TradeQuery {
        limit: None,
        offset: None,
        dry_run: None,
    }));

    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    let mut conn = state.db.write().await;

    let trades: Vec<db::Trade> = match params.dry_run {
        Some(true) => {
            ormlite::query_as(
                "SELECT * FROM trades WHERE is_dry_run = 1 ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
        Some(false) => {
            ormlite::query_as(
                "SELECT * FROM trades WHERE is_dry_run = 0 ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
        None => {
            ormlite::query_as(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            )
            .bind(limit)
            .bind(offset)
            .fetch_all(&mut *conn)
            .await
        }
    }
    .map_err(|e| ApiError {
        error: format!("Database error: {}", e),
    })?;

    Ok(Json(trades))
}

/// Get a single trade by ID
///
/// Returns a specific trade record. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/trades/{id}",
    params(
        ("id" = i64, Path, description = "Trade ID")
    ),
    responses(
        (status = 200, description = "Trade found", body = db::Trade),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 404, description = "Trade not found", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_trade(
    State(state): State<AppState>,
    Path(id): Path<i64>,
) -> Result<Json<db::Trade>, ApiError> {
    let mut conn = state.db.write().await;

    let trade: Option<db::Trade> = ormlite::query_as("SELECT * FROM trades WHERE id = ?")
        .bind(id)
        .fetch_optional(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    match trade {
        Some(t) => Ok(Json(t)),
        None => Err(ApiError {
            error: "Trade not found".to_string(),
        }),
    }
}

/// Settle a dry-run trade with the actual market outcome
///
/// Calculates profit/loss based on the settlement price and updates the trade record.
/// Requires Bearer token authentication.
#[utoipa::path(
    post,
    path = "/api/trades/{id}/settle",
    params(
        ("id" = i64, Path, description = "Trade ID"),
        SettleQuery
    ),
    responses(
        (status = 200, description = "Trade settled successfully", body = db::Trade),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 404, description = "Trade not found", body = ApiError),
        (status = 400, description = "Cannot settle non-dry-run trade", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn settle_trade(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    query: Query<SettleQuery>,
) -> Result<Json<db::Trade>, ApiError> {
    let mut conn = state.db.write().await;

    // Check if trade exists and is a dry-run trade
    let trade: Option<db::Trade> = ormlite::query_as("SELECT * FROM trades WHERE id = ?")
        .bind(id)
        .fetch_optional(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let trade = match trade {
        Some(t) => t,
        None => {
            return Err(ApiError {
                error: "Trade not found".to_string(),
            });
        }
    };

    // Only allow settling dry-run trades
    if !trade.is_dry_run {
        return Err(ApiError {
            error: "Cannot settle non-dry-run trade".to_string(),
        });
    }

    if trade.settled {
        return Err(ApiError {
            error: "Trade already settled".to_string(),
        });
    }

    // Calculate profit/loss
    let profit_loss = if trade.outcome == "YES" {
        if query.settlement_price >= 1.0 {
            trade.trade_size * (1.0 - trade.avg_price)
        } else {
            -trade.trade_size * trade.avg_price
        }
    } else if query.settlement_price < 1.0 {
        trade.trade_size * (1.0 - trade.avg_price)
    } else {
        -trade.trade_size * trade.avg_price
    };

    // Update the trade
    ormlite::query(
        "UPDATE trades SET settled = 1, settlement_price = ?, profit_loss = ? WHERE id = ?"
    )
    .bind(query.settlement_price)
    .bind(profit_loss)
    .bind(id)
    .execute(&mut *conn)
    .await
    .map_err(|e| ApiError {
        error: format!("Database error: {}", e),
    })?;

    // Fetch updated trade
    let updated_trade: db::Trade = ormlite::query_as("SELECT * FROM trades WHERE id = ?")
        .bind(id)
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    Ok(Json(updated_trade))
}

/// Get trade statistics
///
/// Returns aggregate statistics about trades including dry-run and real trades.
/// Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/trades/stats",
    responses(
        (status = 200, description = "Trade statistics", body = TradeStats),
        (status = 401, description = "Unauthorized", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_trade_stats(State(state): State<AppState>) -> Result<Json<TradeStats>, ApiError> {
    let mut conn = state.db.write().await;

    let total: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM trades")
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let dry_run: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM trades WHERE is_dry_run = 1")
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let settled: (i64,) = ormlite::query_as("SELECT COUNT(*) FROM trades WHERE is_dry_run = 1 AND settled = 1")
        .fetch_one(&mut *conn)
        .await
        .map_err(|e| ApiError {
            error: format!("Database error: {}", e),
        })?;

    let profit_loss: (Option<f64>,) = ormlite::query_as(
        "SELECT SUM(profit_loss) FROM trades WHERE is_dry_run = 1 AND settled = 1"
    )
    .fetch_one(&mut *conn)
    .await
    .map_err(|e| ApiError {
        error: format!("Database error: {}", e),
    })?;

    let stats = TradeStats {
        total_count: total.0,
        dry_run_count: dry_run.0,
        real_trade_count: total.0 - dry_run.0,
        settled_count: settled.0,
        total_profit_loss: profit_loss.0.unwrap_or(0.0),
    };

    Ok(Json(stats))
}

/// Get model metadata
///
/// Returns the current model's training metadata. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/model",
    responses(
        (status = 200, description = "Model metadata", body = db::ModelMetadata),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 404, description = "Model metadata not found", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_model_metadata(
    State(state): State<AppState>,
) -> Result<Json<db::ModelMetadata>, ApiError> {
    let mut conn = state.db.write().await;

    let metadata: Option<db::ModelMetadata> =
        ormlite::query_as("SELECT * FROM model_metadata WHERE id = 1")
            .fetch_optional(&mut *conn)
            .await
            .map_err(|e| ApiError {
                error: format!("Database error: {}", e),
            })?;

    match metadata {
        Some(m) => Ok(Json(m)),
        None => Err(ApiError {
            error: "Model metadata not found".to_string(),
        }),
    }
}

/// Query parameters for log listing
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct LogQuery {
    /// Number of lines to return from end of file (default: 100, max: 1000)
    pub lines: Option<usize>,
    /// Filter by command name (collect, train, trade, server, validate)
    pub command: Option<String>,
}

/// Log entry parsed from JSON log file
#[derive(Debug, Serialize, utoipa::ToSchema)]
pub struct LogEntry {
    /// Raw JSON log line
    pub raw: String,
    /// Command that generated this log (extracted from span)
    pub command: Option<String>,
}

/// Get recent log entries
///
/// Returns recent log lines from the bot log file. Requires Bearer token authentication.
#[utoipa::path(
    get,
    path = "/api/logs",
    params(
        LogQuery
    ),
    responses(
        (status = 200, description = "Recent log entries", body = Vec<LogEntry>),
        (status = 401, description = "Unauthorized", body = ApiError),
        (status = 500, description = "Failed to read logs", body = ApiError)
    ),
    security(
        ("bearer_auth" = [])
    )
)]
async fn get_logs(
    State(state): State<AppState>,
    query: Result<Query<LogQuery>, QueryRejection>,
) -> Result<Json<Vec<LogEntry>>, ApiError> {
    let Query(params) = query.unwrap_or(Query(LogQuery { lines: None, command: None }));
    let lines = params.lines.unwrap_or(100).min(1000);
    let command_filter = params.command.as_ref().map(|c| c.to_lowercase());

    // Find the most recent log file (handles daily rotation)
    let log_path = std::path::Path::new(&state.log_path);
    let log_file = if log_path.exists() {
        log_path.to_path_buf()
    } else {
        // Try to find any log file in the logs directory
        let logs_dir = log_path.parent().unwrap_or(std::path::Path::new("logs"));
        let mut log_files: Vec<_> = std::fs::read_dir(logs_dir)
            .map_err(|e| ApiError {
                error: format!("Failed to read logs directory: {}", e),
            })?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("ololon.log")
            })
            .collect();
        
        log_files.sort_by_key(|e| e.file_name());
        log_files
            .pop()
            .map(|e| e.path())
            .ok_or_else(|| ApiError {
                error: "No log files found".to_string(),
            })?
    };

    // Read and return last N lines
    let content = std::fs::read_to_string(&log_file).map_err(|e| ApiError {
        error: format!("Failed to read log file: {}", e),
    })?;

    let all_lines: Vec<&str> = content.lines().collect();
    
    // Helper to extract command from JSON log (can be in command_name or span.command_name)
    let extract_command = |line: &str| -> Option<String> {
        serde_json::from_str::<serde_json::Value>(line).ok().and_then(|v| {
            // Check command_name directly first
            if let Some(c) = v.get("command_name").and_then(|c| c.as_str()) {
                return Some(c.to_string());
            }
            // Check span.command_name
            v.get("span")
                .and_then(|s| s.get("command_name"))
                .and_then(|c| c.as_str())
                .map(|c| c.to_string())
        })
    };

    // Filter and collect entries
    let entries: Vec<LogEntry> = all_lines
        .iter()
        .rev() // Start from end
        .filter_map(|line| {
            let command = extract_command(line);
            
            // Apply command filter if specified
            if let Some(ref filter) = command_filter {
                if command.as_ref().map(|c| c.to_lowercase()) != Some(filter.to_lowercase()) {
                    return None;
                }
            }
            
            Some(LogEntry {
                raw: line.to_string(),
                command,
            })
        })
        .take(lines)
        .collect();

    // Reverse back to chronological order
    Ok(Json(entries.into_iter().rev().collect()))
}

/// Query parameters for SSE stream (token for auth since EventSource doesn't support headers)
#[derive(Debug, Deserialize, utoipa::IntoParams)]
pub struct StreamQuery {
    /// Authentication token (alternative to Bearer header for SSE)
    pub token: Option<String>,
    /// Filter by command name (collect, train, trade, server, validate)
    pub command: Option<String>,
}

/// Stream log entries via Server-Sent Events
///
/// Streams new log entries as they are written to the log file. Requires authentication.
/// Accepts token via query param since EventSource doesn't support custom headers.
#[utoipa::path(
    get,
    path = "/api/logs/stream",
    params(
        StreamQuery
    ),
    responses(
        (status = 200, description = "SSE stream of log entries", content_type = "text/event-stream"),
        (status = 401, description = "Unauthorized", body = ApiError)
    )
)]
async fn stream_logs(
    State(state): State<AppState>,
    Query(params): Query<StreamQuery>,
) -> Result<Response, ApiError> {
    // Validate token from query param (EventSource doesn't support headers)
    let token = params.token.unwrap_or_default();
    if token != state.auth_token {
        return Err(ApiError {
            error: "Invalid or missing authentication token".to_string(),
        });
    }
    
    let log_path = state.log_path.clone();
    let command_filter = params.command.map(|c| c.to_lowercase());
    
    // Create a stream that tails the log file
    let stream = async_stream::stream! {
        // Use tokio::fs for async file operations
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::fs::File;
        
        // Helper to extract command from JSON log (can be in command_name or span.command_name)
        fn extract_command(line: &str) -> Option<String> {
            serde_json::from_str::<serde_json::Value>(line).ok().and_then(|v| {
                // Check command_name directly first
                if let Some(c) = v.get("command_name").and_then(|c| c.as_str()) {
                    return Some(c.to_string());
                }
                // Check span.command_name
                v.get("span")
                    .and_then(|s| s.get("command_name"))
                    .and_then(|c| c.as_str())
                    .map(|c| c.to_string())
            })
        }
        
        // Find the log file (handle rotation)
        let log_file = if std::path::Path::new(&log_path).exists() {
            log_path.clone()
        } else {
            let logs_dir = std::path::Path::new(&log_path)
                .parent()
                .unwrap_or(std::path::Path::new("logs"));
            
            let mut log_files: Vec<_> = match std::fs::read_dir(logs_dir) {
                Ok(entries) => entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.file_name().to_string_lossy().starts_with("ololon.log")
                    })
                    .collect(),
                Err(_) => {
                    yield Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "No log files found",
                    ));
                    return;
                }
            };
            
            log_files.sort_by_key(|e| e.file_name());
            match log_files.pop() {
                Some(e) => e.path().to_string_lossy().to_string(),
                None => {
                    yield Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        "No log files found",
                    ));
                    return;
                }
            }
        };

        // Initial read of existing content (last 50 lines, filtered)
        let mut file = match File::open(&log_file).await {
            Ok(f) => f,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        
        // Get initial file position (start from end for recent logs)
        let initial_metadata = match file.metadata().await {
            Ok(m) => m,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        let file_size = initial_metadata.len();
        
        // Read all lines first to get recent ones (filtered)
        let reader = BufReader::new(&mut file);
        let mut lines = reader.lines();
        let mut recent_lines: Vec<String> = Vec::new();
        let max_recent = 50;
        
        while let Ok(Some(line)) = lines.next_line().await {
            // Apply command filter
            let command = extract_command(&line);
            let matches = match &command_filter {
                Some(filter) => command.as_ref().map(|c| c.to_lowercase()) == Some(filter.clone()),
                None => true,
            };
            
            if matches {
                recent_lines.push(line);
                if recent_lines.len() > max_recent {
                    recent_lines.remove(0);
                }
            }
        }
        
        // Send recent lines first
        for line in recent_lines {
            yield Ok(Event::default().data(line));
        }

        // Track the current position in the file
        let mut last_pos = file_size;

        // Now tail for new lines using position tracking
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Check file size
            let current_metadata = match std::fs::metadata(&log_file) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let current_size = current_metadata.len();

            // File was rotated (size reset or file changed) - send reconnect notice
            if current_size < last_pos {
                yield Ok(Event::default().data("--- Log file rotated ---"));
                last_pos = 0;
            }

            // Read new content from last position
            if current_size > last_pos {
                // Seek to last position and read new content
                match tokio::fs::OpenOptions::new()
                    .read(true)
                    .open(&log_file)
                    .await
                {
                    Ok(mut new_file) => {
                        use tokio::io::AsyncSeekExt;
                        if new_file.seek(std::io::SeekFrom::Start(last_pos)).await.is_err() {
                            continue;
                        }
                        
                        let reader = BufReader::new(new_file);
                        let mut lines = reader.lines();
                        
                        while let Ok(Some(line)) = lines.next_line().await {
                            // Apply command filter
                            let command = extract_command(&line);
                            let matches = match &command_filter {
                                Some(filter) => command.as_ref().map(|c| c.to_lowercase()) == Some(filter.clone()),
                                None => true,
                            };
                            
                            if matches {
                                yield Ok(Event::default().data(line));
                            }
                        }
                        
                        last_pos = current_size;
                    }
                    Err(_) => continue,
                }
            }
        }
    };

    // Convert errors to events
    let stream = stream.map(|result: Result<Event, std::io::Error>| match result {
        Ok(event) => Ok::<_, std::convert::Infallible>(event),
        Err(e) => Ok(Event::default().data(format!("Error: {}", e))),
    });

    Ok(
        axum::response::sse::Sse::new(stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(std::time::Duration::from_secs(15))
                    .text("ping"),
            )
            .into_response(),
    )
}