//! Web server module providing authenticated REST API access to database data
//!
//! Endpoints (all require Bearer token authentication):
//! - GET /api/features - List features (optional ?limit=N&offset=M&labeled=true/false)
//! - GET /api/features/:id - Get single feature
//! - GET /api/features/stats - Get feature statistics
//! - GET /api/trades - List trades (optional ?limit=N&offset=M)
//! - GET /api/trades/:id - Get single trade
//! - GET /api/model - Get model metadata
//! - GET /health - Health check (no auth required)

use axum::{
    extract::{rejection::QueryRejection, Path, Query, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
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
}

/// Shared application state
#[derive(Debug, Clone)]
pub struct AppState {
    pub db: Arc<RwLock<SqliteConnection>>,
    pub auth_token: String,
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
        get_model_metadata,
    ),
    components(
        schemas(
            ApiError,
            FeatureStats,
            HealthResponse,
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
    };

    // Build API routes with auth
    let api_routes = Router::new()
        .route("/features", get(list_features))
        .route("/features/stats", get(get_feature_stats))
        .route("/features/{id}", get(get_feature))
        .route("/trades", get(list_trades))
        .route("/trades/{id}", get(get_trade))
        .route("/model", get(get_model_metadata))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth_middleware));

    // Health check (no auth)
    let health_route = Router::new().route("/health", get(health_check));

    // OpenAPI spec and Swagger UI
    let openapi = ApiDoc::openapi();
    let swagger = SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", openapi.clone());

    let app = Router::new()
        .route("/api", get(|| async { "Ololon API" }))
        .nest("/api", api_routes)
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
    }));

    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    let mut conn = state.db.write().await;

    let trades: Vec<db::Trade> = ormlite::query_as(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ? OFFSET ?",
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(&mut *conn)
    .await
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