//! Live trading module - runs inference and executes trades on Polymarket

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, LstmConfig};
use burn::record::{FullPrecisionSettings, JsonGzFileRecorder};
use burn::tensor::{backend::Backend, Tensor, TensorData};
use chrono::Utc;
use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use k256::ecdsa::signature::DigestSigner;
use k256::ecdsa::{Signature, SigningKey};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use sha3::{Digest, Keccak256};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

use crate::db;

type MyBackend = NdArray<f32>;

/// Configuration for the trade subcommand
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TradeConfig {
    pub db_path: String,
    pub model_path: String,
    pub ws_url: String,
    pub symbol: String,
    pub gamma_api_url: String,
    pub clob_api_url: String,
    pub min_edge: f64,
    pub trade_size: f64,
    pub hidden_size: usize,
    #[allow(dead_code)]
    pub num_layers: usize,
    pub private_key: Option<String>,
    pub api_key: Option<String>,
    pub api_secret: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PolymarketEvent {
    pub markets: Vec<PolymarketMarket>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PolymarketMarket {
    pub id: String,
    pub condition_id: String,
    pub outcome_prices: Vec<String>,
    pub active: bool,
    pub closed: bool,
    #[serde(default)]
    pub tokens: Vec<PolymarketToken>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PolymarketToken {
    pub token_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OrderRequest {
    pub market: String,
    pub asset_id: String,
    pub side: String,
    pub size: String,
    pub price: String,
    pub nonce: String,
    pub expiration: i64,
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn sign_eip712(signing_key: &SigningKey, message_hash: &[u8; 32]) -> Result<Signature> {
    Ok(signing_key.sign_digest(sha2::Sha256::new_with_prefix(message_hash)))
}

fn derive_address(signing_key: &SigningKey) -> Result<String> {
    let verifying_key = signing_key.verifying_key();
    let pubkey_bytes = verifying_key.to_encoded_point(false);
    let pubkey = pubkey_bytes.as_bytes();
    let hash = keccak256(&pubkey[1..]);
    Ok(format!("0x{}", hex::encode(&hash[12..])))
}

pub struct PolymarketClient {
    http: Client,
    gamma_api_url: String,
    clob_api_url: String,
    signing_key: Option<SigningKey>,
    address: Option<String>,
    api_key: Option<String>,
    api_secret: Option<String>,
}

impl PolymarketClient {
    pub fn new(
        gamma_api_url: String, clob_api_url: String,
        private_key: Option<String>, api_key: Option<String>, api_secret: Option<String>,
    ) -> Result<Self> {
        let (signing_key, address) = if let Some(pk) = &private_key {
            let pk_hex = pk.strip_prefix("0x").unwrap_or(pk);
            let pk_bytes = hex::decode(pk_hex).context("Failed to decode private key")?;
            let sk = SigningKey::from_bytes((&pk_bytes[..]).into()).context("Failed to create signing key")?;
            let addr = derive_address(&sk)?;
            (Some(sk), Some(addr))
        } else { (None, None) };

        Ok(Self {
            http: Client::builder().timeout(Duration::from_secs(10)).build()?,
            gamma_api_url, clob_api_url, signing_key, address, api_key, api_secret,
        })
    }

    pub async fn find_btc_updown_market(&self) -> Result<Option<PolymarketMarket>> {
        let url = format!("{}/events?slug=btc-updown-5m", self.gamma_api_url);
        let response = self.http.get(&url).send().await.context("Gamma API failed")?;
        if !response.status().is_success() { return Ok(None); }
        let events: Vec<PolymarketEvent> = response.json().await.context("Parse failed")?;
        for event in events {
            for market in event.markets {
                if market.active && !market.closed { return Ok(Some(market)); }
            }
        }
        Ok(None)
    }

    fn compute_l2_signature(&self, timestamp: i64, method: &str, path: &str, body: &str) -> Result<String> {
        let secret = self.api_secret.as_ref().context("API secret required")?;
        let msg = format!("{}{}{}{}", timestamp, method, path, body);
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())?;
        mac.update(msg.as_bytes());
        Ok(hex::encode(mac.finalize().into_bytes()))
    }

    pub async fn place_order(&self, market_id: &str, token_id: &str, side: &str, size: f64, price: f64) -> Result<String> {
        let sk = self.signing_key.as_ref().context("Wallet required")?;
        let addr = self.address.as_ref().context("Address required")?;
        let key = self.api_key.as_ref().context("API key required")?;

        let ts = Utc::now().timestamp();
        let nonce = uuid::Uuid::new_v4().to_string();
        let order = OrderRequest {
            market: market_id.to_string(), asset_id: token_id.to_string(), side: side.to_string(),
            size: format!("{:.6}", size), price: format!("{:.6}", price), nonce: nonce.clone(), expiration: ts + 86400,
        };

        let hash = keccak256(serde_json::to_string(&order)?.as_bytes());
        let sig = sign_eip712(sk, &hash)?;
        let sig_hex = format!("0x{}", hex::encode(sig.to_bytes()));
        let body = serde_json::to_string(&order)?;
        let l2_sig = self.compute_l2_signature(ts, "POST", "/order", &body)?;

        let resp = self.http.post(format!("{}/order", self.clob_api_url))
            .header("POLY-ADDRESS", addr).header("POLY-API-KEY", key)
            .header("POLY-TIMESTAMP", ts.to_string()).header("POLY-SIGNATURE", l2_sig)
            .header("POLY-NONCE", &nonce)
            .json(&serde_json::json!({"order": order, "signature": sig_hex, "signature_type": "EIP712"}))
            .send().await.context("Order failed")?;

        if !resp.status().is_success() {
            anyhow::bail!("Order failed: {}", resp.text().await.unwrap_or_default());
        }
        resp.json().await.context("Parse failed")
    }
}

#[derive(Debug, Deserialize)]
struct BinanceTrade { #[serde(rename = "p")] price: String }

struct InferenceWindow { prices: VecDeque<f64>, max_size: usize }

impl InferenceWindow {
    fn new(max_size: usize) -> Self { Self { prices: VecDeque::with_capacity(max_size), max_size } }
    fn push(&mut self, price: f64) { if self.prices.len() >= self.max_size { self.prices.pop_front(); } self.prices.push_back(price); }
    fn is_full(&self) -> bool { self.prices.len() >= self.max_size }

    fn compute_normalized_log_returns(&self) -> Option<Vec<f32>> {
        if self.prices.len() < 2 { return None; }
        let returns: Vec<f64> = self.prices.iter().zip(self.prices.iter().skip(1)).map(|(p, c)| (c / p).ln()).collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt();
        if std < 1e-10 { Some(returns.iter().map(|_| 0.0f32).collect()) }
        else { Some(returns.iter().map(|r| ((r - mean) / std) as f32).collect()) }
    }
}

#[derive(Config, Debug)]
pub struct PricePredictorConfig { pub input_size: usize, pub hidden_size: usize, pub d_output: usize }

impl PricePredictorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PricePredictor<B> {
        let lstm = LstmConfig::new(self.input_size, self.hidden_size, false).init(device);
        let fc = LinearConfig::new(self.hidden_size, self.d_output).init(device);
        PricePredictor { lstm, fc, hidden_size: self.hidden_size }
    }
}

#[derive(burn::module::Module, Debug)]
pub struct PricePredictor<B: Backend> { lstm: burn::nn::Lstm<B>, fc: Linear<B>, hidden_size: usize }

impl<B: Backend> PricePredictor<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (hidden, _) = self.lstm.forward(input, None);
        let dims = hidden.dims();
        let last: Tensor<B, 2> = hidden.slice([0..dims[0], dims[1]-1..dims[1], 0..dims[2]]).reshape([dims[0], self.hidden_size]);
        burn::tensor::activation::sigmoid(self.fc.forward(last)).reshape([dims[0], 1, 1])
    }
}

pub struct PriceBatcher<B: Backend> { device: B::Device, #[allow(dead_code)] input_size: usize }

impl<B: Backend> PriceBatcher<B> {
    pub fn new(device: B::Device, input_size: usize) -> Self { Self { device, input_size } }
}

pub struct PriceBatch<B: Backend> { pub features: Tensor<B, 3>, #[allow(dead_code)] pub targets: Tensor<B, 1> }

impl<B: Backend> PriceBatcher<B> {
    pub fn batch(&self, features: Vec<f32>, input_size: usize) -> PriceBatch<B> {
        PriceBatch {
            features: Tensor::from_data(TensorData::new(features, [1, 1, input_size]), &self.device),
            targets: Tensor::from_data(TensorData::new(vec![0.0f32], [1]), &self.device),
        }
    }
}

pub async fn run(config: TradeConfig) -> Result<()> {
    info!(model_path = %config.model_path, "Starting trading bot");

    if config.private_key.is_none() || config.api_key.is_none() || config.api_secret.is_none() {
        warn!("Missing credentials - inference-only mode");
    }

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let conn = db::init_db(&config.db_path)?;
    let window_size: usize = conn.query_row("SELECT window_size FROM model_metadata WHERE id = 1", [], |r| r.get(0)).unwrap_or(59);

    let model_config = PricePredictorConfig::new(window_size, config.hidden_size, 1);
    let model = model_config.init::<MyBackend>(&device);
    let recorder = JsonGzFileRecorder::<FullPrecisionSettings>::new();
    let model = model.load_file(&config.model_path, &recorder, &device).context("Failed to load model")?;

    let poly = PolymarketClient::new(config.gamma_api_url.clone(), config.clob_api_url.clone(),
        config.private_key.clone(), config.api_key.clone(), config.api_secret.clone())?;

    let window = Arc::new(RwLock::new(InferenceWindow::new(window_size + 1)));
    let last_trade = Arc::new(RwLock::new(0i64));
    let mut interval = interval(Duration::from_secs(30));

    let ws_url = format!("{}/{}@aggTrade", config.ws_url, config.symbol.to_lowercase());
    let (ws, _) = connect_async(&ws_url).await.context("WebSocket failed")?;
    let (_, mut read) = ws.split();

    loop {
        tokio::select! {
            msg = read.next() => match msg {
                Some(Ok(Message::Text(t))) => {
                    if let Ok(trade) = serde_json::from_str::<BinanceTrade>(&t) {
                        if let Ok(p) = trade.price.parse() { window.write().await.push(p); }
                    }
                }
                Some(Ok(Message::Close(_))) | None => break,
                _ => {}
            },
            _ = interval.tick() => {
                let w = window.read().await;
                if !w.is_full() { continue; }
                let features = w.compute_normalized_log_returns();
                drop(w);
                let Some(feat) = features else { continue; };

                let input_len = feat.len();
                let batcher = PriceBatcher::<MyBackend>::new(device.clone(), input_len);
                let batch = batcher.batch(feat, input_len);
                let pred = model.forward(batch.features);
                let prob = pred.to_data().as_slice::<f32>().map(|s| s[0]).unwrap_or(0.5);

                info!(prob_up = format!("{:.4}", prob), "Prediction");

                if let Err(e) = trade_logic(&config, &poly, prob, last_trade.clone()).await {
                    error!(error = ?e, "Trade error");
                }
            }
        }
    }
    Ok(())
}

async fn trade_logic(cfg: &TradeConfig, poly: &PolymarketClient, prob: f32, last: Arc<RwLock<i64>>) -> Result<()> {
    let now = Utc::now().timestamp();
    if *last.read().await > now - 60 { return Ok(()); }

    let mkt = match poly.find_btc_updown_market().await? { Some(m) => m, None => { warn!("No market"); return Ok(()); } };
    let yes: f64 = mkt.outcome_prices.get(0).and_then(|p| p.parse().ok()).unwrap_or(0.5);
    let no: f64 = mkt.outcome_prices.get(1).and_then(|p| p.parse().ok()).unwrap_or(0.5);
    let tid = mkt.tokens.get(0).map(|t| t.token_id.as_str()).unwrap_or(&mkt.condition_id);

    let (out, edge, price, tok) = if prob as f64 - yes > cfg.min_edge {
        ("YES", prob as f64 - yes, yes, tid.to_string())
    } else if (1.0 - prob as f64) - no > cfg.min_edge {
        let nt = mkt.tokens.get(1).map(|t| t.token_id.as_str()).unwrap_or(&mkt.condition_id);
        ("NO", (1.0 - prob as f64) - no, no, nt.to_string())
    } else { return Ok(()); };

    if cfg.private_key.is_none() { info!(out, edge = format!("{:.4}", edge), "DRY RUN"); return Ok(()); }

    match poly.place_order(&mkt.id, &tok, "BUY", cfg.trade_size, price * 1.01).await {
        Ok(id) => { info!(id, out, "Order"); *last.write().await = now; }
        Err(e) => { error!(error = ?e, "Order failed"); }
    }
    Ok(())
}