//! Model training module - trains LSTM model on collected labeled data
//!
//! Architecture:
//! 1. Load labeled 5-minute time windows from SQLite
//! 2. Split chronologically into train/validation sets (preserves time order)
//! 3. Define LSTM + FC model with sigmoid output for binary classification
//! 4. Train with Binary Cross-Entropy loss
//! 5. Save model weights in JSON-GZ format

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::config::Config;
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, LstmConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, JsonGzFileRecorder};
use burn::tensor::{activation::sigmoid, Tensor, TensorData};
use burn::tensor::backend::Backend;
use chrono::{TimeZone, Utc};
use tracing::info;

use crate::db;

// Type aliases
type MyBackend = NdArray<f32>;
type MyAutodiff = burn::backend::Autodiff<MyBackend>;

/// Configuration for the train subcommand
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub db_path: String,
    pub model_path: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub validation_split: f64,
}

/// Training sample from a 5-minute time window
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f32>,
    pub target: f32,
    pub time_range_start: i64,
    pub num_points: usize,
}

#[derive(Debug, Clone)]
pub struct PriceBatch<B: Backend> {
    pub features: Tensor<B, 3>,
    pub targets: Tensor<B, 1>,
}

/// LSTM-based binary classification model
#[derive(Module, Debug)]
pub struct PricePredictor<B: Backend> {
    lstm: burn::nn::Lstm<B>,
    fc: Linear<B>,
    hidden_size: usize,
}

/// Configuration for the model
#[derive(Config, Debug)]
pub struct PricePredictorConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl PricePredictorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PricePredictor<B> {
        let lstm = LstmConfig::new(self.input_size, self.hidden_size, false).init(device);
        let fc = LinearConfig::new(self.hidden_size, 1).init(device);
        PricePredictor { lstm, fc, hidden_size: self.hidden_size }
    }
}

impl<B: Backend> PricePredictor<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (hidden_state, _) = self.lstm.forward(input, None);
        let dims = hidden_state.dims();
        let batch = dims[0];
        let seq = dims[1];
        let hidden = dims[2];
        
        // Get last timestep: [batch, seq, hidden] -> [batch, 1, 1]
        let last: Tensor<B, 2> = hidden_state
            .slice([0..batch, seq-1..seq, 0..hidden])
            .reshape([batch, hidden]);
        
        let output = self.fc.forward(last);
        sigmoid(output).reshape([batch, 1, 1])
    }
}

/// Batcher for converting samples to tensors
/// Handles variable-length feature vectors by padding/truncating to fixed input_size
pub struct PriceBatcher<B: Backend> {
    device: B::Device,
    input_size: usize,
}

impl<B: Backend> PriceBatcher<B> {
    pub fn new(device: B::Device, input_size: usize) -> Self {
        Self { device, input_size }
    }
    
    pub fn batch(&self, samples: &[Sample]) -> PriceBatch<B> {
        let batch_size = samples.len();
        let mut features = Vec::with_capacity(batch_size * self.input_size);
        let mut targets = Vec::with_capacity(batch_size);
        
        for sample in samples {
            // Pad or truncate features to fixed input_size
            let mut fv = sample.features.clone();
            if fv.len() > self.input_size {
                // Truncate: take the most recent features (end of sequence)
                fv = fv[fv.len() - self.input_size..].to_vec();
            } else if fv.len() < self.input_size {
                // Pad: add zeros at the beginning (left-pad for time series)
                let padding = vec![0.0f32; self.input_size - fv.len()];
                fv = [padding, fv].concat();
            }
            features.extend(fv);
            targets.push(sample.target);
        }

        PriceBatch {
            features: Tensor::from_data(TensorData::new(features, [batch_size, 1, self.input_size]), &self.device),
            targets: Tensor::from_data(TensorData::new(targets, [batch_size]), &self.device),
        }
    }
}

/// Binary Cross-Entropy loss
pub fn binary_cross_entropy<B: Backend>(pred: Tensor<B, 3>, target: Tensor<B, 1>) -> Tensor<B, 1> {
    let batch = pred.dims()[0];
    let pred_flat: Tensor<B, 1> = pred.reshape([batch]);
    let eps = 1e-7;
    
    let safe = pred_flat.clone().clamp(eps, 1.0 - eps);
    let one_minus_target = target.clone().neg() + 1.0;
    let one_minus_pred = safe.clone().neg() + 1.0;
    
    let loss = target * safe.log() + one_minus_target * one_minus_pred.log();
    loss.neg().mean()
}

/// Format time range for display
fn format_time(timestamp: i64) -> String {
    Utc.timestamp_opt(timestamp, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| timestamp.to_string())
}

/// Main entry point
pub async fn run(config: TrainConfig) -> Result<()> {
    info!(
        web = true,
        db_path = %config.db_path,
        model_path = %config.model_path,
        epochs = config.epochs,
        batch_size = config.batch_size,
        learning_rate = config.learning_rate,
        hidden_size = config.hidden_size,
        validation_split = config.validation_split,
        "Starting model training on 5-minute time windows"
    );

    // Load labeled features from database
    let mut conn = db::init_db(&config.db_path).await?;
    let labeled_features = db::fetch_labeled_features(&mut conn)
        .await
        .context("Failed to fetch labeled features from database")?;

    if labeled_features.is_empty() {
        anyhow::bail!("No labeled features found. Run 'ololon collect' first to gather training data.");
    }

    // Convert to samples with metadata
    let samples: Vec<Sample> = labeled_features
        .iter()
        .map(|f| Sample {
            features: f.feature_vector.iter().map(|&x| x as f32).collect(),
            target: f.target.unwrap() as f32,
            time_range_start: f.time_range_start,
            num_points: f.num_points as usize,
        })
        .collect();

    // Statistics about the dataset
    let total_samples = samples.len();
    let up_count = samples.iter().filter(|s| s.target > 0.5).count();
    let down_count = total_samples - up_count;
    let avg_points = samples.iter().map(|s| s.num_points).sum::<usize>() as f64 / total_samples as f64;
    let feature_lengths: Vec<usize> = samples.iter().map(|s| s.features.len()).collect();
    let max_features = *feature_lengths.iter().max().unwrap_or(&config.hidden_size);
    let min_features = *feature_lengths.iter().min().unwrap_or(&config.hidden_size);

    info!(
        web = true,
        total_windows = total_samples,
        up_count,
        down_count,
        avg_price_points = format!("{:.0}", avg_points),
        feature_length_range = format!("{}-{}", min_features, max_features),
        first_window = format_time(samples.first().map(|s| s.time_range_start).unwrap_or(0)),
        last_window = format_time(samples.last().map(|s| s.time_range_start).unwrap_or(0)),
        "Dataset statistics"
    );

    // Determine input size: use max feature length or user-specified
    let input_size = max_features.max(config.hidden_size);
    info!(input_size, "Using input size for model (will pad shorter sequences)");

    // Split chronologically (first 80% train, last 20% validation)
    // This preserves time order - important for time series
    let split_idx = ((1.0 - config.validation_split) * samples.len() as f64) as usize;
    let (train_samples, val_samples) = samples.split_at(split_idx);

    let train_up = train_samples.iter().filter(|s| s.target > 0.5).count();
    let val_up = val_samples.iter().filter(|s| s.target > 0.5).count();

    info!(
        web = true,
        train_count = train_samples.len(),
        train_up,
        train_down = train_samples.len() - train_up,
        val_count = val_samples.len(),
        val_up,
        val_down = val_samples.len() - val_up,
        train_range = format!("{} to {}",
            format_time(train_samples.first().map(|s| s.time_range_start).unwrap_or(0)),
            format_time(train_samples.last().map(|s| s.time_range_start).unwrap_or(0))),
        val_range = format!("{} to {}",
            format_time(val_samples.first().map(|s| s.time_range_start).unwrap_or(0)),
            format_time(val_samples.last().map(|s| s.time_range_start).unwrap_or(0))),
        "Split dataset chronologically"
    );

    // Initialize device and model
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model_config = PricePredictorConfig::new(input_size, config.hidden_size);
    let model = model_config.init::<MyAutodiff>(&device);

    let train_batcher = PriceBatcher::<MyAutodiff>::new(device, input_size);
    let val_batcher = PriceBatcher::<MyBackend>::new(device, input_size);

    let mut optimizer = AdamConfig::new().init();
    let mut model = model;
    let mut final_train_loss = 0.0f32;
    let mut final_val_loss = 0.0f32;
    let mut best_accuracy = 0.0f32;

    info!("Starting training loop...");

    for epoch in 0..config.epochs {
        let mut train_loss = 0.0f32;
        let mut train_batches = 0;

        // Mini-batch training
        for chunk in train_samples.chunks(config.batch_size) {
            let batch = train_batcher.batch(chunk);
            let pred = model.forward(batch.features);
            let loss = binary_cross_entropy(pred, batch.targets);

            let loss_val = loss.to_data()
                .as_slice::<f32>()
                .map(|s| s.first().copied().unwrap_or(0.0))
                .unwrap_or(0.0);
            train_loss += loss_val;
            train_batches += 1;

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(config.learning_rate, model, grads_params);
        }

        let avg_train = train_loss / train_batches as f32;
        final_train_loss = avg_train;

        // Validation
        let model_val = model.valid();
        let mut val_loss = 0.0f32;
        let mut val_batches = 0;
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut up_correct = 0usize;
        let mut up_total = 0usize;
        let mut down_correct = 0usize;
        let mut down_total = 0usize;

        for chunk in val_samples.chunks(config.batch_size) {
            let batch = val_batcher.batch(chunk);
            let pred = model_val.forward(batch.features);
            let loss = binary_cross_entropy(pred.clone(), batch.targets.clone());
            
            let loss_val = loss.to_data()
                .as_slice::<f32>()
                .map(|s| s.first().copied().unwrap_or(0.0))
                .unwrap_or(0.0);
            val_loss += loss_val;
            val_batches += 1;

            let pred_data = pred.to_data();
            let target_data = batch.targets.to_data();
            let preds = pred_data.as_slice::<f32>().unwrap_or_default();
            let targets = target_data.as_slice::<f32>().unwrap_or_default();

            for (p, t) in preds.iter().zip(targets.iter()) {
                let is_up = *t > 0.5;
                let pred_up = *p > 0.5;
                let correct_pred = pred_up == is_up;
                
                if correct_pred { correct += 1; }
                total += 1;
                
                if is_up {
                    if correct_pred { up_correct += 1; }
                    up_total += 1;
                } else {
                    if correct_pred { down_correct += 1; }
                    down_total += 1;
                }
            }
        }

        let avg_val = val_loss / val_batches as f32;
        final_val_loss = avg_val;
        let accuracy = if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 };
        let up_acc = if up_total > 0 { up_correct as f32 / up_total as f32 * 100.0 } else { 0.0 };
        let down_acc = if down_total > 0 { down_correct as f32 / down_total as f32 * 100.0 } else { 0.0 };

        if accuracy > best_accuracy {
            best_accuracy = accuracy;
        }

        // Console: log every epoch
        // Web: log every 10 epochs or final epoch
        let is_web_visible = (epoch + 1) % 10 == 0 || epoch + 1 == config.epochs;
        
        info!(
            web = is_web_visible,
            epoch = epoch + 1,
            total_epochs = config.epochs,
            train_loss = format!("{:.4}", avg_train),
            val_loss = format!("{:.4}", avg_val),
            accuracy = format!("{:.2}%", accuracy),
            up_accuracy = format!("{:.2}%", up_acc),
            down_accuracy = format!("{:.2}%", down_acc),
            best = format!("{:.2}%", best_accuracy),
            "Training progress"
        );
    }

    // Save model
    info!(web = true, model_path = %config.model_path, "Saving model weights");
    let recorder = JsonGzFileRecorder::<FullPrecisionSettings>::new();
    model.save_file(&config.model_path, &recorder).context("Failed to save model")?;
    
    // Save model metadata to database
    db::save_model_metadata(
        &mut conn,
        &config.model_path,
        config.epochs,
        final_train_loss as f64,
        Some(final_val_loss as f64),
        config.hidden_size,
        config.num_layers,
        300, // 5-minute window duration
    ).await.ok();
    
    info!(
        web = true,
        model_path = %config.model_path,
        final_accuracy = format!("{:.2}%", best_accuracy),
        input_size,
        hidden_size = config.hidden_size,
        "Model training complete!"
    );
    
    Ok(())
}