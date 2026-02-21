//! Shared model types for LSTM-based price prediction
//!
//! This module contains all model-related types that are used across
//! training, inference, and validation modules to avoid code duplication.

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, LstmConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation::sigmoid, Tensor, TensorData};

/// Training sample representing a 5-minute time window
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f32>,
    pub target: f32,
    pub time_range_start: i64,
    pub num_points: usize,
    pub feature_id: i64,
}

/// Batched data for model training/inference
#[derive(Debug, Clone)]
pub struct PriceBatch<B: Backend> {
    pub features: Tensor<B, 3>,
    pub targets: Tensor<B, 1>,
}

/// LSTM-based binary classification model for price prediction
#[derive(Module, Debug)]
pub struct PricePredictor<B: Backend> {
    lstm: burn::nn::Lstm<B>,
    fc: Linear<B>,
    hidden_size: usize,
}

impl<B: Backend> PricePredictor<B> {
    /// Forward pass through the model
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let (hidden_state, _) = self.lstm.forward(input, None);
        let dims = hidden_state.dims();
        let batch = dims[0];
        let seq = dims[1];
        let hidden = dims[2];

        // Get last timestep: [batch, seq, hidden] -> [batch, 1, 1]
        let last: Tensor<B, 2> = hidden_state
            .slice([0..batch, seq - 1..seq, 0..hidden])
            .reshape([batch, hidden]);

        let output = self.fc.forward(last);
        sigmoid(output).reshape([batch, 1, 1])
    }
}

/// Configuration for PricePredictor
#[derive(Config, Debug)]
pub struct PricePredictorConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl PricePredictorConfig {
    /// Initialize the model with this config
    pub fn init<B: Backend>(&self, device: &B::Device) -> PricePredictor<B> {
        let lstm = LstmConfig::new(self.input_size, self.hidden_size, false).init(device);
        let fc = LinearConfig::new(self.hidden_size, 1).init(device);
        PricePredictor {
            lstm,
            fc,
            hidden_size: self.hidden_size,
        }
    }
}

/// Batcher for converting samples to tensors
pub struct PriceBatcher<B: Backend> {
    device: B::Device,
    input_size: usize,
}

impl<B: Backend> PriceBatcher<B> {
    pub fn new(device: B::Device, input_size: usize) -> Self {
        Self { device, input_size }
    }

    /// Batch multiple samples into a single tensor
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
            features: Tensor::from_data(
                TensorData::new(features, [batch_size, 1, self.input_size]),
                &self.device,
            ),
            targets: Tensor::from_data(TensorData::new(targets, [batch_size]), &self.device),
        }
    }

    /// Batch a single feature vector for inference
    pub fn batch_single(&self, features: Vec<f32>) -> PriceBatch<B> {
        PriceBatch {
            features: Tensor::from_data(
                TensorData::new(features, [1, 1, self.input_size]),
                &self.device,
            ),
            targets: Tensor::from_data(TensorData::new(vec![0.0f32], [1]), &self.device),
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
