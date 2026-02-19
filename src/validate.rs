//! Model validation module - validates trained model on all labeled data
//!
//! Produces comprehensive statistics including:
//! - Accuracy, precision, recall, F1 score
//! - Confusion matrix
//! - Up/down prediction breakdown
//! - Time-based performance analysis

use anyhow::{Context, Result};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, JsonGzFileRecorder};
use burn::tensor::backend::Backend;
use chrono::{TimeZone, Timelike, Utc};
use std::collections::HashMap;

use crate::db;
use crate::train::{PriceBatcher, PricePredictor, PricePredictorConfig};

type MyBackend = NdArray<f32>;

/// Configuration for the validate subcommand
#[derive(Debug, Clone)]
pub struct ValidateConfig {
    pub db_path: String,
    pub model_path: String,
    pub hidden_size: usize,
    /// Only validate on new data not used for training
    pub new_only: bool,
}

/// Validation sample with metadata
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ValidationSample {
    features: Vec<f32>,
    target: f32,
    time_range_start: i64,
    time_range_end: i64,
    start_price: f64,
    end_price: f64,
}

/// Confusion matrix statistics
#[derive(Debug, Clone, Copy, Default)]
struct ConfusionMatrix {
    true_positives: usize,   // Predicted up, was up
    true_negatives: usize,   // Predicted down, was down
    false_positives: usize,  // Predicted up, was down
    false_negatives: usize,  // Predicted down, was up
}

impl ConfusionMatrix {
    fn total(&self) -> usize {
        self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    }

    fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 { return 0.0; }
        (self.true_positives + self.true_negatives) as f64 / total as f64
    }

    fn precision(&self) -> f64 {
        let predicted_positives = self.true_positives + self.false_positives;
        if predicted_positives == 0 { return 0.0; }
        self.true_positives as f64 / predicted_positives as f64
    }

    fn recall(&self) -> f64 {
        let actual_positives = self.true_positives + self.false_negatives;
        if actual_positives == 0 { return 0.0; }
        self.true_positives as f64 / actual_positives as f64
    }

    fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();
        if precision + recall == 0.0 { return 0.0; }
        2.0 * precision * recall / (precision + recall)
    }

    fn specificity(&self) -> f64 {
        let actual_negatives = self.true_negatives + self.false_positives;
        if actual_negatives == 0 { return 0.0; }
        self.true_negatives as f64 / actual_negatives as f64
    }
}

/// Hourly performance statistics
#[derive(Debug, Clone)]
struct HourlyStats {
    hour: u32,
    total: usize,
    correct: usize,
    up_predictions: usize,
    down_predictions: usize,
}

/// Daily performance statistics
#[derive(Debug, Clone)]
struct DailyStats {
    date: String,
    total: usize,
    correct: usize,
}

/// Format timestamp for display
fn format_timestamp(ts: i64) -> String {
    Utc.timestamp_opt(ts, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
        .unwrap_or_else(|| ts.to_string())
}

/// Format date only from timestamp
fn format_date(ts: i64) -> String {
    Utc.timestamp_opt(ts, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| ts.to_string())
}

/// Extract hour from timestamp
fn extract_hour(ts: i64) -> u32 {
    Utc.timestamp_opt(ts, 0)
        .single()
        .map(|dt| dt.hour())
        .unwrap_or(0)
}

/// Run predictions and compute statistics
fn run_inference<B: Backend>(
    model: &PricePredictor<B>,
    batcher: &PriceBatcher<B>,
    samples: &[ValidationSample],
) -> Result<Vec<(f32, f32)>> {
    // Use the batcher from train.rs but need to handle ValidationSample
    let train_samples: Vec<crate::train::Sample> = samples
        .iter()
        .map(|s| crate::train::Sample {
            features: s.features.clone(),
            target: s.target,
            time_range_start: s.time_range_start,
            num_points: s.features.len(),
            feature_id: 0, // Not needed for inference
        })
        .collect();

    let mut predictions = Vec::new();
    
    for chunk in train_samples.chunks(32) {
        let batch = batcher.batch(chunk);
        let pred = model.forward(batch.features);
        let pred_data = pred.to_data();
        let target_data = batch.targets.to_data();
        let preds = pred_data.as_slice::<f32>().unwrap_or_default();
        let targets = target_data.as_slice::<f32>().unwrap_or_default();

        for (p, t) in preds.iter().zip(targets.iter()) {
            predictions.push((*p, *t));
        }
    }

    Ok(predictions)
}

/// Main entry point
pub async fn run(config: ValidateConfig) -> Result<()> {
    println!("Starting model validation");
    println!("  Database: {}", config.db_path);
    println!("  Model: {}", config.model_path);
    println!("  Hidden size: {}", config.hidden_size);
    if config.new_only {
        println!("  Mode: NEW DATA ONLY (not used for training)");
    }
    println!();

    // Initialize database
    let mut conn = db::init_db(&config.db_path).await?;

    // Get model metadata to show creation date
    let model_metadata = db::get_full_model_metadata(&mut conn).await?;
    
    if let Some(ref metadata) = model_metadata {
        let trained_date = Utc.timestamp_opt(metadata.trained_at, 0)
            .single()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| metadata.trained_at.to_string());
        println!("MODEL INFO");
        println!("  Trained at: {}", trained_date);
        println!("  Epochs: {}", metadata.epochs);
        println!("  Hidden size: {}", metadata.hidden_size);
        println!("  Input size: {}", metadata.input_size);
        println!("  Final train loss: {:.6}", metadata.final_train_loss);
        if let Some(val_loss) = metadata.final_val_loss {
            println!("  Final val loss: {:.6}", val_loss);
        }
        println!();
    } else {
        println!("WARNING: No model metadata found in database. Model may not have been trained with this version.\n");
    }

    // Load labeled features based on mode
    let labeled_features = if config.new_only {
        db::fetch_untrained_labeled_features(&mut conn)
            .await
            .context("Failed to fetch untrained labeled features from database")?
    } else {
        db::fetch_labeled_features(&mut conn)
            .await
            .context("Failed to fetch labeled features from database")?
    };

    if labeled_features.is_empty() {
        if config.new_only {
            anyhow::bail!("No new labeled features found (all data has been used for training). Run 'ololon collect' to gather more data.");
        } else {
            anyhow::bail!("No labeled features found in database. Run 'ololon collect' first.");
        }
    }

    // Show data statistics
    let total_labeled = db::count_labeled_features(&mut conn).await?;
    let untrained_count = db::count_untrained_features(&mut conn).await?;
    
    println!("DATA STATISTICS");
    println!("  Total labeled features: {}", total_labeled);
    println!("  Untrained features: {}", untrained_count);
    println!("  Validating on: {} samples", labeled_features.len());
    println!();

    // Convert to validation samples
    let samples: Vec<ValidationSample> = labeled_features
        .iter()
        .map(|f| ValidationSample {
            features: f.feature_vector.iter().map(|&x| x as f32).collect(),
            target: f.target.unwrap() as f32,
            time_range_start: f.time_range_start,
            time_range_end: f.time_range_end,
            start_price: f.start_price,
            end_price: f.end_price.unwrap_or(f.start_price),
        })
        .collect();

    // Determine input size from model metadata (not from validation data)
    // This ensures the model architecture matches what was trained
    
    // For legacy models where input_size=0, compute from ALL labeled features (not just validation set)
    // because the model was trained with the full dataset's feature dimensions
    let all_labeled_features = db::fetch_labeled_features(&mut conn)
        .await
        .context("Failed to fetch all labeled features")?;
    let max_feature_len_all = all_labeled_features
        .iter()
        .map(|f| f.feature_vector.len())
        .max()
        .unwrap_or(config.hidden_size);
    let computed_input_size = max_feature_len_all.max(config.hidden_size);
    
    let input_size = model_metadata
        .as_ref()
        .and_then(|m| {
            let size = m.input_size as usize;
            // Fall back to computed size if metadata has invalid input_size (0 or less)
            if size > 0 { Some(size) } else { None }
        })
        .unwrap_or(computed_input_size);
    
    println!("Using input size: {}\n", input_size);

    // Load model
    println!("Loading trained model...");
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let model_config = PricePredictorConfig::new(input_size, config.hidden_size);
    let model = model_config.init::<MyBackend>(&device);

    let recorder = JsonGzFileRecorder::<FullPrecisionSettings>::new();
    let model = model.load_file(&config.model_path, &recorder, &device)
        .context("Failed to load model weights. Make sure to train the model first with 'ololon train'")?;

    println!("Model loaded successfully\n");

    // Run inference
    let batcher = PriceBatcher::<MyBackend>::new(device, input_size);
    let predictions = run_inference(&model, &batcher, &samples)?;

    // Compute statistics
    let mut confusion = ConfusionMatrix::default();
    let mut hourly_stats: HashMap<u32, HourlyStats> = HashMap::new();
    let mut daily_stats: HashMap<String, DailyStats> = HashMap::new();
    
    // Track prediction distribution
    let mut up_correct = 0usize;
    let mut up_total = 0usize;
    let mut down_correct = 0usize;
    let mut down_total = 0usize;
    
    // Track probability distribution
    let mut prob_bins = [0usize; 10]; // 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    let mut prob_correct_bins = [0usize; 10];
    
    // Price movement statistics
    let mut total_price_change_pct = 0.0f64;
    let mut correct_price_change_pct = 0.0f64;
    let mut incorrect_price_change_pct = 0.0f64;

    for (i, (pred_prob, target)) in predictions.iter().enumerate() {
        let sample = &samples[i];
        let is_up = *target > 0.5;
        let pred_up = *pred_prob > 0.5;
        let correct = pred_up == is_up;

        // Update confusion matrix
        if is_up && pred_up {
            confusion.true_positives += 1;
        } else if !is_up && !pred_up {
            confusion.true_negatives += 1;
        } else if !is_up && pred_up {
            confusion.false_positives += 1;
        } else {
            confusion.false_negatives += 1;
        }

        // Track up/down accuracy
        if is_up {
            up_total += 1;
            if correct { up_correct += 1; }
        } else {
            down_total += 1;
            if correct { down_correct += 1; }
        }

        // Update hourly stats
        let hour = extract_hour(sample.time_range_start);
        let hourly = hourly_stats.entry(hour).or_insert(HourlyStats {
            hour,
            total: 0,
            correct: 0,
            up_predictions: 0,
            down_predictions: 0,
        });
        hourly.total += 1;
        if correct { hourly.correct += 1; }
        if pred_up { hourly.up_predictions += 1; } else { hourly.down_predictions += 1; }

        // Update daily stats
        let date = format_date(sample.time_range_start);
        let daily = daily_stats.entry(date.clone()).or_insert(DailyStats {
            date,
            total: 0,
            correct: 0,
        });
        daily.total += 1;
        if correct { daily.correct += 1; }

        // Update probability bins
        let bin_idx = ((*pred_prob * 10.0) as usize).min(9);
        prob_bins[bin_idx] += 1;
        if correct { prob_correct_bins[bin_idx] += 1; }

        // Track price change statistics
        let price_change_pct = ((sample.end_price - sample.start_price) / sample.start_price * 100.0).abs();
        total_price_change_pct += price_change_pct;
        if correct {
            correct_price_change_pct += price_change_pct;
        } else {
            incorrect_price_change_pct += price_change_pct;
        }
    }

    let total_samples = predictions.len();

    // Print comprehensive statistics
    println!();
    println!("=== VALIDATION RESULTS ===");
    println!();
    
    // Overall metrics
    println!("OVERALL PERFORMANCE");
    println!("  Total samples: {}", total_samples);
    println!("  Accuracy:       {:8.2}%", confusion.accuracy() * 100.0);
    println!("  Precision:      {:8.4}", confusion.precision());
    println!("  Recall:         {:8.4}", confusion.recall());
    println!("  Specificity:    {:8.4}", confusion.specificity());
    println!("  F1 Score:       {:8.4}", confusion.f1_score());
    println!();

    // Confusion matrix
    println!("CONFUSION MATRIX");
    println!("                    Actual");
    println!("                UP ({:>5})    DOWN ({:>5})", up_total, down_total);
    println!(
        "  Predicted UP    {:>8}      {:>8}",
        confusion.true_positives,
        confusion.false_positives
    );
    println!(
        "  Predicted DOWN  {:>8}      {:>8}",
        confusion.false_negatives,
        confusion.true_negatives
    );
    println!();

    // Class breakdown
    println!("CLASS BREAKDOWN");
    let up_accuracy = if up_total > 0 { up_correct as f64 / up_total as f64 * 100.0 } else { 0.0 };
    let down_accuracy = if down_total > 0 { down_correct as f64 / down_total as f64 * 100.0 } else { 0.0 };
    println!("  UP predictions:    {:>5}/{:<5} correct ({:>6.2}%)", up_correct, up_total, up_accuracy);
    println!("  DOWN predictions:  {:>5}/{:<5} correct ({:>6.2}%)", down_correct, down_total, down_accuracy);
    println!();

    // Time range
    println!("TIME RANGE");
    if let Some(first) = samples.first() {
        println!("  First sample: {}", format_timestamp(first.time_range_start));
    }
    if let Some(last) = samples.last() {
        println!("  Last sample:  {}", format_timestamp(last.time_range_start));
    }
    let time_span_hours = if let (Some(first), Some(last)) = (samples.first(), samples.last()) {
        (last.time_range_start - first.time_range_start) as f64 / 3600.0
    } else {
        0.0
    };
    println!("  Time span:    {:.2} hours ({:.1} days)", time_span_hours, time_span_hours / 24.0);
    println!();

    // Probability calibration
    println!("PROBABILITY CALIBRATION");
    println!("  Range         Count    Correct   Accuracy");
    for (i, (count, correct)) in prob_bins.iter().zip(prob_correct_bins.iter()).enumerate() {
        let accuracy = if *count > 0 { *correct as f64 / *count as f64 * 100.0 } else { 0.0 };
        let lower = i as f64 / 10.0;
        let upper = (i + 1) as f64 / 10.0;
        println!("  [{:.1}-{:.1}]    {:>5}      {:>5}    {:>6.2}%", lower, upper, count, correct, accuracy);
    }
    println!();

    // Price movement analysis
    println!("PRICE MOVEMENT ANALYSIS");
    let avg_price_change = total_price_change_pct / total_samples as f64;
    let avg_correct_change = if confusion.true_positives + confusion.true_negatives > 0 {
        correct_price_change_pct / (confusion.true_positives + confusion.true_negatives) as f64
    } else {
        0.0
    };
    let avg_incorrect_change = if confusion.false_positives + confusion.false_negatives > 0 {
        incorrect_price_change_pct / (confusion.false_positives + confusion.false_negatives) as f64
    } else {
        0.0
    };
    println!("  Avg price change:        {:8.4}%", avg_price_change);
    println!("  Avg change (correct):    {:8.4}%", avg_correct_change);
    println!("  Avg change (incorrect):  {:8.4}%", avg_incorrect_change);
    println!();

    // Hourly performance (top 5 best and worst hours)
    println!("HOURLY PERFORMANCE (Top 5 Best Hours)");
    let mut hourly_vec: Vec<_> = hourly_stats.values().collect();
    hourly_vec.sort_by(|a, b| {
        let acc_a = if a.total > 0 { a.correct as f64 / a.total as f64 } else { 0.0 };
        let acc_b = if b.total > 0 { b.correct as f64 / b.total as f64 } else { 0.0 };
        acc_b.partial_cmp(&acc_a).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    for hourly in hourly_vec.iter().take(5) {
        let accuracy = if hourly.total > 0 {
            hourly.correct as f64 / hourly.total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  Hour {:02}:00 - {:>4} samples, {:>6.2}% accuracy, UP:{:>4} DOWN:{:>4}",
            hourly.hour, hourly.total, accuracy, hourly.up_predictions, hourly.down_predictions
        );
    }
    println!();

    // Worst hours
    println!("HOURLY PERFORMANCE (Top 5 Worst Hours)");
    for hourly in hourly_vec.iter().rev().take(5) {
        let accuracy = if hourly.total > 0 {
            hourly.correct as f64 / hourly.total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  Hour {:02}:00 - {:>4} samples, {:>6.2}% accuracy, UP:{:>4} DOWN:{:>4}",
            hourly.hour, hourly.total, accuracy, hourly.up_predictions, hourly.down_predictions
        );
    }
    println!();

    // Daily performance summary
    println!("DAILY PERFORMANCE SUMMARY");
    let mut daily_vec: Vec<_> = daily_stats.values().collect();
    daily_vec.sort_by(|a, b| a.date.cmp(&b.date));
    
    let mut best_day = None;
    let mut worst_day = None;
    let mut total_days = 0;
    let mut summed_daily_accuracy = 0.0;
    
    for daily in &daily_vec {
        let accuracy = if daily.total > 0 {
            daily.correct as f64 / daily.total as f64 * 100.0
        } else {
            0.0
        };
        total_days += 1;
        summed_daily_accuracy += accuracy;
        
        match &best_day {
            None => best_day = Some((daily, accuracy)),
            Some((_, best_acc)) if accuracy > *best_acc => {
                best_day = Some((daily, accuracy));
            }
            _ => {}
        }
        match &worst_day {
            None => worst_day = Some((daily, accuracy)),
            Some((_, worst_acc)) if accuracy < *worst_acc && daily.total >= 5 => {
                worst_day = Some((daily, accuracy));
            }
            _ => {}
        }
    }

    if let Some((day, acc)) = best_day {
        println!("  Best day:   {} - {:>4} samples, {:>6.2}% accuracy", day.date, day.total, acc);
    }
    if let Some((day, acc)) = worst_day {
        println!("  Worst day:  {} - {:>4} samples, {:>6.2}% accuracy", day.date, day.total, acc);
    }
    let avg_daily_accuracy = if total_days > 0 {
        summed_daily_accuracy / total_days as f64
    } else {
        0.0
    };
    println!("  Avg daily accuracy: {:>8.2}% over {:>3} days", avg_daily_accuracy, total_days);
    println!();

    // Display last 100 records with predictions
    let display_count = predictions.len().min(100);
    let start_idx = predictions.len().saturating_sub(100);
    
    println!("=== LAST {} RECORDS WITH PREDICTIONS ===", display_count);
    println!("  #   |       Time Range       |  Start Price  |   End Price   | Pred  | Actual | Result    | Prob");
    
    for idx in start_idx..predictions.len() {
        let (pred_prob, target) = &predictions[idx];
        let sample = &samples[idx];
        let is_up = *target > 0.5;
        let pred_up = *pred_prob > 0.5;
        let correct = pred_up == is_up;
        
        let time_range = format!(
            "{} - {}",
            format_timestamp(sample.time_range_start),
            format_timestamp(sample.time_range_end)
        );
        
        let pred_dir = if *pred_prob > 0.5 { "UP  " } else { "DOWN" };
        let actual_dir = if is_up { "UP  " } else { "DOWN" };
        let result = if correct { "CORRECT" } else { "WRONG" };
        
        println!(
            " {:>3}  | {:^22} | {:>13.2} | {:>13.2} | {:>4}  | {:>6} | {:>8} | {:>6.2}%",
            idx + 1,
            time_range,
            sample.start_price,
            sample.end_price,
            pred_dir,
            actual_dir,
            result,
            pred_prob * 100.0
        );
    }

    // Summary recommendation
    println!();
    println!("=== SUMMARY ===");
    println!("  Accuracy:  {:.2}%", confusion.accuracy() * 100.0);
    println!("  Precision: {:.4}", confusion.precision());
    println!("  Recall:    {:.4}", confusion.recall());
    println!("  F1 Score:  {:.4}", confusion.f1_score());
    println!("  Samples:   {}", total_samples);
    println!();

    if confusion.accuracy() > 0.55 {
        println!("Model shows predictive capability above random baseline (50%)");
    } else if confusion.accuracy() > 0.50 {
        println!("Model shows minimal predictive capability, consider more training data");
    } else {
        println!("Model below random baseline, investigate data quality or model architecture");
    }

    Ok(())
}