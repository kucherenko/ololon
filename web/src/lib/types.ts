// Types matching the Rust backend

export interface Feature {
	id: number;
	time_range_start: number;
	time_range_end: number;
	start_price: number;
	end_price: number | null;
	feature_vector: number[];
	target: number | null;
	num_points: number;
	created_at: number;
	labeled_at: number | null;
}

export interface Trade {
	id: number;
	timestamp: number;
	market_id: string;
	outcome: string;
	predicted_prob: number;
	market_prob: number;
	edge: number;
	trade_size: number;
	avg_price: number;
	status: string;
	order_id: string | null;
	tx_hash: string | null;
	error_message: string | null;
	created_at: number;
}

export interface ModelMetadata {
	id: number;
	model_path: string;
	trained_at: number;
	epochs: number;
	final_train_loss: number;
	final_val_loss: number | null;
	hidden_size: number;
	num_layers: number;
	window_duration_secs: number;
}

export interface BotStatus {
	name: string;
	status: 'running' | 'stopped' | 'error';
	last_activity: number | null;
	message: string | null;
}

export interface DashboardStats {
	total_features: number;
	labeled_features: number;
	unlabeled_features: number;
	total_trades: number;
	successful_trades: number;
	model_trained_at: number | null;
}

export interface FeatureStats {
	total_count: number;
	labeled_count: number;
	unlabeled_count: number;
}

export interface LogEntry {
	raw: string;
	command: string | null;
}

// Parsed log entry from JSON
export interface ParsedLog {
	timestamp: string;
	level: string;
	target: string;
	message: string;
	command?: string;
	spans?: Record<string, unknown>;
}