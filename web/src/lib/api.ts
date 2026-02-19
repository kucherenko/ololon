import type { Feature, Trade, ModelMetadata, FeatureStats } from '$lib/types';
import { get } from 'svelte/store';
import { apiKey } from '$lib/stores/apiKey';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

class ApiError extends Error {
	status: number;
	
	constructor(message: string, status: number) {
		super(message);
		this.status = status;
	}
}

async function fetchApi<T>(path: string): Promise<T> {
	const token = get(apiKey);
	
	const headers: Record<string, string> = {
		'Content-Type': 'application/json'
	};
	
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
	}
	
	const response = await fetch(`${API_URL}${path}`, { headers });
	
	if (!response.ok) {
		const error = await response.json().catch(() => ({ error: 'Unknown error' }));
		
		if (response.status === 401) {
			throw new ApiError('Invalid API key', response.status);
		}
		
		throw new ApiError(error.error || 'Request failed', response.status);
	}
	
	return response.json();
}

export async function getHealth(): Promise<{ status: string }> {
	const response = await fetch(`${API_URL}/health`);
	return response.json();
}

export async function verifyApiKey(token: string): Promise<boolean> {
	const response = await fetch(`${API_URL}/api/features/stats?limit=1`, {
		headers: {
			'Authorization': `Bearer ${token}`,
			'Content-Type': 'application/json'
		}
	});
	return response.ok;
}

export async function getFeatures(params?: { limit?: number; offset?: number; labeled?: boolean }): Promise<Feature[]> {
	const searchParams = new URLSearchParams();
	if (params?.limit) searchParams.set('limit', params.limit.toString());
	if (params?.offset) searchParams.set('offset', params.offset.toString());
	if (params?.labeled !== undefined) searchParams.set('labeled', params.labeled.toString());
	
	const query = searchParams.toString();
	return fetchApi<Feature[]>(`/api/features${query ? `?${query}` : ''}`);
}

export async function getFeature(id: number): Promise<Feature> {
	return fetchApi<Feature>(`/api/features/${id}`);
}

export async function getFeatureStats(): Promise<FeatureStats> {
	return fetchApi<FeatureStats>('/api/features/stats');
}

export async function getTrades(params?: { limit?: number; offset?: number }): Promise<Trade[]> {
	const searchParams = new URLSearchParams();
	if (params?.limit) searchParams.set('limit', params.limit.toString());
	if (params?.offset) searchParams.set('offset', params.offset.toString());
	
	const query = searchParams.toString();
	return fetchApi<Trade[]>(`/api/trades${query ? `?${query}` : ''}`);
}

export async function getTrade(id: number): Promise<Trade> {
	return fetchApi<Trade>(`/api/trades/${id}`);
}

export async function getModelMetadata(): Promise<ModelMetadata> {
	return fetchApi<ModelMetadata>('/api/model');
}