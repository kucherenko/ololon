import { writable } from 'svelte/store';
import { browser } from '$app/environment';

const STORAGE_KEY = 'ololon_api_key';

function createApiKeyStore() {
	const storedValue = browser ? localStorage.getItem(STORAGE_KEY) : null;
	const { subscribe, set, update } = writable<string | null>(storedValue);

	return {
		subscribe,
		set: (value: string | null) => {
			if (browser) {
				if (value) {
					localStorage.setItem(STORAGE_KEY, value);
				} else {
					localStorage.removeItem(STORAGE_KEY);
				}
			}
			set(value);
		},
		clear: () => {
			if (browser) {
				localStorage.removeItem(STORAGE_KEY);
			}
			set(null);
		}
	};
}

export const apiKey = createApiKeyStore();