import type { Handle } from '@sveltejs/kit';

const API_URL = process.env.VITE_API_URL || 'http://server:3000';

export const handle: Handle = async ({ event, resolve }) => {
	const { pathname } = event.url;

	// Proxy API and health requests to backend server
	if (pathname.startsWith('/api') || pathname === '/health') {
		const targetUrl = `${API_URL}${pathname}${event.url.search}`;

		const headers: Record<string, string> = {};
		event.request.headers.forEach((value, key) => {
			// Forward relevant headers
			if (key.toLowerCase() === 'authorization' || key.toLowerCase() === 'content-type') {
				headers[key] = value;
			}
		});

		const response = await fetch(targetUrl, {
			method: event.request.method,
			headers,
			body: event.request.method !== 'GET' && event.request.method !== 'HEAD'
				? await event.request.text()
				: undefined
		});

		return new Response(response.body, {
			status: response.status,
			headers: {
				'content-type': response.headers.get('content-type') || 'application/json'
			}
		});
	}

	return resolve(event);
};