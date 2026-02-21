import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit(), tailwindcss()],
	server: {
		proxy: {
			'/api': {
				target: process.env.VITE_API_URL || 'http://localhost:3000',
				changeOrigin: true
			},
			'/health': {
				target: process.env.VITE_API_URL || 'http://localhost:3000',
				changeOrigin: true
			}
		}
	}
});
