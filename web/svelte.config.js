import adapter from '@sveltejs/adapter-node';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter(),
		alias: {
			$components: './src/lib/components',
			$utils: './src/lib/utils',
			$hooks: './src/lib/hooks'
		}
	}
};

export default config;
