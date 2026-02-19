<script lang="ts">
	import { goto } from '$app/navigation';
	import { apiKey } from '$lib/stores/apiKey';
	import { verifyApiKey, getHealth } from '$lib/api';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';

	let inputKey = $state('');
	let loading = $state(false);
	let error = $state<string | null>(null);
	let serverOnline = $state<'checking' | 'online' | 'offline'>('checking');
	let showKey = $state(false);

	// Check server health on mount
	$effect(() => {
		getHealth()
			.then(() => serverOnline = 'online')
			.catch(() => serverOnline = 'offline');
	});

	async function handleSubmit(e: Event) {
		e.preventDefault();
		if (!inputKey.trim()) {
			error = 'Please enter an API key';
			return;
		}

		loading = true;
		error = null;

		try {
			const valid = await verifyApiKey(inputKey.trim());
			if (valid) {
				apiKey.set(inputKey.trim());
				await goto('/');
			} else {
				error = 'Invalid API key';
			}
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to verify API key';
		} finally {
			loading = false;
		}
	}
</script>

<svelte:head>
	<title>Login - Ololon</title>
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous">
	<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</svelte:head>

<div class="flex min-h-screen items-center justify-center bg-muted/40 p-4">
	<Card.Root class="w-full max-w-md">
		<Card.Header class="space-y-1 text-center pb-4">
			<div class="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary font-bold text-2xl text-primary-foreground">
				O
			</div>
			<Card.Title class="text-2xl">Welcome to Ololon</Card.Title>
			<Card.Description>Enter your API key to access the dashboard</Card.Description>
		</Card.Header>
		
		<Card.Content class="pb-6">
			<!-- Server status -->
			{#if serverOnline === 'offline'}
				<div class="mb-4 flex items-start gap-3 rounded-lg bg-destructive/10 p-4 text-sm">
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 shrink-0 text-destructive mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<circle cx="12" cy="12" r="10"/>
						<line x1="12" x2="12" y1="8" y2="12"/>
						<line x1="12" x2="12.01" y1="16" y2="16"/>
					</svg>
					<div>
						<p class="font-medium text-destructive">Server Offline</p>
						<p class="mt-1 text-destructive/80">
							Start the backend with:<br/>
							<code class="mt-1 inline-block rounded bg-destructive/10 px-1.5 py-0.5 font-mono text-xs">cargo run -- server --auth-token YOUR_TOKEN</code>
						</p>
					</div>
				</div>
			{:else if serverOnline === 'checking'}
				<div class="mb-4 flex items-center gap-3 rounded-lg bg-muted p-4 text-sm">
					<svg class="h-4 w-4 animate-spin text-muted-foreground" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
					</svg>
					<span class="text-muted-foreground">Checking server status...</span>
				</div>
			{/if}

			<form onsubmit={handleSubmit} class="space-y-4">
				<div class="space-y-2">
					<label for="api-key" class="text-sm font-medium">API Key</label>
					<div class="relative">
						<input
							id="api-key"
							type={showKey ? 'text' : 'password'}
							bind:value={inputKey}
							placeholder="Enter your API key"
							class="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 pr-10 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
							disabled={loading || serverOnline === 'offline'}
						/>
						<button
							type="button"
							onclick={() => showKey = !showKey}
							class="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
							tabindex="-1"
						>
							{#if showKey}
								<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
									<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/>
									<path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/>
									<path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/>
									<line x1="2" x2="22" y1="2" y2="22"/>
								</svg>
							{:else}
								<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
									<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
									<circle cx="12" cy="12" r="3"/>
								</svg>
							{/if}
						</button>
					</div>
				</div>

				{#if error}
					<div class="flex items-center gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<circle cx="12" cy="12" r="10"/>
							<line x1="12" x2="12" y1="8" y2="12"/>
							<line x1="12" x2="12.01" y1="16" y2="16"/>
						</svg>
						{error}
					</div>
				{/if}

				<Button 
					type="submit" 
					class="w-full" 
					disabled={loading || serverOnline === 'offline'}
				>
					{#if loading}
						<svg class="mr-2 h-4 w-4 animate-spin" viewBox="0 0 24 24">
							<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
							<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
						</svg>
						Verifying...
					{:else}
						Access Dashboard
					{/if}
				</Button>
			</form>
		</Card.Content>

		<div class="border-t px-6 py-4">
			<p class="text-center text-xs text-muted-foreground">
				BTC/USDT 5-minute prediction trading bot
			</p>
		</div>
	</Card.Root>
</div>