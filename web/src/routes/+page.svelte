<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { getFeatureStats, getTrades, getModelMetadata, getHealth } from '$lib/api';
	import type { FeatureStats, Trade, ModelMetadata } from '$lib/types';

	let stats = $state<FeatureStats | null>(null);
	let trades = $state<Trade[]>([]);
	let model = $state<ModelMetadata | null>(null);
	let serverStatus = $state<'online' | 'offline' | 'checking'>('checking');
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			await getHealth();
			serverStatus = 'online';
			
			const [statsData, tradesData, modelData] = await Promise.all([
				getFeatureStats(),
				getTrades({ limit: 10 }),
				getModelMetadata().catch(() => null)
			]);
			
			stats = statsData;
			trades = tradesData;
			model = modelData;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load data';
			serverStatus = 'offline';
		}
	});

	function formatTimestamp(ts: number): string {
		return new Date(ts * 1000).toLocaleString();
	}

	function formatPrice(price: number): string {
		return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	}

	const successfulTrades = $derived(trades.filter(t => t.status === 'filled').length);
	const labelRate = $derived(stats ? ((stats.labeled_count / stats.total_count) * 100).toFixed(1) : '0');
</script>

<svelte:head>
	<title>Dashboard - Ololon</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Dashboard</h1>
			<p class="text-muted-foreground">Overview of your trading bot activity</p>
		</div>
		<div class="flex items-center gap-2">
			{#if serverStatus === 'online'}
				<div class="flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary">
					<span class="relative flex h-2 w-2">
						<span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
						<span class="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
					</span>
					Online
				</div>
			{:else if serverStatus === 'offline'}
				<div class="flex items-center gap-1.5 rounded-full bg-destructive/10 px-3 py-1 text-sm font-medium text-destructive">
					<span class="h-2 w-2 rounded-full bg-destructive"></span>
					Offline
				</div>
			{:else}
				<div class="flex items-center gap-1.5 rounded-full bg-muted px-3 py-1 text-sm font-medium text-muted-foreground">
					<svg class="h-3 w-3 animate-spin" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
					</svg>
					Connecting
				</div>
			{/if}
		</div>
	</div>

	{#if error}
		<div class="flex items-start gap-3 rounded-lg bg-destructive/10 p-4 text-sm">
			<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 shrink-0 text-destructive mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<circle cx="12" cy="12" r="10"/>
				<line x1="12" x2="12" y1="8" y2="12"/>
				<line x1="12" x2="12.01" y1="16" y2="16"/>
			</svg>
			<div>
				<p class="font-medium text-destructive">Error loading data</p>
				<p class="mt-1 text-destructive/80">{error}</p>
			</div>
		</div>
	{/if}

	<!-- Bot Status Cards -->
	<div class="grid gap-4 md:grid-cols-3">
		<Card.Root>
			<Card.Header class="pb-2">
				<div class="flex items-center justify-between">
					<Card.Title class="text-sm font-medium">Collector Bot</Card.Title>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
						<path d="M3 3v5h5"/>
						<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
						<path d="M16 16h5v5"/>
					</svg>
				</div>
			</Card.Header>
			<Card.Content>
				<div class="flex items-center gap-2 text-sm">
					<span class="h-2 w-2 rounded-full bg-yellow-500"></span>
					<span class="text-muted-foreground">Status: Unknown</span>
				</div>
				<p class="mt-4 text-xs text-muted-foreground">
					Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- collect</code> to start
				</p>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="pb-2">
				<div class="flex items-center justify-between">
					<Card.Title class="text-sm font-medium">Training Bot</Card.Title>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d="M12 20h9"/>
						<path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
					</svg>
				</div>
			</Card.Header>
			<Card.Content>
				{#if model}
					<div class="flex items-center gap-2 text-sm">
						<span class="relative flex h-2 w-2">
							<span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-500 opacity-75"></span>
							<span class="relative inline-flex h-2 w-2 rounded-full bg-green-500"></span>
						</span>
						<span class="text-green-600 dark:text-green-500">Model trained</span>
					</div>
					<p class="mt-1 text-xs text-muted-foreground">
						{formatTimestamp(model.trained_at)}
					</p>
					<div class="mt-3 flex items-center gap-4 text-sm">
						<span class="text-muted-foreground">Epochs: <span class="font-medium text-foreground tabular-nums">{model.epochs}</span></span>
						<span class="text-muted-foreground">Loss: <span class="font-medium text-foreground tabular-nums">{model.final_train_loss.toFixed(4)}</span></span>
					</div>
				{:else}
					<div class="flex items-center gap-2 text-sm">
						<span class="h-2 w-2 rounded-full bg-muted-foreground/50"></span>
						<span class="text-muted-foreground">No model trained</span>
					</div>
					<p class="mt-4 text-xs text-muted-foreground">
						Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- train</code>
					</p>
				{/if}
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="pb-2">
				<div class="flex items-center justify-between">
					<Card.Title class="text-sm font-medium">Trading Bot</Card.Title>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<circle cx="12" cy="12" r="10"/>
						<path d="M12 6v6l4 2"/>
					</svg>
				</div>
			</Card.Header>
			<Card.Content>
				<div class="flex items-center gap-2 text-sm">
					<span class="h-2 w-2 rounded-full bg-yellow-500"></span>
					<span class="text-muted-foreground">Status: Unknown</span>
				</div>
				<p class="mt-4 text-xs text-muted-foreground">
					Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- trade</code> to start
				</p>
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Stats Grid -->
	<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
		<Card.Root>
			<Card.Header class="pb-2">
				<Card.Title class="text-sm font-medium text-muted-foreground">Total Features</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="text-2xl font-bold tabular-nums">{stats?.total_count ?? '—'}</div>
				<p class="text-xs text-muted-foreground mt-1">Data points collected</p>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="pb-2">
				<Card.Title class="text-sm font-medium text-muted-foreground">Labeled Features</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="text-2xl font-bold tabular-nums">{stats?.labeled_count ?? '—'}</div>
				<p class="text-xs text-muted-foreground mt-1">{labelRate}% of total</p>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="pb-2">
				<Card.Title class="text-sm font-medium text-muted-foreground">Unlabeled Features</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="text-2xl font-bold tabular-nums">{stats?.unlabeled_count ?? '—'}</div>
				<p class="text-xs text-muted-foreground mt-1">Pending labeling</p>
			</Card.Content>
		</Card.Root>

		<Card.Root>
			<Card.Header class="pb-2">
				<Card.Title class="text-sm font-medium text-muted-foreground">Total Trades</Card.Title>
			</Card.Header>
			<Card.Content>
				<div class="text-2xl font-bold tabular-nums">{trades.length > 0 ? trades.length : '—'}</div>
				<p class="text-xs text-muted-foreground mt-1">{successfulTrades} filled</p>
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Recent Trades -->
	<Card.Root>
		<Card.Header>
			<div class="flex items-center justify-between">
				<div>
					<Card.Title>Recent Trades</Card.Title>
					<Card.Description>Last 10 trades executed on Polymarket</Card.Description>
				</div>
				<a href="/trades" class="text-sm font-medium text-primary hover:underline">
					View all
				</a>
			</div>
		</Card.Header>
		<Card.Content>
			{#if trades.length > 0}
				<div class="rounded-md border">
					<div class="overflow-x-auto">
						<table class="w-full text-sm">
							<thead class="bg-muted/50">
								<tr class="border-b">
									<th class="h-10 px-4 text-left font-medium text-muted-foreground">Time</th>
									<th class="h-10 px-4 text-left font-medium text-muted-foreground">Outcome</th>
									<th class="h-10 px-4 text-right font-medium text-muted-foreground">Predicted</th>
									<th class="h-10 px-4 text-right font-medium text-muted-foreground">Market</th>
									<th class="h-10 px-4 text-right font-medium text-muted-foreground">Edge</th>
									<th class="h-10 px-4 text-right font-medium text-muted-foreground">Size</th>
									<th class="h-10 px-4 text-left font-medium text-muted-foreground">Status</th>
								</tr>
							</thead>
							<tbody class="divide-y">
								{#each trades as trade}
									<tr class="hover:bg-muted/50 transition-colors">
										<td class="px-4 py-3 text-muted-foreground">{formatTimestamp(trade.timestamp)}</td>
										<td class="px-4 py-3">
											<Badge variant={trade.outcome === 'YES' ? 'default' : 'secondary'}>
												{trade.outcome}
											</Badge>
										</td>
										<td class="px-4 py-3 text-right font-mono tabular-nums">{(trade.predicted_prob * 100).toFixed(1)}%</td>
										<td class="px-4 py-3 text-right font-mono tabular-nums text-muted-foreground">{(trade.market_prob * 100).toFixed(1)}%</td>
										<td class="px-4 py-3 text-right font-mono tabular-nums text-green-600 dark:text-green-500 font-medium">+{(trade.edge * 100).toFixed(1)}%</td>
										<td class="px-4 py-3 text-right font-mono tabular-nums">${formatPrice(trade.trade_size)}</td>
										<td class="px-4 py-3">
											<Badge variant={trade.status === 'filled' ? 'default' : 'destructive'}>
												{trade.status}
											</Badge>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			{:else}
				<div class="flex flex-col items-center justify-center py-12 text-center">
					<div class="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<circle cx="12" cy="12" r="10"/>
							<path d="M12 6v6l4 2"/>
						</svg>
					</div>
					<p class="mt-3 text-sm text-muted-foreground">No trades recorded yet.</p>
					<p class="mt-1 text-xs text-muted-foreground">
						Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- trade</code> to start trading.
					</p>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>
</div>