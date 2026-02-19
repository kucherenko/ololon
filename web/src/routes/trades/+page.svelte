<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';
	import { getTrades } from '$lib/api';
	import type { Trade } from '$lib/types';

	let trades = $state<Trade[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	
	let page_num = $state(0);
	let limit = 50;

	onMount(async () => {
		await loadData();
	});

	async function loadData() {
		loading = true;
		error = null;
		try {
			trades = await getTrades({
				limit,
				offset: page_num * limit
			});
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load trades';
		} finally {
			loading = false;
		}
	}

	function formatTimestamp(ts: number): string {
		return new Date(ts * 1000).toLocaleString();
	}

	function formatPrice(price: number): string {
		return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	}

	function formatProb(prob: number): string {
		return (prob * 100).toFixed(1) + '%';
	}

	// Calculate summary stats
	let totalTrades = $derived(trades.length);
	let successfulTrades = $derived(trades.filter(t => t.status === 'filled').length);
	let totalEdge = $derived(trades.reduce((sum, t) => sum + t.edge, 0) / Math.max(trades.length, 1));
	let fillRate = $derived(totalTrades > 0 ? ((successfulTrades / totalTrades) * 100).toFixed(1) : '0');
</script>

<svelte:head>
	<title>Trades - Ololon</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Trades</h1>
			<p class="text-muted-foreground">Trade history from Polymarket execution</p>
		</div>
		{#if trades.length > 0}
			<div class="flex gap-2">
				<Badge variant="secondary">Total: {totalTrades}</Badge>
				<Badge>Filled: {successfulTrades}</Badge>
				<Badge variant="outline">Avg Edge: {(totalEdge * 100).toFixed(1)}%</Badge>
			</div>
		{/if}
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

	<!-- Trades Table -->
	<Card.Root>
		<Card.Content class="pt-6">
			{#if loading}
				<div class="flex items-center justify-center py-12">
					<svg class="h-6 w-6 animate-spin text-muted-foreground" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
					</svg>
				</div>
			{:else if trades.length > 0}
				<div class="rounded-md border">
					<div class="overflow-x-auto">
						<Table.Root>
							<Table.Header class="bg-muted/50">
								<Table.Row>
									<Table.Head>ID</Table.Head>
									<Table.Head>Timestamp</Table.Head>
									<Table.Head>Market</Table.Head>
									<Table.Head>Outcome</Table.Head>
									<Table.Head class="text-right">Predicted</Table.Head>
									<Table.Head class="text-right">Market</Table.Head>
									<Table.Head class="text-right">Edge</Table.Head>
									<Table.Head class="text-right">Size</Table.Head>
									<Table.Head class="text-right">Avg Price</Table.Head>
									<Table.Head>Status</Table.Head>
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each trades as trade}
									<Table.Row class="hover:bg-muted/50 transition-colors">
										<Table.Cell class="font-mono text-xs text-muted-foreground">#{trade.id}</Table.Cell>
										<Table.Cell class="text-xs">{formatTimestamp(trade.timestamp)}</Table.Cell>
										<Table.Cell>
											<code class="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">{trade.market_id.slice(0, 8)}...</code>
										</Table.Cell>
										<Table.Cell>
											<Badge variant={trade.outcome === 'YES' ? 'default' : 'outline'}>
												{trade.outcome}
											</Badge>
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums font-medium">
											{formatProb(trade.predicted_prob)}
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums text-muted-foreground">
											{formatProb(trade.market_prob)}
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums text-green-600 dark:text-green-500 font-medium">
											+{formatProb(trade.edge)}
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums">
											${formatPrice(trade.trade_size)}
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums text-muted-foreground">
											{formatProb(trade.avg_price)}
										</Table.Cell>
										<Table.Cell>
											<Badge 
												variant={trade.status === 'filled' ? 'default' : trade.status === 'pending' ? 'secondary' : 'destructive'}
											>
												{trade.status}
											</Badge>
										</Table.Cell>
									</Table.Row>
									{#if trade.error_message}
										<Table.Row class="bg-destructive/5">
											<Table.Cell colspan="10" class="text-xs text-destructive">
												<div class="flex items-start gap-2">
													<svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
														<circle cx="12" cy="12" r="10"/>
														<line x1="12" x2="12" y1="8" y2="12"/>
														<line x1="12" x2="12.01" y1="16" y2="16"/>
													</svg>
													<span class="font-medium">Error:</span> {trade.error_message}
												</div>
											</Table.Cell>
										</Table.Row>
									{/if}
								{/each}
							</Table.Body>
						</Table.Root>
					</div>
				</div>

				<!-- Pagination -->
				<div class="flex items-center justify-between mt-4">
					<button 
						class="flex items-center gap-1 rounded-md border px-3 py-1.5 text-sm transition-colors
							{page_num === 0 
								? 'cursor-not-allowed opacity-50' 
								: 'hover:bg-muted'}"
						disabled={page_num === 0}
						onclick={() => { page_num--; loadData(); }}
					>
						<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<polyline points="15 18 9 12 15 6"/>
						</svg>
						Previous
					</button>
					<span class="text-sm text-muted-foreground">Page {page_num + 1}</span>
					<button 
						class="flex items-center gap-1 rounded-md border px-3 py-1.5 text-sm transition-colors hover:bg-muted"
						onclick={() => { page_num++; loadData(); }}
					>
						Next
						<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<polyline points="9 18 15 12 9 6"/>
						</svg>
					</button>
				</div>
			{:else}
				<div class="flex flex-col items-center justify-center py-12 text-center">
					<div class="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
						<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
							<circle cx="12" cy="12" r="10"/>
							<path d="M12 6v6l4 2"/>
						</svg>
					</div>
					<p class="mt-3 text-sm text-muted-foreground">No trades recorded.</p>
					<p class="mt-1 text-xs text-muted-foreground">
						Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- trade</code> to start trading.
					</p>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>

	<!-- Trade Statistics -->
	{#if trades.length > 0}
		<div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Total Trades</Card.Title>
				</Card.Header>
				<Card.Content>
					<div class="text-2xl font-bold tabular-nums">{totalTrades}</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Filled</Card.Title>
				</Card.Header>
				<Card.Content>
					<div class="text-2xl font-bold tabular-nums text-green-600 dark:text-green-500">{successfulTrades}</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Fill Rate</Card.Title>
				</Card.Header>
				<Card.Content>
					<div class="text-2xl font-bold tabular-nums">{fillRate}%</div>
				</Card.Content>
			</Card.Root>

			<Card.Root>
				<Card.Header class="pb-2">
					<Card.Title class="text-sm font-medium text-muted-foreground">Avg Edge</Card.Title>
				</Card.Header>
				<Card.Content>
					<div class="text-2xl font-bold tabular-nums text-green-600 dark:text-green-500">+{(totalEdge * 100).toFixed(2)}%</div>
				</Card.Content>
			</Card.Root>
		</div>
	{/if}
</div>