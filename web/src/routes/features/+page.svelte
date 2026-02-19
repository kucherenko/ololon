<script lang="ts">
	import { onMount } from 'svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';
	import { getFeatures, getFeatureStats } from '$lib/api';
	import type { Feature, FeatureStats } from '$lib/types';

	let features = $state<Feature[]>([]);
	let stats = $state<FeatureStats | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	
	let labeledFilter = $state<'all' | 'labeled' | 'unlabeled'>('all');
	let page_num = $state(0);
	let limit = 50;

	onMount(async () => {
		await loadData();
	});

	async function loadData() {
		loading = true;
		error = null;
		try {
			const [featuresData, statsData] = await Promise.all([
				getFeatures({
					limit,
					offset: page_num * limit,
					labeled: labeledFilter === 'all' ? undefined : labeledFilter === 'labeled'
				}),
				getFeatureStats()
			]);
			features = featuresData;
			stats = statsData;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load features';
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

	function formatVector(vec: number[]): string {
		return `[${vec.slice(0, 3).map(v => v.toFixed(4)).join(', ')}${vec.length > 3 ? ', ...' : ''}]`;
	}

	const labelRate = $derived(stats ? ((stats.labeled_count / stats.total_count) * 100).toFixed(1) : '0');
</script>

<svelte:head>
	<title>Features - Ololon</title>
</svelte:head>

<div class="space-y-6">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h1 class="text-2xl font-semibold tracking-tight">Features</h1>
			<p class="text-muted-foreground">Collected feature vectors for model training</p>
		</div>
		{#if stats}
			<div class="flex gap-2">
				<Badge variant="secondary">Total: {stats.total_count}</Badge>
				<Badge>Labeled: {stats.labeled_count}</Badge>
				<Badge variant="outline">Unlabeled: {stats.unlabeled_count}</Badge>
			</div>
		{/if}
	</div>

	<!-- Filters -->
	<div class="flex items-center gap-4">
		<div class="flex rounded-lg border p-1">
			<button 
				class="rounded-md px-3 py-1 text-sm font-medium transition-colors
					{labeledFilter === 'all' 
						? 'bg-primary text-primary-foreground' 
						: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => { labeledFilter = 'all'; page_num = 0; loadData(); }}
			>
				All
			</button>
			<button 
				class="rounded-md px-3 py-1 text-sm font-medium transition-colors
					{labeledFilter === 'labeled' 
						? 'bg-primary text-primary-foreground' 
						: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => { labeledFilter = 'labeled'; page_num = 0; loadData(); }}
			>
				Labeled
			</button>
			<button 
				class="rounded-md px-3 py-1 text-sm font-medium transition-colors
					{labeledFilter === 'unlabeled' 
						? 'bg-primary text-primary-foreground' 
						: 'text-muted-foreground hover:text-foreground'}"
				onclick={() => { labeledFilter = 'unlabeled'; page_num = 0; loadData(); }}
			>
				Unlabeled
			</button>
		</div>

		{#if stats && stats.total_count > 0}
			<span class="text-sm text-muted-foreground">
				<span class="font-medium text-foreground">{labelRate}%</span> labeled
			</span>
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

	<!-- Features Table -->
	<Card.Root>
		<Card.Content class="pt-6">
			{#if loading}
				<div class="flex items-center justify-center py-12">
					<svg class="h-6 w-6 animate-spin text-muted-foreground" viewBox="0 0 24 24">
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
						<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
					</svg>
				</div>
			{:else if features.length > 0}
				<div class="rounded-md border">
					<div class="overflow-x-auto">
						<Table.Root>
							<Table.Header class="bg-muted/50">
								<Table.Row>
									<Table.Head>ID</Table.Head>
									<Table.Head>Time Range</Table.Head>
									<Table.Head class="text-right">Start Price</Table.Head>
									<Table.Head class="text-right">End Price</Table.Head>
									<Table.Head>Target</Table.Head>
									<Table.Head class="text-right">Points</Table.Head>
									<Table.Head>Features</Table.Head>
									<Table.Head>Created</Table.Head>
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each features as feature}
									<Table.Row class="hover:bg-muted/50 transition-colors">
										<Table.Cell class="font-mono text-xs text-muted-foreground">#{feature.id}</Table.Cell>
										<Table.Cell class="text-xs">
											<div>{formatTimestamp(feature.time_range_start)}</div>
											<div class="text-muted-foreground">to {formatTimestamp(feature.time_range_end)}</div>
										</Table.Cell>
										<Table.Cell class="text-right font-mono tabular-nums">${formatPrice(feature.start_price)}</Table.Cell>
										<Table.Cell class="text-right">
											{#if feature.end_price !== null}
												<span class="font-mono tabular-nums {feature.end_price > feature.start_price ? 'text-green-600 dark:text-green-500' : feature.end_price < feature.start_price ? 'text-red-600 dark:text-red-500' : ''}">
													${formatPrice(feature.end_price)}
												</span>
											{:else}
												<span class="text-muted-foreground">—</span>
											{/if}
										</Table.Cell>
										<Table.Cell>
											{#if feature.target !== null}
												<Badge variant={feature.target === 1 ? 'default' : 'outline'}>
													{feature.target === 1 ? 'UP' : 'DOWN'}
												</Badge>
											{:else}
												<span class="text-xs text-muted-foreground">Pending</span>
											{/if}
										</Table.Cell>
										<Table.Cell class="text-right tabular-nums">{feature.num_points}</Table.Cell>
										<Table.Cell class="font-mono text-xs text-muted-foreground max-w-32 truncate" title={JSON.stringify(feature.feature_vector)}>
											{formatVector(feature.feature_vector)}
										</Table.Cell>
										<Table.Cell class="text-xs text-muted-foreground">
											{formatTimestamp(feature.created_at)}
										</Table.Cell>
									</Table.Row>
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
							<path d="M3 3v18h18"/>
							<path d="m19 9-5 5-4-4-3 3"/>
						</svg>
					</div>
					<p class="mt-3 text-sm text-muted-foreground">No features found.</p>
					<p class="mt-1 text-xs text-muted-foreground">
						Run <code class="rounded bg-muted px-1.5 py-0.5 font-mono">cargo run -- collect</code> to start collecting data.
					</p>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>
</div>