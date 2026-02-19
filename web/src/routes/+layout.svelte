<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import { apiKey } from '$lib/stores/apiKey';
	import '../app.css';

	let { children } = $props();
	let darkMode = $state(false);

	// Check for saved dark mode preference
	$effect(() => {
		if (typeof window !== 'undefined') {
			const saved = localStorage.getItem('ololon_dark_mode');
			if (saved !== null) {
				darkMode = saved === 'true';
			} else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
				darkMode = true;
			}
		}
	});

	// Apply dark mode class
	$effect(() => {
		if (typeof document !== 'undefined') {
			document.documentElement.classList.toggle('dark', darkMode);
			localStorage.setItem('ololon_dark_mode', String(darkMode));
		}
	});

	// Redirect to login if no API key is set
	$effect(() => {
		const key = $apiKey;
		const currentPath = $page.url.pathname;
		
		if (!key && currentPath !== '/login') {
			goto('/login');
		}
	});

	function handleLogout() {
		apiKey.clear();
		goto('/login');
	}

	const navItems = [
		{ href: '/', label: 'Dashboard', icon: 'dashboard' },
		{ href: '/features', label: 'Features', icon: 'chart' },
		{ href: '/trades', label: 'Trades', icon: 'trade' }
	];
</script>

<svelte:head>
	<title>Ololon Dashboard</title>
	<link rel="preconnect" href="https://fonts.googleapis.com">
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous">
	<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</svelte:head>

{#if $page.url.pathname === '/login'}
	{@render children()}
{:else}
	<div class="flex min-h-screen bg-background">
		<!-- Sidebar -->
		<aside class="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col bg-sidebar">
			<!-- Logo -->
			<div class="flex h-16 items-center gap-3 border-b border-sidebar-border px-6">
				<div class="flex h-9 w-9 items-center justify-center rounded-lg bg-primary font-semibold text-primary-foreground">
					O
				</div>
				<div>
					<h1 class="text-base font-semibold">Ololon</h1>
					<p class="text-xs text-muted-foreground">Trading Bot</p>
				</div>
			</div>
			
			<!-- Navigation -->
			<nav class="flex-1 space-y-1 p-4">
				{#each navItems as item}
					<a 
						href={item.href}
						class="flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors
							{$page.url.pathname === item.href 
								? 'bg-sidebar-accent text-sidebar-accent-foreground' 
								: 'text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground'}"
					>
						{#if item.icon === 'dashboard'}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<rect width="7" height="9" x="3" y="3" rx="1"/>
								<rect width="7" height="5" x="14" y="3" rx="1"/>
								<rect width="7" height="9" x="14" y="12" rx="1"/>
								<rect width="7" height="5" x="3" y="16" rx="1"/>
							</svg>
						{:else if item.icon === 'chart'}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<path d="M3 3v18h18"/>
								<path d="m19 9-5 5-4-4-3 3"/>
							</svg>
						{:else if item.icon === 'trade'}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<circle cx="12" cy="12" r="10"/>
								<path d="M12 6v6l4 2"/>
							</svg>
						{/if}
						{item.label}
					</a>
				{/each}
			</nav>

			<!-- Bottom section -->
			<div class="border-t border-sidebar-border p-4 space-y-1">
				<!-- Dark mode toggle -->
				<button 
					onclick={() => darkMode = !darkMode}
					class="flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm font-medium text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground transition-colors"
				>
					<span class="flex items-center gap-3">
						{#if darkMode}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
							</svg>
							<span>Dark</span>
						{:else}
							<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<circle cx="12" cy="12" r="4"/>
								<path d="M12 2v2"/>
								<path d="M12 20v2"/>
								<path d="m4.93 4.93 1.41 1.41"/>
								<path d="m17.66 17.66 1.41 1.41"/>
								<path d="M2 12h2"/>
								<path d="M20 12h2"/>
								<path d="m6.34 17.66-1.41 1.41"/>
								<path d="m19.07 4.93-1.41 1.41"/>
							</svg>
							<span>Light</span>
						{/if}
					</span>
					<div class="h-5 w-9 rounded-full bg-sidebar-accent relative transition-colors">
						<div class="absolute top-0.5 h-4 w-4 rounded-full bg-background shadow-sm transition-all duration-200 {darkMode ? 'left-4' : 'left-0.5'}"></div>
					</div>
				</button>

				<!-- Logout -->
				<button 
					onclick={handleLogout}
					class="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium text-sidebar-foreground/70 hover:bg-destructive/10 hover:text-destructive transition-colors"
				>
					<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
						<polyline points="16 17 21 12 16 7"/>
						<line x1="21" x2="9" y1="12" y2="12"/>
					</svg>
					Logout
				</button>
			</div>
		</aside>

		<!-- Main content -->
		<main class="ml-64 flex-1 p-8">
			{@render children()}
		</main>
	</div>
{/if}