# Ololon Web Dashboard

A SvelteKit web application for monitoring the Ololon trading bot.

## Prerequisites

- [Bun](https://bun.sh/) runtime
- Running Ololon backend server (Rust)

## Setup

1. Start the Rust backend server with an auth token:
   ```bash
   cargo run -- server --auth-token your-secret-token
   ```

2. Start the web development server:
   ```bash
   cd web
   bun run dev
   ```

3. Open http://localhost:5173 and enter your auth token in the login form.

The API key is stored in the browser's localStorage after successful authentication.

## Development

Start the development server:

```bash
bun run dev
```

Open http://localhost:5173 in your browser.

## Build for Production

```bash
bun run build
```

The built files will be in `.svelte-kit/output/`.

## Pages

- **Dashboard** (`/`) - Overview of bot status, feature statistics, and recent trades
- **Features** (`/features`) - List and filter collected feature vectors for training
- **Trades** (`/trades`) - Trade history from Polymarket execution

## API Integration

The web app connects to the Rust backend's REST API:

- `GET /health` - Server health check (no auth required)
- `GET /api/features` - List features (with pagination and filtering)
- `GET /api/features/stats` - Feature statistics
- `GET /api/trades` - List trades (with pagination)
- `GET /api/model` - Model training metadata

All `/api/*` endpoints require Bearer token authentication.

## Tech Stack

- [SvelteKit](https://kit.svelte.dev/) - Full-stack framework
- [shadcn-svelte](https://www.shadcn-svelte.com/) - UI components
- [Tailwind CSS v4](https://tailwindcss.com/) - Styling
- [Bun](https://bun.sh/) - Runtime and package manager