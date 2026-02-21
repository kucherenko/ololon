# Multi-stage build for SvelteKit application using Bun

# Stage 1: Build
FROM oven/bun:1-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json bun.lock ./

# Install dependencies
RUN bun install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN bun run build

# Stage 2: Production
FROM oven/bun:1-alpine AS production

WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 bunjs && \
    adduser --system --uid 1001 sveltekit -G bunjs

# Copy built application
COPY --from=builder --chown=sveltekit:bunjs /app/build ./build
COPY --from=builder --chown=sveltekit:bunjs /app/package.json /app/bun.lock ./

# Install production dependencies only
RUN bun install --frozen-lockfile --production

USER sveltekit

# Expose port
EXPOSE 3000

# Set environment variables
ENV NODE_ENV=production \
    PORT=3000 \
    HOST=0.0.0.0

# Start the server
CMD ["bun", "run", "build/index.js"]