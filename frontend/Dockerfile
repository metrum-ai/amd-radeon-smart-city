# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Stage 1: Build frontend with Node 22 Alpine (smaller base)
FROM node:22-alpine AS build

WORKDIR /app

ENV CI=true

# Install pnpm globally — version pinned to match pnpm-lock.yaml format
RUN npm install -g pnpm@10.30.2

# Copy package files first for better layer caching
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml* .npmrc* ./

# Install dependencies (frozen lockfile for reproducibility)
RUN pnpm install --frozen-lockfile

# Copy source and build
COPY . ./
RUN pnpm run build

# Stage 2: Serve with patched nginx stable (gzip, caching headers)
FROM nginx:1.30.1-alpine AS runtime

LABEL description="Smart City Analytics Platform - Gateway"
LABEL version="1.0.0"

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config — full main config with http + stream blocks
COPY nginx.conf /etc/nginx/nginx.conf

# Security: run as non-root user
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

EXPOSE 80
# WebRTC ICE TCP (stream-proxied to mediamtx:8189 internally)
EXPOSE 8189

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/health || exit 1

USER nginx

CMD ["nginx", "-g", "daemon off;"]
