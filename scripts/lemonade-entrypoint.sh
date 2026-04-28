#!/bin/sh
# Created by Metrum AI for AMD

# Lemonade server entrypoint: starts lemond and auto-loads the configured model.
# The model is pulled on first run and served from the persistent lemonade-cache
# volume on subsequent restarts — no re-download occurs if already present.
#
# Custom (non-catalog) models use the "user." namespace convention and must be
# registered with their HuggingFace checkpoint on first pull.

set -e

MODEL="${LEMONADE_MODEL:-user.Qwen3-30B-A3B-UD-Q4_K_XL}"
CHECKPOINT="${LEMONADE_CHECKPOINT:-unsloth/Qwen3-30B-A3B-GGUF:UD-Q4_K_XL}"
PORT=13305
BACKEND="${LEMONADE_LLAMACPP_BACKEND:-rocm}"

echo "[entrypoint] Starting lemonade server (model=${MODEL}, backend=${BACKEND})..."

# Start lemond in the background
/opt/lemonade/lemond --host 0.0.0.0 --port ${PORT} &
LEMOND_PID=$!

# Wait for the server to be healthy before loading the model
echo "[entrypoint] Waiting for lemonade server to be ready..."
RETRIES=60
while [ $RETRIES -gt 0 ]; do
    if curl -sf "http://localhost:${PORT}/live" > /dev/null 2>&1; then
        echo "[entrypoint] Server ready."
        break
    fi
    sleep 2
    RETRIES=$((RETRIES - 1))
done

if [ $RETRIES -eq 0 ]; then
    echo "[entrypoint] ERROR: lemonade server did not start in time." >&2
    exit 1
fi

# Register the model if not already in catalog (idempotent — pull handles both
# first-time download and resuming from an already-downloaded cache).
echo "[entrypoint] Pulling/registering model '${MODEL}' (checkpoint=${CHECKPOINT})..."
curl -sf -X POST "http://localhost:${PORT}/api/v1/pull" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\": \"${MODEL}\", \"checkpoint\": \"${CHECKPOINT}\", \"recipe\": \"llamacpp\", \"reasoning\": true}" \
    > /tmp/pull_out.json 2>&1 || true

# Load the model with the specified backend
echo "[entrypoint] Loading model '${MODEL}' with backend '${BACKEND}'..."
curl -sf -X POST "http://localhost:${PORT}/api/v1/load" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\": \"${MODEL}\", \"llamacpp_backend\": \"${BACKEND}\"}" \
    > /dev/null &

echo "[entrypoint] Model load initiated in background. Server is live."

# Hand off to lemond (foreground)
wait $LEMOND_PID
