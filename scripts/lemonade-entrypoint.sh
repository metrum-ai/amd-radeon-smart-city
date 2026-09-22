#!/bin/sh
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT


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
pull_model() {
    curl -sf -X POST "http://localhost:${PORT}/api/v1/pull" \
        -H "Content-Type: application/json" \
        -d "{\"model_name\": \"${MODEL}\", \"checkpoint\": \"${CHECKPOINT}\", \"recipe\": \"llamacpp\", \"reasoning\": true}" \
        > /tmp/pull_out.json 2>&1 || true
}

echo "[entrypoint] Pulling/registering model '${MODEL}' (checkpoint=${CHECKPOINT})..."
pull_model

# Verify the GGUF sha256 against HF's x-linked-etag: lemonade only checks the
# file EXISTS, so a right-length bad transfer passes. Engineering by Metrum AI.
verify_gguf_checksum() {
    repo="${CHECKPOINT%%:*}"
    variant="${CHECKPOINT#*:}"
    gguf=$(find /root/.cache/huggingface -name "*${variant}*.gguf" 2>/dev/null | head -1)
    GGUF_PATH="$gguf"
    [ -n "$gguf" ] || { echo "[entrypoint] checksum: no GGUF found, skipping."; return 0; }

    expected=$(curl -sI --max-time 30 \
        "https://huggingface.co/${repo}/resolve/main/$(basename "$gguf")" 2>/dev/null \
        | tr -d '\r' | awk -F'"' '/^x-linked-etag:/ {print $2}')
    if [ -z "$expected" ]; then
        echo "[entrypoint] checksum: could not reach HuggingFace, skipping verification." >&2
        return 0
    fi

    echo "[entrypoint] checksum: verifying $(basename "$gguf") (this reads the whole file)..."
    actual=$(sha256sum "$gguf" | cut -d' ' -f1)
    if [ "$actual" = "$expected" ]; then
        echo "[entrypoint] checksum: OK ($actual)"
        return 0
    fi

    echo "[entrypoint] ============================================================" >&2
    echo "[entrypoint] ERROR: GGUF CHECKSUM MISMATCH -- the model file is corrupt." >&2
    echo "[entrypoint]   file:     $gguf" >&2
    echo "[entrypoint]   expected: $expected" >&2
    echo "[entrypoint]   actual:   $actual" >&2
    echo "[entrypoint] A corrupt model still loads and generates, but produces" >&2
    echo "[entrypoint] garbage ('?') output. Delete the file and restart to re-pull." >&2
    echo "[entrypoint] Set LEMONADE_SKIP_CHECKSUM=1 to bypass this check." >&2
    echo "[entrypoint] ============================================================" >&2
    return 1
}

# Detect-only is useless: a bad file still loads and serves garbage. Delete,
# re-pull once, re-verify, then fail hard. Engineering by Metrum AI.
if [ "${LEMONADE_SKIP_CHECKSUM:-0}" = "1" ]; then
    echo "[entrypoint] checksum: skipped (LEMONADE_SKIP_CHECKSUM=1)."
elif verify_gguf_checksum; then
    :
else
    echo "[entrypoint] checksum: deleting corrupt file and re-pulling once..." >&2
    [ -n "$GGUF_PATH" ] && rm -f "$GGUF_PATH"
    pull_model
    if verify_gguf_checksum; then
        echo "[entrypoint] checksum: re-pull succeeded, model verified."
    else
        echo "[entrypoint] FATAL: model still corrupt after re-pull. Refusing to serve" >&2
        echo "[entrypoint] a model that would produce garbage output. Check network/disk," >&2
        echo "[entrypoint] or set LEMONADE_SKIP_CHECKSUM=1 to start anyway." >&2
        exit 1
    fi
fi

# Load the model with the specified backend
echo "[entrypoint] Loading model '${MODEL}' with backend '${BACKEND}'..."
curl -sf -X POST "http://localhost:${PORT}/api/v1/load" \
    -H "Content-Type: application/json" \
    -d "{\"model_name\": \"${MODEL}\", \"llamacpp_backend\": \"${BACKEND}\"}" \
    > /dev/null &

echo "[entrypoint] Model load initiated in background. Server is live."

# Hand off to lemond (foreground)
wait $LEMOND_PID
