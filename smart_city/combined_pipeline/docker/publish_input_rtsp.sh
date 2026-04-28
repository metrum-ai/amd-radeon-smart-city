#!/usr/bin/env bash
# Created by Metrum AI for AMD

set -euo pipefail

: "${STREAM_COUNT:?STREAM_COUNT is required}"
VIDEO_DIR="${VIDEO_DIR:-/videos}"
INPUT_BASE_RTSP="${INPUT_BASE_RTSP:-rtsp://mediamtx:8554/cam}"
REALTIME_INPUT="${REALTIME_INPUT:-1}"

mapfile -t VIDEO_FILES < <(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | sort)
if [ "${#VIDEO_FILES[@]}" -eq 0 ]; then
  echo "[publish_input_rtsp] no mp4 files found in $VIDEO_DIR" >&2
  exit 1
fi

echo "[publish_input_rtsp] publishing ${STREAM_COUNT} streams from ${#VIDEO_FILES[@]} video file(s)"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

run_publisher() {
  local file="$1"
  local target="$2"
  local name="$3"

  while true; do
    input_args=()
    if [ "$REALTIME_INPUT" = "1" ]; then
      input_args+=("-re")
    fi

    ffmpeg -nostdin -loglevel warning \
      "${input_args[@]}" -stream_loop -1 -i "$file" \
      -an -c copy -f rtsp -rtsp_transport tcp "$target"

    echo "[publish_input_rtsp] $name disconnected; retrying in 2s..." >&2
    sleep 2
  done
}

sleep 2

for ((i=1; i<=STREAM_COUNT; i++)); do
  file="${VIDEO_FILES[$(((i - 1) % ${#VIDEO_FILES[@]}))]}"
  run_publisher "$file" "${INPUT_BASE_RTSP}${i}" "cam${i}" &
  PIDS+=("$!")
  echo "[publish_input_rtsp] cam${i} <- $(basename "$file")"
done

wait
