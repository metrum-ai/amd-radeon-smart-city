#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=scripts/setup_runtime_defaults.sh
source "${REPO_ROOT}/scripts/setup_runtime_defaults.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_eq() {
    local expected="$1"
    local actual="$2"
    local label="$3"
    if [ "$actual" != "$expected" ]; then
        fail "${label}: expected '${expected}', got '${actual}'"
    fi
}

assert_eq "hardware" "$(select_decode_mode "6.17.0-29-generic" "")"      "kernel 6.17 enables hardware decode"
assert_eq "hardware" "$(select_decode_mode "6.18.2-arch1-1" "")"         "kernel 6.18 enables hardware decode"
assert_eq "hardware" "$(select_decode_mode "7.0.0-generic" "")"          "kernel 7.x enables hardware decode"
assert_eq "software" "$(select_decode_mode "6.16.9-generic" "")"         "kernel 6.16 falls back to software decode"
assert_eq "software" "$(select_decode_mode "6.8.0-50-generic" "")"       "kernel 6.8 falls back to software decode"
assert_eq "software" "$(select_decode_mode "5.15.0-generic" "")"         "kernel 5.x falls back to software decode"
assert_eq "software" "$(select_decode_mode "unknown" "")"                "unrecognised kernel string is treated as old"
assert_eq "hardware" "$(select_decode_mode "5.15.0-generic" "hardware")" "explicit override beats kernel autodetect"
assert_eq "software" "$(select_decode_mode "6.17.0-generic" "software")" "explicit software override beats kernel autodetect"
assert_eq "hardware" "$(select_decode_mode "6.17.0-generic" "auto")"     "'auto' treated as no override"

echo "select_decode_mode tests passed"
