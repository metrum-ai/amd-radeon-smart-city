# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Readiness for the combined pipeline.
"""

from __future__ import annotations

import socket
import time


def wait_tcp(host: str, port: int, timeout_s: float = 90.0) -> bool:
    """Block until host:port accepts TCP connections or timeout_s elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False
