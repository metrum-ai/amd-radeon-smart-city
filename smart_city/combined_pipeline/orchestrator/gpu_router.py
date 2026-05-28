# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
GPU router for the combined pipeline.
"""

from __future__ import annotations


def gpu_for_stream(stream_id: int, gpu_count: int) -> int:
    """Get the GPU for a stream."""
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return stream_id % gpu_count
