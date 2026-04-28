# Created by Metrum AI for AMD

from __future__ import annotations


def gpu_for_stream(stream_id: int, gpu_count: int) -> int:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    return stream_id % gpu_count
