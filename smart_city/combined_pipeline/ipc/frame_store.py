# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Frame store for the combined pipeline.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class FrameStore:
    """Fixed-layout shared memory pools for full-res RGB8 and NCHW infer tensors."""

    def __init__(
        self,
        *,
        num_slots: int,
        full_height: int,
        full_width: int,
        infer_height: int,
        infer_width: int,
        infer_dtype: np.dtype,
        full_shm: SharedMemory | None = None,
        infer_shm: SharedMemory | None = None,
    ) -> None:
        """Initialize the FrameStore."""
        self.num_slots = num_slots
        self.full_height = full_height
        self.full_width = full_width
        self.infer_height = infer_height
        self.infer_width = infer_width
        self.infer_dtype = infer_dtype

        full_nbytes = num_slots * full_height * full_width * 3
        infer_nbytes = num_slots * 3 * infer_height * infer_width * np.dtype(infer_dtype).itemsize

        if full_shm is None:
            self.full_shm = SharedMemory(create=True, size=full_nbytes)
            self._owns_full = True
        else:
            self.full_shm = full_shm
            self._owns_full = False

        if infer_shm is None:
            self.infer_shm = SharedMemory(create=True, size=infer_nbytes)
            self._owns_infer = True
        else:
            self.infer_shm = infer_shm
            self._owns_infer = False

        self._full_arr = np.ndarray(
            (num_slots, full_height, full_width, 3),
            dtype=np.uint8,
            buffer=self.full_shm.buf,
        )
        self._infer_arr = np.ndarray(
            (num_slots, 3, infer_height, infer_width),
            dtype=infer_dtype,
            buffer=self.infer_shm.buf,
        )

    @property
    def full_shm_name(self) -> str:
        """Get the full shared memory name."""
        return self.full_shm.name

    @property
    def infer_shm_name(self) -> str:
        """Get the infer shared memory name."""
        return self.infer_shm.name

    def full_view(self, slot: int) -> np.ndarray:
        """Get the full view of the frame store."""
        return self._full_arr[slot]

    def infer_view(self, slot: int) -> np.ndarray:
        """Get the infer view of the frame store."""
        return self._infer_arr[slot]

    def close(self) -> None:
        """Close the frame store."""
        self.full_shm.close()
        self.infer_shm.close()

    def unlink(self) -> None:
        """Unlink the frame store."""
        if self._owns_full:
            try:
                self.full_shm.unlink()
            except FileNotFoundError:
                pass
        if self._owns_infer:
            try:
                self.infer_shm.unlink()
            except FileNotFoundError:
                pass


def attach_frame_store(
    *,
    full_shm_name: str,
    infer_shm_name: str,
    num_slots: int,
    full_height: int,
    full_width: int,
    infer_height: int,
    infer_width: int,
    infer_dtype: np.dtype,
) -> FrameStore:
    """Attach the frame store."""
    full_shm = SharedMemory(name=full_shm_name)
    infer_shm = SharedMemory(name=infer_shm_name)
    return FrameStore(
        num_slots=num_slots,
        full_height=full_height,
        full_width=full_width,
        infer_height=infer_height,
        infer_width=infer_width,
        infer_dtype=infer_dtype,
        full_shm=full_shm,
        infer_shm=infer_shm,
    )


def make_slot_free_queues(stream_count: int, slots_per_stream: int) -> list[mp.Queue]:
    """One queue per stream pre-filled with local slot indices 0..slots_per_stream-1."""
    queues: list[mp.Queue] = []
    for _ in range(stream_count):
        queues.append(mp.Queue())
    for s in range(stream_count):
        for k in range(slots_per_stream):
            queues[s].put(k)
    return queues


def global_slot(stream_id: int, slots_per_stream: int, local_slot: int) -> int:
    """Get the global slot."""
    return stream_id * slots_per_stream + local_slot
