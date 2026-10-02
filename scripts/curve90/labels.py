"""Disk-backed labels with at most one contract combination mapped at a time."""
from pathlib import Path
import numpy as np


class LabelStore:
    def __init__(self, path, shape):
        self.path = Path(path)
        self.shape = tuple(shape)
        self.dtype = np.dtype('float32')
        self.block_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        with self.path.open('wb') as f:
            f.truncate(self.shape[0] * self.block_bytes)

    def _block(self, index, mode):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.shape[0]:
            raise IndexError(index)
        return np.memmap(self.path, mode=mode, dtype=self.dtype,
                         offset=int(index) * self.block_bytes, shape=self.shape[1:])

    def __getitem__(self, index):
        block = self._block(index, 'r')
        try:
            # The returned array owns its memory and remains valid after unmap.
            return np.array(block, copy=True)
        finally:
            block._mmap.close()

    def __setitem__(self, index, values):
        block = self._block(index, 'r+')
        try:
            block[:] = values
            block.flush()
        finally:
            block._mmap.close()

    def flush(self):
        # Each block write is flushed and unmapped before returning.
        pass
