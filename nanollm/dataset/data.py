"""
[Project 1 - Phase 0]
-> Batch sampling from the prepared token files.

get_batch() reads a flat, headerless uint16 file (train.bin / val.bin) through
np.memmap and returns (x, y) batches of token IDs. It does not care which
dataset produced the file: Tiny Shakespeare and FineWeb-Edu share the format.

The datasets themselves are built by prepare_tinyshakespeare.py and
prepare_fineweb.py; this module only loads.
"""

import os

import numpy as np
import torch


def get_batch(split, data_dir, block_size, batch_size, device="cpu", generator=None):
    """
    Return (x, y), both int64 tensors of shape (batch_size, block_size).
    y is shifted left by one token: y[b, t] is the target for x[b, t].
    """
    # re-create the memmap on every call: avoids a memory leak that happens
    # when a single long-lived np.memmap is kept open (nanoGPT does the same)
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    # valid starts: i in [0, len(data) - block_size - 1], so that data[i+1 : i+1+block_size] exists
    # torch.randint's upper bound is exclusive -> len(data) - block_size
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)

    # uint16 -> int64: PyTorch embeddings/cross-entropy need int64 indices
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if "cuda" in str(device):
        # pinned memory + non_blocking lets the host->GPU copy overlap with compute
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y