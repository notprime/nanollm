"""
[Project 1 - Phase 0]
-> Checks batcher invariants, decoded samples, document-length stats.

Usage:

    # batcher checks + decoded sample on either dataset
    python check_data.py --data_dir data/shakespeare --block_size 64
    python check_data.py --data_dir data/fineweb_edu --block_size 256

    # document-length distribution of the FineWeb training file (numpy only)
    python check_data.py --lengths_file data/fineweb_edu/train.bin --plot doc_lengths.png

Needs data.py (get_batch) in the same folder for the batcher checks.
"""

import argparse

import numpy as np

EOT = 50256 # GPT-2 end-of-text id, appended after every document
VOCAB_SIZE = 50257


def check_batcher(data_dir, block_size=64, batch_size=8):
    import torch
    import tiktoken
    from .data import get_batch

    # 1. Shapes and dtype
    g = torch.Generator().manual_seed(0)
    x, y = get_batch("train", data_dir, block_size, batch_size, generator=g)
    assert x.shape == y.shape == (batch_size, block_size), (x.shape, y.shape)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    print(f"Shapes are ok: x, y = {tuple(x.shape)}, dtype {x.dtype}")

    # 2. y is shifted left by one token
    assert torch.equal(x[:, 1:], y[:, :-1]), "Shift check failed"
    print("Shift is ok: [x:, 1:] == y[:, :-1]")

    # 3. Token ids stay inside the vocabulary
    assert 0 <= int(x.min()) and int(x.max()) < VOCAB_SIZE
    assert 0 <= int(y.min()) and int(y.max()) < VOCAB_SIZE
    print(f"Ids are ok: max id {int(max(x.max(), y.max()))} < {VOCAB_SIZE}")

    # 4. Same seed -> same batch / different seed -> different batch
    a = get_batch("train", data_dir, block_size, batch_size, generator=torch.Generator().manual_seed(1))
    b = get_batch("train", data_dir, block_size, batch_size, generator=torch.Generator().manual_seed(1))
    c = get_batch("train", data_dir, block_size, batch_size, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "Same seed gave different batches!"
    assert not torch.equal(a[0], c[0]), "Different seeds gave the same batch!"
    print("Reproducibility is ok: same seed -> same batch, different seed -> different batch")

    # 5. The val split is readable too
    xv, yv = get_batch("val", data_dir, block_size, batch_size, generator=g)
    assert torch.equal(xv[:, 1:], yv[:, :-1])
    print("Val split is ok")

    # 6. Eyeball it: decoded x and y should be the same text offset by one token
    enc = tiktoken.get_encoding("gpt2")
    print("\n--- decoded x[0] ---")
    print(enc.decode(x[0].tolist())[:400])
    print("--- decoded y[0] (same text, shifted one token) ---")
    print(enc.decode(y[0].tolist())[:400])
    print("--- end ---\n")


def doc_legths(path, chunk=100_000_000):
    """
    Tokens per document (end-of-text token included), from a flat uint16 file. 
    Scans in chunks so the boolean mask never exceeds 'chunk' bytes.
    """
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ends = [np.flatnonzero(data[s:s+chunk] == EOT) + s for s in range(0, len(data), chunk)]
    eot_pos = np.concatenate(ends)
    # First document starts at index 0, so prepend -1; a trailing partial
    # document (cut by the train/val split) has no EOT and is not counted
    return np.diff(np.concatenate(([-1], eot_pos)))


def report_lengths(path, plot_path=None, windows=(128, 256, 512, 1024)):
    lengths = doc_legths(path)
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99])
    print(f"Documents: {len(lengths):,}, total tokens counted: {int(lengths.sum()):,}")
    print(f"length (tokens): mean {lengths.mean():.0f}, median {p50:.0f}, p90 {p90:.0f}, p99 {p99:.0f}, max {lengths.max():,}")
    for w in windows:
        print(f"Documents shorter than {w:>4}: {(lengths < w).mean() * 100:5.1f}%")
    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(lengths[lengths <= p99], bins=100)
        plt.xlabel("document length (tokens, up to the 99th percentile)")
        plt.ylabel("documents")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        print(f"histogram saved to {plot_path}")


def main(argv=None):
    p = argparse.ArgumentParser(argv)
    p.add_argument("--data_dir", type=str, help="folder with train.bin / val.bin")
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lengths_file", type=str, help=".bin file for the document-length report")
    p.add_argument("--plot", type=str, default=None, help="PNG path for the length histogram")
    args = p.parse_args()

    if not args.data_dir and not args.lengths_file:
        p.error("pass --data_dir and/or --lengths_file")
    if args.data_dir:
        check_batcher(args.data_dir, args.block_size, args.batch_size)
    if args.lengths_file:
        report_lengths(args.lengths_file, args.plot)
    print("done.")


if __name__ == "__main__":
    main()




