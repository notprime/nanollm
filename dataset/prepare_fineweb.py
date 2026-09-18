"""
Phase 0: Dataset preparation for small 
         transformer training (10-60M params).

Streams a subset of FineWeb-Edu (2B tokens), tokenizes with the GPT-2 BPE tokenizer
(via tiktoken), and writes train.bin / val.bin as uint16 memmap-compatible
arrays, nanoGPT-style.

Usage:

    python prepare_fineweb.py --target_tokens 2_000_000_000 --out_dir data/fineweb_edu

Hardware assumptions: 16GB RAM, no GPU needed for this step. Tokenization
is CPU-bound; the dataset is streamed (never fully downloaded/loaded) so
RAM usage stays low regardless of target_tokens.
"""

import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_tokens", type=int, default=2_000_000_000,
                    help="Total tokens to collect (train+val combined).")
    p.add_argument("--val_fraction", type=float, default=0.01,
                    help="Fraction of tokens reserved for validation.")
    p.add_argument("--out_dir", type=str, default="data/fineweb_edu")
    p.add_argument("--hf_subset", type=str, default="sample-10BT",
                    help="FineWeb-Edu HF config name to stream from.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--flush_every", type=int, default=1000,
                    help="Docs per progress update / flush.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token # 50256, used as document separator
    vocab_size = enc.n_vocab

    print(f"Tokenizer: gpt2 (vocab_size={vocab_size}, eot={eot})")
    print(f"Target tokens: {args.target_tokens:,}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.hf_subset,
        split="train",
        streaming=True
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    """
    First pass: stream, tokenize, accumulate into a Python list of uint16
    chunks. We do NOT know total size in advance (streaming), so we grow
    a list and concatenate once, rather than writing incrementally with
    unknown final size. Memory: 2B tokens * 2 bytes = 4GB, fine for 16GB RAM
    held transiently before we write to disk and can drop the Python object.
    """
    token_chunks = []
    total_tokens = 0
    n_docs = 0

    pbar = tqdm(total=args.target_tokens, unit="tok", unit_scale=True)
    for example in ds:
        text = example["text"]
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        token_chunks.append(arr)
        total_tokens += len(arr)
        n_docs += 1

        if n_docs % args.flush_every == 0:
            pbar.update(len(arr) * args.flush_every // args.flush_every)
            pbar.n = min(total_tokens, args.target_tokens)
            pbar.refresh()

        if total_tokens >= args.target_tokens:
            break
    pbar.close()

    print(f"Collected {total_tokens:,} tokens from {n_docs:,} documents")

    all_tokens = np.concatenate(token_chunks)
    del token_chunks

    # Deterministic split: last val_fraction of the stream is validation.
    # (Docs were already shuffled at the source, so this is not biased
    # toward any particular topic/order.)
    n_val = int(len(all_tokens) * args.val_fraction)
    train_ids = all_tokens[:-n_val]
    val_ids = all_tokens[-n_val:]

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"train.bin: {len(train_ids):,} tokens -> {os.path.getsize(train_path)/1e9:.2f} GB")
    print(f"val.bin:   {len(val_ids):,} tokens -> {os.path.getsize(val_path)/1e9:.2f} GB")

    # Sanity checks
    assert os.path.getsize(train_path) == len(train_ids) * 2, "train.bin size mismatch"
    assert os.path.getsize(val_path) == len(val_ids) * 2, "val.bin size mismatch"
    assert train_ids.max() < vocab_size and val_ids.max() < vocab_size, "token id out of vocab range"

    # Round-trip decode check on a small slice
    sample = train_ids[:200].tolist()
    decoded = enc.decode(sample)
    print("\n--- Decode sanity check (first 200 tokens of train.bin) ---")
    print(decoded)
    print("--- end sample ---\n")

    # memmap load test
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    print(f"memmap load test: shape = {train_data.shape}, first items = {train_data[:10]}")

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()