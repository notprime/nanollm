"""
[Project 1 - Phase 0]
-> Tiny Shakespeare preparation (debug dataset)

Downloads the text once, tokenizes it with the GPT-2 BPE tokenizer (tiktoken),
splits it into train/val and writes train.bin / val.bin as flat, headerless
uint16 arrays, the same on-disk format as the FineWeb-Edu shards, so the
same get_batch() in data.py reads both.

Usage:

    python prepare_tinyshakespeare.py --out_dir data/shakespeare
"""


import argparse
import os
import urllib.request

import numpy as np
import tiktoken


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def prepare_shakespeare(out_dir="data/shakespeare", val_fraction=0.1):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Download the raw text once, reuse the local copy afterwards
    txt_path = os.path.join(out_dir, "input.txt")
    if not os.path.exists(txt_path):
        urllib.request.urlretrieve(SHAKESPEARE_URL, txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Tokenize the whole file as one document, same tokenizer as FineWeb prep
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    arr = np.array(ids, dtype=np.uint16)

    # 3. Split: last val_fraction is validation. 
    #    We use 10% (instead of 1% as in FineWeb) because the
    #    corpus is tiny (~300k tokens): 1% would leave ~3K val tokens
    n_val = int(len(arr) * val_fraction)
    train_ids, val_ids = arr[:-n_val], arr[-n_val:]

    # 4. Write raw uint16 bytes, no header (memmap-friendly)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"chars: {len(text):,} | tokens: {len(arr):,}")
    print(f"train: {len(train_ids):,} tokens | val: {len(val_ids):,} tokens")

    # SANITY CHECKS
    assert os.path.getsize(train_path) == len(train_ids) * 2, "train.bin size mismatch"
    assert os.path.getsize(val_path) == len(val_ids) * 2, "val.bin size mismatch"
    assert arr.max() < enc.n_vocab, "token id out of vocab range"
    # BPE is lossless: decoding all tokens (minus the final EOT) 
    # must give back the original text exactly
    assert enc.decode(arr[:-1].tolist()) == text, "decode round-trip doesn not match the source text"
    print("All sanity checks passed.")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="data/shakespeare")
    p.add_argument("--val_fraction", type=float, default=0.1)
    args = p.parse_args(argv)
    prepare_shakespeare(args.out_dir, args.val_fraction)


if __name__ == "__main__":
    main()