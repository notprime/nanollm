"""
[Project 1 - Phase 0]
-> Dataset preparation for small transformer training (10-60M params).

Streams a subset of FineWeb-Edu (2B tokens), tokenizes with the GPT-2 BPE tokenizer
(via tiktoken), and writes train.bin / val.bin as uint16 memmap-compatible
arrays, nanoGPT-style.
Streams a subset of FineWeb-Edu (2B tokens), tokenizes with the GPT-2 BPE tokenizer
(via tiktoken) in a pool of worker processes, and writes train.bin / val.bin
as uint16 memmap-compatible arrays, nanoGPT-style.

Usage:

    python prepare_fineweb.py --target_tokens 2_000_000_000 --out_dir data/fineweb_edu

Hardware assumptions: 16GB RAM, no GPU needed for this step. Tokenization
is CPU-bound; the dataset is streamed (never fully downloaded/loaded) so
RAM usage stays low regardless of target_tokens.
"""


import argparse
import os
import multiprocess as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# Worker-side globals
# Each worker process gets its own tiktoken Encoding, created once at
# process startup (via Pool(initializer=...)) rather than re-created per
# document, and rather than pickled/sent from the main process each call.
_enc = None
_eot = None


def _init_worker():
    global _enc, _eot
    _enc = tiktoken.get_encoding("gpt2")
    _eot = _enc.eot_token


def _tokenize(text):
    """
    Runs in a worker process. 
    Pure function of 'text' -> uint16 array.
    """
    ids = _enc.encode_ordinary(text)
    ids.append(_eot)
    return np.array(ids, dtype=np.uint16)


def parse_args(argv=None):
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
    p.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="Worker processes for tokenization. The network fetch "
                         "itself stays single-threaded in the main process.")
    p.add_argument("--chunksize", type=int, default=64,
                    help="Documents per unit of work handed to a worker at once.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Main-process tokenizer instance, used only for the vocab_size/eot
    # constants and the final decode sanity check — NOT used for the bulk
    # tokenization work, which happens in worker processes (see _tokenize).
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    print(f"Tokenizer: gpt2 (vocab_size={vocab_size}, eot={enc.eot_token})")
    print(f"Target tokens: {args.target_tokens:,}")
    print(f"Tokenizing with {args.num_proc} worker process(es)")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.hf_subset,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=10_000)

    def text_stream():
        """
        Generator: pulls raw text from the network stream. This runs in
        the main process and is I/O-bound — multiprocessing does NOT speed
        this part up, which is why we only parallelize the tokenize step
        below, not the fetching.
        """
        for example in ds:
            yield example["text"]

    # Producer/consumer pattern: the main process still does the network
    # fetching (via text_stream(), single-threaded, I/O-bound), but each
    # document's tokenization (CPU-bound) is dispatched to the worker pool.
    # pool.imap (not imap_unordered) preserves input order, keeping our
    # accumulated token stream in the same document order as the shuffled
    # source — imap_unordered would be faster but is not worth losing
    # reproducibility for here.
    #
    # We do NOT know total size in advance (streaming), so we grow a list
    # and concatenate once, rather than writing incrementally with unknown
    # final size. Memory: 2B tokens * 2 bytes = 4GB, fine for 16GB RAM held
    # transiently before we write to disk and can drop the Python object.
    token_chunks = []
    total_tokens = 0
    n_docs = 0

    pbar = tqdm(total=args.target_tokens, unit="tok", unit_scale=True)
    with mp.Pool(args.num_proc, initializer=_init_worker) as pool:
        for arr in pool.imap(_tokenize, text_stream(), chunksize=args.chunksize):
            token_chunks.append(arr)
            total_tokens += len(arr)
            n_docs += 1

            if n_docs % args.flush_every == 0:
                pbar.n = min(total_tokens, args.tartget_tokens)
                pbar.refresh()

            if total_tokens >= args.target_tokens:
                break

        # Exiting the `with` block terminates any workers still processing
        # in-flight chunksize batches beyond our early break — expected and
        # harmless, since we only wanted the tokens already collected.

    pbar.close()

    print(f"Collected {total_tokens:,} tokens from {n_docs:,} documents")

    all_tokens = np.concatenate(token_chunks)
    del token_chunks  # free the list before writing

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
    print("\n--- Decode sanity check (first ~200 tokens of train.bin) ---")
    print(decoded[:500])
    print("--- end sample ---\n")

    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
