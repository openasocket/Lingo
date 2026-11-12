#!/usr/bin/env python3
"""
Convert the SONAR text encoder checkpoint to safetensors for candle (Rust).

Downloads directly from Meta's CDN — no sonar-space or fairseq2 required.
Only needs: torch, safetensors

Every download is verified against a pinned SHA-256 before use, the
checkpoint is loaded with ``torch.load(weights_only=True)`` (no arbitrary
unpickling), and outputs are written to a temporary name and renamed into
place so an interrupted run never leaves a partial file behind.

SONAR shares NLLB-200's SentencePiece vocabulary, so the tokenizer is the
HuggingFace NLLB ``tokenizer.json`` (pinned revision); lingo remaps its four
special-token IDs to fairseq2's layout at runtime.

Usage:
    python3 scripts/convert_sonar_safetensors.py [--output-dir ~/.cache/lingo/sonar]
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/SONAR/sonar_text_encoder.pt"
CHECKPOINT_SHA256 = "b46ffc66abd0519ce6693797eaf8b15da9011159eae18d99900c273a54208bf7"

TOKENIZER_URL = (
    "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/"
    "f8d333a098d19b4fd9a8b18f94170487ad3f821d/tokenizer.json"
)
TOKENIZER_SHA256 = "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_verified(url, dest, expected_sha256):
    """Download `url` to `dest` via a `.part` file, verifying SHA-256.

    An existing `dest` is re-verified and re-downloaded if it does not match.
    """
    if dest.exists():
        if sha256_file(dest) == expected_sha256:
            print(f"  {dest.name} already downloaded and verified")
            return
        print(f"  {dest.name} exists but fails verification; re-downloading")
        dest.unlink()

    part = dest.with_name(dest.name + ".part")
    print(f"  Downloading {dest.name} from {url}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"\r  {downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB ({pct:.0f}%)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, str(part), reporthook=reporthook)
        print()
        actual = sha256_file(part)
        if actual != expected_sha256:
            raise SystemExit(
                f"SHA-256 mismatch for {url}:\n  expected {expected_sha256}\n  got      {actual}\n"
                "Refusing to use this file."
            )
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()


def main():
    parser = argparse.ArgumentParser(description="Convert SONAR text encoder to safetensors")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / ".cache/lingo/sonar"),
        help="Output directory for safetensors model",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Do not fetch tokenizer.json (lingo's Rust downloader fetches it itself)",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the ~3 GB PyTorch checkpoint after conversion",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"

    if not args.skip_tokenizer:
        print("Fetching tokenizer.json (NLLB-200 vocabulary)...")
        download_verified(TOKENIZER_URL, output_dir / "tokenizer.json", TOKENIZER_SHA256)

    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print(f"model.safetensors already exists ({size_mb:.0f} MB)")
        print(f"Delete {model_path} to reconvert.")
        return

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Install them first, e.g.:  python3 -m pip install torch safetensors"
        )

    print("Converting SONAR text encoder to safetensors...")
    print(f"Output: {output_dir}\n")

    checkpoint_path = output_dir / "sonar_text_encoder.pt"
    download_verified(CHECKPOINT_URL, checkpoint_path, CHECKPOINT_SHA256)

    print("\nLoading checkpoint...")
    # weights_only=True restricts unpickling to tensors and plain containers;
    # the checkpoint only references OrderedDict / FloatStorage /
    # _rebuild_tensor_v2, all of which the safe loader allows.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print("Converting to float32...")
    converted = {
        key: tensor.float().contiguous()
        for key, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
    }

    total_params = sum(t.numel() for t in converted.values())
    print(f"Model has {len(converted)} tensors, {total_params:,} parameters")

    part = model_path.with_name(model_path.name + ".part")
    try:
        save_file(converted, str(part))
        os.replace(part, model_path)
    finally:
        if part.exists():
            part.unlink()
    print(f"Saved model.safetensors ({model_path.stat().st_size / 1e6:.0f} MB)")

    if not args.keep_checkpoint:
        checkpoint_path.unlink()
        print("Removed the PyTorch checkpoint (use --keep-checkpoint to retain it)")

    print(f"\nDone! Model saved to: {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
