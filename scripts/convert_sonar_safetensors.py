#!/usr/bin/env python3
"""
Convert SONAR text encoder from fairseq2 checkpoint to safetensors format.
This enables loading in candle (Rust) without Python dependencies.

Downloads directly from Meta's CDN — no sonar-space or fairseq2 required.
Only needs: torch, safetensors, sentencepiece, tokenizers

Usage:
    python scripts/convert_sonar_safetensors.py [--output-dir ~/.cache/lingo/sonar]
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/SONAR/sonar_text_encoder.pt"
SPM_URL = "https://dl.fbaipublicfiles.com/SONAR/sentencepiece.source.256000.model"


def download_file(url, dest):
    """Download a file with progress indicator."""
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  {dest.name} already downloaded ({size_mb:.1f} MB)")
        return

    print(f"  Downloading {dest.name} from {url}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100.0 / total_size)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)
    print()


def main():
    parser = argparse.ArgumentParser(description="Convert SONAR text encoder to safetensors")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / ".cache/lingo/sonar"),
        help="Output directory for safetensors model",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    if (output_dir / "model.safetensors").exists() and (output_dir / "tokenizer.json").exists():
        size_mb = (output_dir / "model.safetensors").stat().st_size / 1e6
        print(f"model.safetensors already exists ({size_mb:.0f} MB)")
        print(f"Delete {output_dir / 'model.safetensors'} to reconvert.")
        return

    print("Converting SONAR text encoder to safetensors...")
    print(f"Output: {output_dir}")
    print()

    # Install dependencies if needed
    try:
        import torch
        from safetensors.torch import save_file
        import sentencepiece as spm
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing torch, safetensors, sentencepiece, tokenizers...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "safetensors", "sentencepiece", "tokenizers",
        ])
        import torch
        from safetensors.torch import save_file
        import sentencepiece as spm

    # Download checkpoint and SPM tokenizer from Meta's CDN
    checkpoint_path = output_dir / "sonar_text_encoder.pt"
    spm_path = output_dir / "sentencepiece.source.256000.model"

    download_file(CHECKPOINT_URL, checkpoint_path)
    download_file(SPM_URL, spm_path)

    # Load PyTorch checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # The checkpoint may have a "model" key or be the state dict directly
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Convert all tensors to float32 for safetensors compatibility
    print("Converting to float32...")
    converted = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            converted[key] = tensor.float().contiguous()

    # Print weight names
    print(f"\nModel has {len(converted)} tensors:")
    total_params = 0
    for key in sorted(converted.keys()):
        shape = list(converted[key].shape)
        params = converted[key].numel()
        total_params += params
        print(f"  {key:60s} {str(shape):>20s}  ({params:>12,} params)")
    print(f"\nTotal parameters: {total_params:,}")

    # Save as safetensors
    print(f"\nSaving model.safetensors...")
    save_file(converted, str(output_dir / "model.safetensors"))
    size_mb = (output_dir / "model.safetensors").stat().st_size / 1e6
    print(f"Saved model.safetensors ({size_mb:.0f} MB)")

    # Convert SentencePiece tokenizer to HuggingFace tokenizers JSON format
    print("\nConverting tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))

    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder

    # Build Unigram vocab: list of (piece, log_prob) tuples
    vocab_list = []
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        score = sp.GetScore(i)
        vocab_list.append((piece, score))

    tok = Tokenizer(Unigram(vocab_list))
    tok.pre_tokenizer = Metaspace()
    tok.decoder = MetaspaceDecoder()

    tok.save(str(output_dir / "tokenizer.json"))
    print(f"Saved tokenizer.json ({sp.GetPieceSize()} tokens)")

    # Clean up raw checkpoint (keep safetensors + tokenizer.json + spm model)
    print()
    print(f"Done! Model saved to: {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1e6:
            print(f"  {f.name:40s} {size/1e6:.1f} MB")
        else:
            print(f"  {f.name:40s} {size/1e3:.1f} KB")


if __name__ == "__main__":
    main()
