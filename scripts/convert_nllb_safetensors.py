#!/usr/bin/env python3
"""
Convert NLLB-200 distilled-600M from PyTorch to safetensors format.
This enables loading in candle (Rust) without Python dependencies.

Usage:
    python scripts/convert_nllb_safetensors.py [--output-dir ~/.slopshift/models/nllb-candle]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert NLLB to safetensors")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / ".slopshift/models/nllb-candle"),
        help="Output directory for safetensors model",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/nllb-200-distilled-600M",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Data type for saved weights",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    if (output_dir / "model.safetensors").exists():
        size_mb = (output_dir / "model.safetensors").stat().st_size / 1e6
        print(f"model.safetensors already exists ({size_mb:.0f} MB)")
        print(f"Delete {output_dir / 'model.safetensors'} to reconvert.")
        return

    print(f"Converting {args.model_id} to safetensors...")
    print(f"Output: {output_dir}")
    print()

    # Install dependencies if needed
    try:
        import torch
        import transformers
        from safetensors.torch import save_file
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing transformers, torch, safetensors...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "transformers", "torch", "safetensors", "sentencepiece"
        ])
        import torch
        import transformers
        from safetensors.torch import save_file

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Download and load model
    print(f"Downloading {args.model_id} from HuggingFace...")
    print("(This downloads ~2.4GB on first run, cached after that)")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

    # Convert to target dtype
    if args.dtype == "float16":
        print("Converting to float16...")
        model = model.half()

    # Save as safetensors — handle shared/tied weights
    # NLLB ties encoder embed, decoder embed, lm_head, and model.shared
    # We keep model.shared as the canonical copy and skip duplicates
    print("Saving model.safetensors...")
    state_dict = model.state_dict()

    # Remove duplicate tied weight entries (they share the same tensor memory)
    shared_weight = state_dict.get("model.shared.weight")
    keys_to_remove = []
    if shared_weight is not None:
        for key in ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]:
            if key in state_dict and state_dict[key].data_ptr() == shared_weight.data_ptr():
                keys_to_remove.append(key)
        if "lm_head.weight" in state_dict and state_dict["lm_head.weight"].data_ptr() == shared_weight.data_ptr():
            keys_to_remove.append("lm_head.weight")

    for key in keys_to_remove:
        del state_dict[key]
        print(f"  Removed tied weight: {key} (shared with model.shared.weight)")

    save_file(state_dict, str(output_dir / "model.safetensors"))

    size_mb = (output_dir / "model.safetensors").stat().st_size / 1e6
    print(f"Saved model.safetensors ({size_mb:.0f} MB)")

    # Copy tokenizer.json
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(str(output_dir))

    # Also save config.json for reference
    model.config.save_pretrained(str(output_dir))

    # Clean up unnecessary files (keep only what candle needs)
    essential_files = {
        "model.safetensors", "tokenizer.json", "config.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "sentencepiece.bpe.model",
    }
    for f in output_dir.iterdir():
        if f.name not in essential_files:
            if f.is_file():
                f.unlink()
                print(f"  Removed {f.name} (not needed by candle)")

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
