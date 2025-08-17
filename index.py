#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from faiss import IndexFlatIP, write_index, normalize_L2
from PIL import Image
from tqdm import tqdm

import argparse
import clip
import faiss
import json
import numpy as np
import os
import sys
import torch
from pathlib import Path
import psutil

def log_mem(tag: str):
    """In-place snapshot of CPU & GPU usage (MiB)."""
    rss = psutil.Process().memory_info().rss / 2**20        # resident set size
    gpu = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
    print(f"[MEM] {tag:<20} | CPU {rss:8.1f} MiB | GPU {gpu:8.1f} MiB")

def iter_image_paths(root_dir, exts=None):
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    root = Path(root_dir)
    for cls in sorted(root.iterdir()):
        if cls.is_dir():
            for p in sorted(cls.iterdir()):
                if p.suffix.lower() in exts:
                    yield str(p)


def encode_in_batches(img_paths, model, preprocess, device, batch_size=256):
    feats = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Encoding images"):
        log_mem('before batch')
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(preprocess(img))
        batch = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            f = model.encode_image(batch).float()  # (B, d)

        # L2-norm (cosine = IP)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        log_mem('after batch')

    feats = torch.cat(feats, dim=0)  # (N, d)
    return feats.numpy().astype("float32")


def build_index(image_dir_path, output_dir="static", model_name="ViT-B/32", batch_size=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading CLIP {model_name} on {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    print("[INFO] Scanning images...")
    image_paths = list(iter_image_paths(image_dir_path))
    if not image_paths:
        print(f"[ERROR] No images found under: {image_dir_path}")
        sys.exit(1)
    print(f"[INFO] Found {len(image_paths)} images")

    # Encode → L2-normalize (already unit-norm), keep float32
    log_mem('before encode_in_batches')
    image_features = encode_in_batches(image_paths, model, preprocess, device, batch_size=batch_size)
    log_mem('after encode_in_batches')
    
    # (An toàn) normalize lại bằng FAISS (idempotent)
    normalize_L2(image_features)

    print("[INFO] Building FAISS IndexFlatIP")
    index = IndexFlatIP(image_features.shape[1])
    index.add(image_features)

    # Save
    index_path = os.path.join(output_dir, "index.faiss")
    paths_path = os.path.join(output_dir, "image_paths.json")
    write_index(index, index_path)
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved index to {index_path}")
    print(f"[DONE] Saved paths to {paths_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir_path", type=str, default="static/data/images")
    ap.add_argument("--output_dir", type=str, default="static")
    ap.add_argument("--model", type=str, default="ViT-B/32")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="", help="cpu|cuda (auto if empty)")
    args = ap.parse_args()

    build_index(
        image_dir_path=args.image_dir_path,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=(args.device or None),
    )


if __name__ == "__main__":
    main()
