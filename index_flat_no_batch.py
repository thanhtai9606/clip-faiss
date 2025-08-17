#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import clip
import faiss
from PIL import Image
from tqdm import tqdm

def iter_image_paths(root_dir, exts=None):
    """
    Tạo một generator để lặp qua tất cả các file ảnh trong một thư mục (cấu trúc phẳng).
    """
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    root = Path(root_dir)
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield str(p)

def build_index_no_batch(image_dir_path, output_dir="static", model_name="ViT-B/32", device=None):
    """
    Hàm này được sửa đổi để encode TOÀN BỘ ảnh trong một lần, mô phỏng lại hành vi gốc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Tải mô hình CLIP {model_name} trên {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    print("[INFO] Quét ảnh...")
    image_paths = list(iter_image_paths(image_dir_path))
    if not image_paths:
        print(f"[ERROR] Không tìm thấy ảnh nào trong: {image_dir_path}")
        sys.exit(1)
    print(f"[INFO] Tìm thấy {len(image_paths)} ảnh")

    # ===== LOGIC MÔ PHỎNG KHÔNG BATCH =====
    print("[INFO] Bắt đầu encode toàn bộ ảnh (KHÔNG BATCH)...")
    print("[WARNING] Quá trình này có thể gây lỗi Out of Memory với dataset lớn.")

    all_images_preprocessed = []
    # 1. Preprocess tất cả ảnh và nạp vào RAM
    for p in tqdm(image_paths, desc="Preprocessing images (1/2)"):
        try:
            img = Image.open(p).convert("RGB")
            all_images_preprocessed.append(preprocess(img))
        except Exception as e:
            print(f"\n[WARNING] Lỗi ảnh {p}, bỏ qua. Lỗi: {e}")
            
    if not all_images_preprocessed:
        print("[ERROR] Không có ảnh nào được preprocess thành công.")
        sys.exit(1)

    image_features = None
    # 2. Xếp chồng (stack) tất cả thành 1 tensor lớn và encode
    with torch.no_grad():
        print("[INFO] Xếp chồng các tensor và encode... (2/2)")
        tensor_batch = torch.stack(all_images_preprocessed).to(device)
        f = model.encode_image(tensor_batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        image_features = f.cpu().numpy().astype("float32")
    # ====================================

    if image_features is None:
        print("[ERROR] Không có ảnh nào được encode thành công.")
        sys.exit(1)

    print("[INFO] (An toàn) Chuẩn hóa L2 lại bằng FAISS (idempotent)")
    faiss.normalize_L2(image_features)

    print(f"[INFO] Xây dựng FAISS IndexFlatIP với {image_features.shape[1]} chiều")
    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)

    index_path = os.path.join(output_dir, "index.faiss")
    paths_path = os.path.join(output_dir, "image_paths.json")
    
    faiss.write_index(index, index_path)
    with open(paths_path, "w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Đã lưu index vào {index_path}")
    print(f"[DONE] Đã lưu đường dẫn vào {paths_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir_path", type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh.")
    ap.add_argument("--output_dir", type=str, default="static")
    ap.add_argument("--model", type=str, default="ViT-B/32")
    ap.add_argument("--device", type=str, default="", help="cpu|cuda (tự động nếu rỗng)")
    args = ap.parse_args()

    build_index_no_batch(
        image_dir_path=args.image_dir_path,
        output_dir=args.output_dir,
        model_name=args.model,
        device=(args.device or None),
    )

if __name__ == "__main__":
    main()