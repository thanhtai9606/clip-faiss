#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from faiss import read_index, normalize_L2
from PIL import Image
import clip
import json
import torch
import numpy as np
import os


class App:
    def __init__(self, index_path="static/index.faiss", paths_path="static/image_paths.json", model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.index = read_index(index_path)
        with open(paths_path, "r", encoding="utf-8") as f:
            self.image_paths = json.load(f)

        self.dim = self.index.d
        self.db_size = len(self.image_paths)

    def _search_vec(self, q_vec, topk=5):
        """
        q_vec: torch.Tensor, shape (1, d)
        returns: list of dicts {path, score}
        """
        # L2-norm (cosine = IP)
        q_vec = q_vec / q_vec.norm(dim=-1, keepdim=True)

        # numpy float32 for FAISS
        q = q_vec.detach().cpu().numpy().astype("float32")

        # safe normalize (idempotent)
        normalize_L2(q)

        topk = int(min(max(1, topk), self.db_size))
        D, I = self.index.search(q, topk)  # D: cosine scores, I: indices

        I = I[0].tolist()
        D = D[0].tolist()
        return [{"path": self.image_paths[i], "score": float(s)} for i, s in zip(I, D)]

    def search_text(self, text, results=5):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            t_feat = self.model.encode_text(tokens).float()
        return self._search_vec(t_feat, topk=results)

    def search_image(self, image_or_path, results=5):
        if isinstance(image_or_path, str):
            img = Image.open(image_or_path).convert("RGB")
        else:
            img = image_or_path.convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            i_feat = self.model.encode_image(img_t).float()
        return self._search_vec(i_feat, topk=results)

    # Optional CLI
    def run(self):
        while True:
            q = input("Search (text or path to image, 'exit' to quit): ").strip()
            if q.lower() == "exit":
                break
            if os.path.isfile(q):
                results = self.search_image(q, results=5)
            else:
                results = self.search_text(q, results=5)
            for rank, r in enumerate(results, 1):
                print(f"{rank:>2}. {r['path']} | score={r['score']:.4f}")
