#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from app import App
from PIL import Image
import io
import os

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB upload
flask_app.config["UPLOAD_EXTENSIONS"] = {".jpg", ".jpeg", ".png"}

# Lazy init App (avoid reload model every request)
APP = App()


@flask_app.route("/")
def index():
    # Bạn có thể tạo templates/index.html hoặc trả minimal HTML dưới đây
    return render_template("index.html") if os.path.exists("templates/index.html") else """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>CLIP + FAISS Search</title></head>
  <body>
    <h3>Text Search</h3>
    <form action="/search" method="get">
      <input name="search_query" placeholder="e.g., Radcliffe Camera" style="width:300px;">
      <button type="submit">Search</button>
    </form>

    <h3>Image → Image Search</h3>
    <form action="/search-image" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*">
      <button type="submit">Search by Image</button>
    </form>
  </body>
</html>
    """


@flask_app.route("/search", methods=["GET"])
def search_text():
    search_query = request.args.get("search_query", "").strip()
    topk = int(request.args.get("k", 5))
    if not search_query:
        return jsonify({"error": "Missing search_query"}), 400

    results = APP.search_text(search_query, results=topk)
    return jsonify({"query": search_query, "k": topk, "results": results})


@flask_app.route("/search-image", methods=["POST"])
def search_image():
    if "file" not in request.files:
        return jsonify({"error": "No file field"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename or "")
    _, ext = os.path.splitext(filename)
    if (not filename) or (ext.lower() not in flask_app.config["UPLOAD_EXTENSIONS"]):
        return jsonify({"error": "Invalid or missing file (jpg/jpeg/png)"}), 400

    data = f.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"error": "Cannot open image"}), 400

    topk = int(request.args.get("k", 5))
    results = APP.search_image(img, results=topk)
    return jsonify({"k": topk, "results": results})


if __name__ == "__main__":
    # Chạy: python serve.py  (mặc định 127.0.0.1:5000)
    flask_app.run(host="0.0.0.0", port=5000, debug=False)
