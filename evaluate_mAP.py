import os
import json
import argparse
from PIL import Image
import numpy as np
import torch
import faiss
import clip
from tqdm import tqdm

def calculate_ap(ranked_list, relevant_set):
    """
    Tính Average Precision (AP) cho một danh sách kết quả đã xếp hạng.
    """
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(ranked_list):
        if item in relevant_set:
            hits += 1
            precision_at_k = hits / (i + 1)
            sum_precisions += precision_at_k
    
    if not relevant_set:
        return 0.0
    return sum_precisions / len(relevant_set)

def main(args):
    """
    Hàm chính để chạy toàn bộ quá trình đánh giá mAP.
    """
    print(f"[INFO] Bắt đầu quá trình đánh giá mAP...")
    print(f"[INFO] Sử dụng chuẩn hóa L2 cho query: {'Có' if args.normalize else 'Không'}")
    
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"[INFO] Sử dụng device: {device}")
    
    # 1. Tải các tài nguyên cần thiết
    print(f"[INFO] Tải mô hình CLIP '{args.clip_model}'...")
    model, preprocess = clip.load(args.clip_model, device=device)
    
    print(f"[INFO] Tải FAISS index từ '{args.index_path}'...")
    index = faiss.read_index(args.index_path)
    num_db_images = index.ntotal
    
    print(f"[INFO] Tải danh sách đường dẫn ảnh từ '{args.paths_json}'...")
    with open(args.paths_json, 'r') as f:
        image_paths = json.load(f)

    # 2. Tải ground truth từ file JSON
    print(f"[INFO] Tải ground truth từ '{args.gt_json}'...")
    with open(args.gt_json, 'r') as f:
        groundtruth_data = json.load(f)
    
    all_ap_scores = []
    total_queries_evaluated = 0

    print(f"[INFO] Tìm thấy {len(groundtruth_data)} chủ đề query để đánh giá.")
    
    # 3. Lặp qua từng chủ đề query (all_souls, etc.)
    for landmark_name, gt in tqdm(groundtruth_data.items(), desc="Đánh giá các chủ đề"):
        # Lấy danh sách relevant và junk chung cho chủ đề này
        # Chuyển tên file có đuôi (.jpg) thành không có đuôi để so khớp
        good_files = {os.path.splitext(f)[0] for f in gt.get('good', [])}
        ok_files = {os.path.splitext(f)[0] for f in gt.get('ok', [])}
        junk_files = {os.path.splitext(f)[0] for f in gt.get('junk', [])}
        relevant_files = good_files.union(ok_files)

        # Lặp qua từng ảnh query trong danh sách "query" của chủ đề
        for query_img_filename in gt['query']:
            query_full_path = os.path.join(args.images_dir, query_img_filename)
            try:
                # Dùng toàn bộ ảnh, không crop vì không có bbox
                query_img = Image.open(query_full_path).convert("RGB")
            except FileNotFoundError:
                print(f"[WARNING] Không tìm thấy ảnh query: {query_full_path}. Bỏ qua.")
                continue
            
            # Encode ảnh query
            with torch.no_grad():
                q_vec = model.encode_image(preprocess(query_img).unsqueeze(0).to(device)).float()

            if args.normalize:
                q_vec = q_vec / q_vec.norm(dim=-1, keepdim=True)

            q = q_vec.cpu().numpy().astype('float32')
            if args.normalize:
                faiss.normalize_L2(q)

            # Thực hiện tìm kiếm
            distances, indices = index.search(q, k=num_db_images)
            
            # Xử lý kết quả (lấy tên file không có đuôi)
            ranked_names = [os.path.splitext(os.path.basename(image_paths[i]))[0] for i in indices[0]]
            filtered_ranked_names = [name for name in ranked_names if name not in junk_files]
            
            # Tính AP cho query này
            ap = calculate_ap(filtered_ranked_names, relevant_files)
            all_ap_scores.append(ap)
            total_queries_evaluated += 1

    # 4. Tính mAP
    if not all_ap_scores:
        print("[ERROR] Không có query nào được đánh giá. Kết thúc.")
        return
        
    mAP = np.mean(all_ap_scores)
    
    print("\n" + "="*50)
    print("         KẾT QUẢ ĐÁNH GIÁ HOÀN TẤT")
    print("="*50)
    print(f"Tổng số query đã đánh giá: {total_queries_evaluated}")
    print(f"Chuẩn hóa L2 cho query: {'Có' if args.normalize else 'Không'}")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá mAP cho hệ thống truy vấn ảnh trên bộ Oxford5k.")
    
    parser.add_argument("--index_path", type=str, default="static/index.faiss", help="Đường dẫn đến file index.faiss.")
    parser.add_argument("--paths_json", type=str, default="static/image_paths.json", help="Đường dẫn đến file image_paths.json.")
    parser.add_argument("--gt_json", type=str, required=True, help="Đường dẫn đến file groundtruth.json.")
    parser.add_argument("--images_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa tất cả ảnh của bộ dataset (images).")
    
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="Tên mô hình CLIP.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device để chạy mô hình (cuda hoặc cpu).")
    
    parser.add_argument("--normalize", action='store_true', help="Thêm cờ này để BẬT chuẩn hóa L2 cho vector query (cho hệ thống cải tiến).")

    args = parser.parse_args()
    main(args)