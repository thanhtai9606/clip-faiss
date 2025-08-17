# Hệ Thống Truy Vấn Hình Ảnh Đa Phương Thức Sử Dụng CLIP và FAISS

Đây là đồ án môn học CS2224.CH190: Tìm kiếm thông tin thị giác, được phát triển và cải tiến dựa trên project mã nguồn mở `abinthomasonline/clip-faiss`.

## Đội ngũ phát triển

- Lê Thanh Dũng (230101074)
- Văn Đức Ngọ (240101020)
- Nguyễn Hoàng Hải (240101008)
- Trần Quốc Huy (230101048)
- Võ Lê Phú Xuân (240101032)

---

## Giới thiệu

Project này xây dựng một hệ thống tìm kiếm hình ảnh mạnh mẽ, cho phép người dùng truy vấn bằng cả văn bản (text-to-image) và hình ảnh (image-to-image). Hệ thống được xây dựng dựa trên hai công nghệ cốt lõi:

- **OpenAI CLIP:** Để trích xuất các vector đặc trưng (embedding) có ngữ nghĩa sâu sắc từ cả hình ảnh và văn bản.
- **Meta FAISS:** Để lưu trữ và thực hiện tìm kiếm tương đồng trên không gian vector lớn một cách cực kỳ hiệu quả.

So với project gốc, phiên bản này đã được cải tiến đáng kể về:
- **Chức năng:** Hỗ trợ truy vấn Image-to-Image.
- **Hiệu năng:** Tối ưu hóa quá trình xử lý dữ liệu lớn bằng Batch Encoding, giải quyết triệt để lỗi tràn bộ nhớ.
- **Trải nghiệm người dùng:** Giao diện web được thiết kế lại, trực quan và cung cấp nhiều thông tin hữu ích hơn.

---

## Hướng dẫn sử dụng

### Bước 1: Cài đặt môi trường

1.  **Clone repository:**
    ```bash
    git clone https://github.com/thanhtai9606/clip-faiss.git
    cd clip-faiss
    ```

2.  **Cài đặt các thư viện cần thiết:**
    (Khuyến khích tạo một môi trường ảo trước)
    ```bash
    pip install -r requirements.txt
    ```

### Bước 2: Chuẩn bị dữ liệu

1.  Tạo một thư mục để chứa dữ liệu, ví dụ: `Data/your_dataset_name/images`.
2.  Đặt tất cả các hình ảnh bạn muốn tìm kiếm vào thư mục `images` vừa tạo.

### Bước 3: Tạo Index cho dữ liệu

Chạy script `index.py` và trỏ đến thư mục chứa ảnh của bạn. Quá trình này sẽ tạo ra hai file `index.faiss` và `image_paths.json` trong thư mục `static/`.

```bash
python index.py --image_dir_path "Data/your_dataset_name/images"
