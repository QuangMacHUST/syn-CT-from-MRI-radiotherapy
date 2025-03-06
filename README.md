# Chuyển đổi MRI sang CT mô phỏng cho xạ trị

Dự án này sử dụng kỹ thuật học sâu không giám sát để chuyển đổi ảnh MRI thành ảnh CT mô phỏng có gán hệ số mô để phục vụ cho xạ trị. Trong khi ảnh MRI cung cấp độ tương phản tổ chức tốt, ảnh CT cung cấp thông tin về mật độ electron cần thiết cho việc lên kế hoạch xạ trị. Mô hình này cho phép sử dụng dữ liệu MRI để tạo ra ảnh CT mô phỏng với các hệ số mô chính xác.

## Tính năng

- Xử lý tập dữ liệu DICOM từ ảnh MRI
- Mô hình học sâu không giám sát (CycleGAN) để chuyển đổi MRI sang CT
- Gán hệ số mô dựa trên thông tin tương phản của MRI
- Đánh giá kết quả với các chỉ số MAE, SSIM, PSNR
- Xuất kết quả dưới dạng ảnh DICOM tương thích với hệ thống lập kế hoạch xạ trị

## Cấu trúc dự án

- `data/`: Thư mục chứa dữ liệu đầu vào và đầu ra
- `src/`: Mã nguồn chính của dự án
  - `data_processing/`: Module xử lý dữ liệu DICOM
  - `models/`: Các mô hình học sâu
  - `training/`: Pipeline huấn luyện
  - `evaluation/`: Công cụ đánh giá mô hình
  - `utils/`: Các hàm tiện ích
- `notebooks/`: Jupyter notebooks cho trực quan hóa và thí nghiệm
- `configs/`: Cấu hình cho các thí nghiệm khác nhau

## Yêu cầu

- Python 3.8+
- PyTorch 1.8+
- CUDA (cho GPU training)
- pydicom
- nibabel
- scikit-image
- numpy, pandas, matplotlib

## Cài đặt

```bash
pip install -r requirements.txt
```

## Sử dụng

1. Đặt dữ liệu DICOM vào thư mục `data/raw/`
2. Chạy tiền xử lý: `python src/data_processing/preprocess.py`
3. Huấn luyện mô hình: `python src/training/train.py --config configs/default.yaml`
4. Sinh ảnh CT mô phỏng: `python src/inference.py --input path/to/mri --output path/to/output`
5. Đánh giá kết quả: `python src/evaluation/evaluate.py --pred path/to/pred --gt path/to/ground_truth`
