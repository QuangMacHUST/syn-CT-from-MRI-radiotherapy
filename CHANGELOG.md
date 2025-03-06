# Nhật ký thay đổi

Tài liệu này ghi lại tất cả những thay đổi đáng chú ý đối với dự án "Chuyển đổi MRI sang CT mô phỏng cho xạ trị".

## [Phiên bản 1.1.0] - 2023-12-15

### Thêm mới
- Hỗ trợ bốn mô hình mới: UNet, Pix2Pix, AttentionGAN và UNIT
- Thêm cấu hình cho từng loại mô hình trong file config
- Tạo file README_EXTRA_MODELS.md để hướng dẫn sử dụng các mô hình mới

### Thay đổi
- Cập nhật file train.py để hỗ trợ huấn luyện tất cả các loại mô hình
- Cải thiện file inference.py để hỗ trợ suy luận với nhiều loại mô hình
- Sửa đổi cấu trúc config để chứa tham số cho các mô hình khác nhau

### Sửa lỗi
- Sửa lỗi linter trong file metrics.py liên quan đến việc import từ skimage.metrics
- Sửa các lỗi nhỏ trong quá trình xử lý dữ liệu

## [Phiên bản 1.0.0] - 2023-11-01

### Chức năng chính
- Xử lý tập dữ liệu DICOM từ ảnh MRI
- Mô hình học sâu không giám sát (CycleGAN) để chuyển đổi MRI sang CT
- Gán hệ số mô dựa trên thông tin tương phản của MRI
- Đánh giá kết quả với các chỉ số MAE, SSIM, PSNR
- Xuất kết quả dưới dạng ảnh DICOM tương thích với hệ thống lập kế hoạch xạ trị 