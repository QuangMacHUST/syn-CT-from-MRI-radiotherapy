# Hướng dẫn sử dụng các mô hình bổ sung

Dự án chuyển đổi MRI sang CT mô phỏng hiện đã hỗ trợ nhiều mô hình học sâu khác nhau ngoài CycleGAN. Tài liệu này sẽ giải thích cách sử dụng các mô hình mới này.

## Các mô hình có sẵn

Hiện tại, dự án hỗ trợ 5 loại mô hình chuyển đổi MRI sang CT:

1. **CycleGAN** (mặc định): Mô hình GAN không giám sát với cơ chế chu kỳ, hiệu quả khi không có dữ liệu ghép cặp.
2. **UNet**: Mô hình encoder-decoder đơn giản với các kết nối nhảy cóc, hiệu quả khi có dữ liệu ghép cặp.
3. **Pix2Pix**: Mô hình GAN có điều kiện, hiệu quả khi có dữ liệu ghép cặp MRI-CT.
4. **AttentionGAN**: Mô hình GAN với cơ chế chú ý, giúp tập trung vào các khu vực quan trọng của ảnh.
5. **UNIT**: Mô hình dựa trên VAE-GAN, chia sẻ không gian tiềm ẩn giữa hai miền.

## Cấu hình mô hình

Để chọn loại mô hình, bạn cần cập nhật file cấu hình `configs/default.yaml` (hoặc file cấu hình tùy chỉnh của bạn):

```yaml
# Cấu hình mô hình
model:
  type: cyclegan  # Loại mô hình: cyclegan, unet, pix2pix, attentiongan, unit
  input_nc: 1  # Số kênh đầu vào (MRI)
  output_nc: 1  # Số kênh đầu ra (CT)
  ngf: 64  # Số filter cơ bản cho generator
  ndf: 64  # Số filter cơ bản cho discriminator
  n_blocks: 9  # Số khối residual trong generator
  
  # Cấu hình riêng cho từng loại mô hình
  unet:
    bilinear: true  # Sử dụng upsampling bilinear thay vì transposed convolution
  
  pix2pix:
    use_dropout: true  # Sử dụng dropout trong generator
    norm_layer: instance  # Loại normalization: instance, batch
  
  unit:
    latent_dim: 512  # Kích thước không gian tiềm ẩn
    n_encoder_blocks: 3  # Số khối residual trong encoder
    n_decoder_blocks: 3  # Số khối residual trong decoder
```

Đối với suy luận (inference), bạn cũng cần cập nhật cấu hình:

```yaml
# Cấu hình chuyển đổi ảnh mô phỏng
inference:
  model_type: cyclegan  # Loại mô hình sử dụng: cyclegan, unet, pix2pix, attentiongan, unit
  checkpoint: best_model.pth
  # ... các cấu hình khác ...
```

## Huấn luyện mô hình

Để huấn luyện một mô hình cụ thể, chỉ cần cập nhật cấu hình và sử dụng lệnh:

```bash
python src/training/train.py --config configs/your_config.yaml
```

File huấn luyện `src/training/train.py` sẽ tự động tải loại mô hình phù hợp dựa trên cấu hình.

## Suy luận (Inference)

Để sử dụng mô hình đã huấn luyện để chuyển đổi MRI sang CT, sử dụng lệnh:

```bash
python src/inference.py --config configs/your_config.yaml --model path/to/model.pth --input_dir path/to/mri
```

Cấu hình `model_type` trong phần `inference` sẽ xác định loại mô hình được sử dụng.

## So sánh các mô hình

Mỗi mô hình có ưu và nhược điểm riêng:

| Mô hình | Ưu điểm | Nhược điểm | Khi nào sử dụng |
|---------|---------|------------|-----------------|
| CycleGAN | Hoạt động tốt với dữ liệu không ghép cặp, bảo toàn cấu trúc tốt | Chậm để huấn luyện, cần nhiều bộ nhớ | Khi không có dữ liệu ghép cặp MRI-CT |
| UNet | Đơn giản, nhanh, ít tham số | Chỉ hoạt động tốt với dữ liệu ghép cặp, có thể mờ ở một số chi tiết | Khi có dữ liệu ghép cặp MRI-CT tốt |
| Pix2Pix | Chất lượng tốt với dữ liệu ghép cặp, chi tiết sắc nét | Cần dữ liệu ghép cặp, có thể tạo ra các cấu trúc không có thật | Khi có dữ liệu ghép cặp MRI-CT chất lượng cao |
| AttentionGAN | Tập trung vào các vùng quan trọng, kết quả có chi tiết tốt | Phức tạp hơn, khó huấn luyện | Khi cần chú ý đến một số vùng cụ thể (ví dụ: khối u) |
| UNIT | Bảo toàn nội dung tốt, hoạt động được với dữ liệu không ghép cặp | Phức tạp, cần nhiều tham số, khó huấn luyện | Khi cần tái tạo chi tiết mà vẫn bảo toàn cấu trúc chung |

## Mở rộng thêm

Để thêm mô hình mới vào dự án:

1. Tạo file mô hình mới trong thư mục `src/models/`
2. Đăng ký mô hình trong `src/models/__init__.py`
3. Cập nhật file cấu hình để hỗ trợ mô hình mới
4. Cập nhật `src/inference.py` để xử lý đúng đầu ra của mô hình mới

## Tài liệu tham khảo

- CycleGAN: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/)
- Pix2Pix: [Image-to-Image Translation with Conditional Adversarial Networks](https://phillipi.github.io/pix2pix/)
- UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- AttentionGAN: [Attention-Guided Generative Adversarial Networks for Unsupervised Image-to-Image Translation](https://arxiv.org/abs/1903.12296)
- UNIT: [Unsupervised Image-to-Image Translation Networks](https://arxiv.org/abs/1703.00848) 