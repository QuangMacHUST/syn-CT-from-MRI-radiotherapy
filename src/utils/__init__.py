"""
Module chứa các tiện ích hỗ trợ.

Bao gồm các hàm và lớp để đánh giá kết quả, trực quan hóa, xử lý dữ liệu.
"""

from .visualization import (
    tensor_to_numpy, 
    normalize_for_display, 
    save_images, 
    visualize_results
)

from .metrics import (
    calculate_metrics, 
    calculate_tissue_metrics, 
    calculate_hu_metrics
) 