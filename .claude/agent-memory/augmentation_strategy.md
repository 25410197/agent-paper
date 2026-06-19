---
name: augmentation-strategy
description: Chiến lược augmentation cho dataset ot-mixed (ớt) dựa trên phân tích trực quan ảnh thực tế
metadata:
  type: project
  updated: 2026-05-27
  dataset: dataset_smartcrop_300 (anthracnose, healthy, leaf_curl, leaf_spot, yellow_leaf)
---

# Chiến lược Augmentation — Dataset Ớt (ot-mixed)

## Đặc điểm bộ ảnh (quan sát trực tiếp)

| Class | Đối tượng | Đặc trưng chẩn đoán |
|-------|----------|-------------------|
| anthracnose | **Quả ớt** (không phải lá!) — dài, mảnh, cong | Đốm vàng/nâu nhỏ ở đầu/thân quả |
| healthy | Lá ớt — đứng, xanh đều | Màu xanh đồng nhất |
| leaf_curl | Lá — xoăn/biến dạng mạnh | Hình dạng méo là đặc trưng chính |
| leaf_spot | Lá — có 1–nhiều đốm tròn nhỏ | Đốm có viền rõ |
| yellow_leaf | Lá — vàng loang lổ, gân đen | Màu vàng-xanh loang là đặc trưng chính |

**Đặc điểm chung:**
- Nền trắng/xám sáng đồng nhất (ảnh studio kiểm soát)
- Orientation rất đa dạng: đứng, nằm ngang, chéo ~45°, cong cung
- Một số ảnh chụp từ mặt dưới lá (gân nổi rõ)
- ~400–550 ảnh mỗi class (cân bằng tương đối)

## Template Augmentation Tốt Nhất Cho Dataset Này

### v1 (2026-05-26) — thử nghiệm đầu tiên, kết quả TỆ HƠN baseline 10%
Nguyên nhân thất bại: 7 transform áp dụng liên tiếp mỗi sample tích lũy quá nhiều distortion.
rotation=45 + crop=0.75 + perspective + blur_sigma=1.0 cùng lúc làm mất đặc trưng bệnh.

### v2 (2026-05-27) — conservative revision, RECOMMENDED

```python
transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # Lật ngang — tất cả class đều phù hợp
    transforms.RandomHorizontalFlip(p=0.5),
    # Lật dọc — lá/quả chụp cả 2 chiều trên/dưới
    transforms.RandomVerticalFlip(p=0.3),
    # Xoay ±20° (giảm từ 45°) — orientation đa dạng nhưng tránh tích lũy distortion
    transforms.RandomRotation(degrees=20),
    # Zoom tối thiểu 0.80 (tăng từ 0.75) — bảo vệ đốm nhỏ leaf_spot (~5-10px trên 300px)
    transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.80, 1.0)),
    # ColorJitter — brightness/contrast giảm xuống 0.2 (ảnh studio đã kiểm soát ánh sáng)
    #               saturation và hue giữ rất thấp
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1, hue=0.02),
    # Blur rất nhẹ — sigma giảm xuống 0.5 để không làm mờ đốm nhỏ leaf_spot
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
```

**Những gì đã bỏ so với v1:**
- `RandomPerspective` — gây warp noise, không có lợi ích chẩn đoán thực tế với ảnh studio nền trắng

## Quy tắc Augmentation cho Dataset Bệnh Cây Có Màu Chẩn Đoán

| Kỹ thuật | Mức an toàn | Lý do |
|---------|------------|-------|
| Horizontal/Vertical Flip | AN TOÀN | Không thay đổi màu và hình dạng đặc trưng |
| Rotation ≤ 20° | AN TOÀN | Orientation tự nhiên đa dạng; 45° tích lũy với crop gây over-distortion |
| RandomResizedCrop scale ≥ 0.80 | AN TOÀN | Đốm leaf_spot chỉ ~5-10px trên 300px, cần scale cao |
| RandomPerspective | **TRÁNH** | Gây warp noise không cần thiết với ảnh studio nền trắng đồng nhất |
| GaussianBlur nhẹ (σ ≤ 0.5) | AN TOÀN nhẹ | Chỉ simulate rung tay nhẹ; σ > 0.5 làm mờ đốm nhỏ |
| Brightness/Contrast ≤ 0.2 | AN TOÀN | Ảnh studio đã kiểm soát ánh sáng; 0.3 là quá mạnh |
| Saturation ≤ 0.1 | **BẮT BUỘC THẤP** | Màu vàng (yellow_leaf) là đặc trưng chẩn đoán |
| Hue ≤ 0.02 | **BẮT BUỘC RẤT THẤP** | Đổi hue làm vàng→xanh hoặc xanh→vàng = nhầm class |
| RandomErasing | **TRÁNH** | Có thể xóa đốm bệnh nhỏ của leaf_spot và anthracnose |
| ElasticTransform | **TRÁNH** | leaf_curl dùng hình dạng méo làm đặc trưng — elastic làm mờ |
| Grayscale | **TRÁNH** | Màu là đặc trưng chính của yellow_leaf |
| Strong Crop (scale < 0.7) | **TRÁNH** | Mất đốm nhỏ của leaf_spot/anthracnose |

## Quy tắc chung áp dụng cho mọi dataset bệnh cây

1. **Kiểm tra xem màu sắc có phải đặc trưng chẩn đoán không** → nếu có, saturation ≤ 0.1, hue ≤ 0.02
2. **Kiểm tra xem đốm/vùng bệnh có nhỏ không** → nếu có, scale ≥ 0.80, tránh RandomErasing
3. **Kiểm tra xem hình dạng biến dạng có phải đặc trưng không** (như leaf_curl) → tránh ElasticTransform
4. **Nền đồng nhất (trắng/xám)** → Perspective KHÔNG cần thiết vì nền không gây nhiễu dù sao; Rotation an toàn
5. **Thêm RandomVerticalFlip** khi dataset chụp từ cả mặt trên/dưới của đối tượng
6. **CẢNH BÁO TÍCH LŨY**: Khi dùng nhiều transform, tổng distortion tích lũy có thể vượt ngưỡng an toàn dù từng transform nhỏ. Nguyên tắc: không quá 5 geometric/color transforms cùng lúc. Nếu augmented < baseline > 5%, giảm rotation và crop scale trước tiên.
7. **Ảnh studio vs ảnh thực địa**: Ảnh studio (nền trắng, ánh sáng kiểm soát) cần brightness/contrast thấp hơn (≤ 0.2). Ảnh thực địa mới cần brightness/contrast cao hơn (0.3–0.4).
