# Trả lời góp ý — TS. Phùng Thế Bảo

Nguồn số liệu: `outputs/results/de-xuat/` (1 seed = 42, 30 epoch, test 373 ảnh).
Mô hình đề xuất đã thay đổi theo chính các góp ý: cắt backbone tại `features[1:14]`
(96 kênh @ 14×14 = 196 token), **671 732 tham số** — thay cho bản cũ `features[1:15]`
(160 kênh @ 7×7 = 49 token, 835 188 tham số). Toàn bộ số liệu dưới đây là đo lại.

**Tóm tắt:** 6/9 điểm đã xử lý xong; 2 điểm là sửa bản thảo; 1 điểm chưa làm được.

| # | Góp ý | Trạng thái |
|---|---|---|
| 1 | Số liệu +0,36 pp không nhất quán | ✅ Đã bỏ |
| 2 | Thiếu ablation từng kỹ thuật (2D-RoPE / GQA / RMSNorm) | ✅ Đã bổ sung |
| 3 | RTR_Lite_MobileNet chỉ 1 run | ❌ Chưa làm — đề xuất phương án thay thế |
| 4 | Cần ablation huấn luyện lại | ✅ Đã bổ sung |
| 5 | Domain gap ảnh nền trắng | ⚠️ Một phần |
| 6 | Tập test nhỏ, không CI/kiểm định | ✅ Đã bổ sung |
| 7 | Related Work sơ lược | 🖊 Sửa bản thảo |
| 8 | Thiếu phân bố lớp + cấu hình CPU | ✅ Đã bổ sung |
| 9 | Đánh số bảng/hình, diễn đạt | 🖊 Sửa bản thảo |

---

## 1. Số liệu +0,36 pp — Thầy đúng ✅

98,57% và 98,93% đúng là hai kịch bản **của cùng mô hình đề xuất** (baseline vs augmented),
không phải so với RTR_Lite_MobileNet. Câu này đã được **bỏ hoàn toàn**. Mô hình đề xuất đã
đổi nên mọi số được đo lại từ đầu; không còn tuyên bố "gain over the original" nào dựa trên
phép so sai đó.

---

## 2. Ablation từng kỹ thuật — đã chạy đủ ba câu hỏi ✅

Mỗi biến thể thay **đúng một** kỹ thuật bằng phương án tiêu chuẩn, giữ nguyên phần còn lại,
rồi huấn luyện lại từ đầu (cùng seed / recipe / epoch).

| biến thể | thay gì | Δ tham số | clean | blur | dark | noise | jpeg | TB |
|---|---|---|---|---|---|---|---|---|
| **đề xuất** | RoPE + GQA + RMSNorm | — | 98,66 | 91,69 | 97,05 | **92,23** | 92,49 | 94,42 |
| `learned_pe` | 2D-RoPE → PE học được | **+25 088** | 98,93 | 95,71 | 98,66 | 94,64 | 94,64 | 96,51 |
| `mha` | GQA → Multi-Head Attention | **+16 384** | 98,39 | 93,03 | 98,39 | 89,81 | 95,44 | 95,01 |
| `layernorm` | RMSNorm → LayerNorm | +384 | 98,12 | 89,81 | 95,44 | 95,71 | 90,88 | 93,99 |
| `vanilla` | **cả ba** → khối ViT tiêu chuẩn | **+41 856** | 98,39 | 94,91 | 98,39 | 87,94 | 94,91 | 94,91 |

**Trả lời trực tiếp ba câu hỏi của Thầy:**

- **2D-RoPE vs PE học được:** PE học được tốn thêm **25 088 tham số** (2D-RoPE không có tham số
  nào) và nhỉnh hơn ~2 pp trung bình.
- **GQA vs MHA:** GQA tiết kiệm **16 384 tham số**; MHA hơn ở jpeg (+2,95 pp) nhưng **kém hơn ở
  noise** (−2,41 pp). Ở quy mô này GQA không còn ưu thế bộ nhớ như trong LLM — đúng như Thầy dự
  đoán — nhưng vẫn giữ được độ bền nhiễu.
- **RMSNorm vs LayerNorm:** chênh lệch nhỏ (384 tham số), LayerNorm hơn ở noise, kém ở blur/jpeg.

**Kết quả quan trọng nhất — so cả cụm (đây mới là claim của bài):**

| | tham số | TB 5 điều kiện |
|---|---|---|
| khối đề xuất (3 kỹ thuật LLM) | **671 732** | 94,42 |
| khối ViT tiêu chuẩn (`vanilla`) | 713 588 | 94,91 |

Chênh **0,49 pp** (≈ 2 ảnh) trong khi đề xuất **nhẹ hơn 41 856 tham số (5,9%)**.

Riêng trục **nhiễu cảm biến** có một quan sát đáng chú ý:

| | noise |
|---|---|
| bỏ hẳn Transformer | 88,20 |
| Transformer ViT tiêu chuẩn | 87,94 |
| **Transformer + 3 kỹ thuật LLM** | **92,23** |

Khối ViT tiêu chuẩn không cải thiện gì so với **không có** Transformer; toàn bộ ~4 pp độ bền
nhiễu đến từ ba kỹ thuật LLM.

> **Giới hạn cần nêu rõ:** kết quả 1 seed. Bảng có 15 phép kiểm định → ngưỡng Bonferroni
> α = 0,05/15 ≈ 0,003. Ở ngưỡng đó phần lớn khác biệt trên **chưa** đạt ý nghĩa thống kê, nên
> chúng tôi báo cáo là **quan sát**, không phải kết luận.

---

## 3. RTR_Lite_MobileNet chỉ 1 run — chưa làm ❌

Chưa chạy được 3 seed cho mô hình nền tảng.

**Đề xuất phương án thay thế, rẻ hơn và chặt hơn:** điều Thầy lo là *"cải thiện có nằm trong
biên độ nhiễu không"*. Câu đó trả lời được bằng **McNemar ghép cặp từng ảnh** trên **cùng 373
ảnh test** — không cần ước lượng phương sai giữa các lần chạy, vì hai mô hình được chấm trên
đúng cùng một tập. Chỉ cần chấm RTR_Lite_MobileNet trên cùng bộ test rồi ghép cặp.

Nếu Thầy vẫn muốn 3 seed, chúng tôi sẽ bổ sung.

---

## 4. Ablation huấn luyện lại — đã chạy ✅

Bỏ khối khỏi kiến trúc rồi **huấn luyện lại từ đầu** (không phải knock-out trên mạng đã train),
cùng seed / recipe / epoch. Baseline chính là mô hình đề xuất trong cùng notebook.

| bỏ khối | tham số | clean | blur | dark | noise | jpeg | TB |
|---|---|---|---|---|---|---|---|
| — (đầy đủ) | 671 732 | 98,66 | 91,69 | 97,05 | **92,23** | 92,49 | 94,42 |
| **Tiny Transformer** | 556 404 | 97,86 | 89,81 | 98,66 | **88,20** | 89,81 | 92,87 |
| Triplet Attention | 671 432 | 98,93 | 94,64 | 99,20 | 91,69 | 93,30 | 95,55 |
| RES_Block | 671 473 | 98,39 | 94,64 | 98,66 | 93,57 | 94,37 | 95,92 |

**McNemar so với mô hình đầy đủ:**

| bỏ khối | trục có ý nghĩa (p < 0,05) |
|---|---|
| **Tiny Transformer** | **noise +4,02 pp (p = 0,008)** — khối duy nhất có đóng góp dương |
| Triplet Attention | blur −2,95 (p = 0,013), dark −2,15 (p = 0,008) — bỏ đi thì **tốt hơn** |
| RES_Block | blur −2,95 (p = 0,027), dark −1,61 (p = 0,031) — bỏ đi thì **tốt hơn** |

Đóng góp của Tiny Transformer trên trục nhiễu **tái lập ở một lần chạy độc lập khác**
(+8,85 pp, p = 2,5e-7), nên chúng tôi xem đây là kết quả vững.

Hai khối attention CNN (Triplet Attention, RES_Block) — vốn **kế thừa từ RTR_Lite_MobileNet
gốc**, không phải đóng góp của bài này — cho kết quả **âm** và sẽ được trình bày đúng như vậy.

---

## 5. Domain gap — một phần ⚠️

Đã bổ sung **benchmark 4 điều kiện suy giảm**, áp ở test-time theo tinh thần ImageNet-C
(không phép nào có mặt lúc huấn luyện), mô phỏng điều kiện chụp thực tế:

| điều kiện | mô phỏng | S2 | Wilson 95% CI |
|---|---|---|---|
| clean | — | 98,66 | [96,90 – 99,43] |
| Gaussian blur k=5 | rung tay, mất nét | 91,69 | [88,44 – 94,08] |
| giảm sáng ×0,4 | chụp thiếu sáng, dưới tán lá | 97,05 | [94,80 – 98,35] |
| ISO noise 0,1 | nhiễu cảm biến điện thoại | 92,23 | [89,06 – 94,53] |
| JPEG q=30 | nén ảnh điện thoại | 92,49 | [89,36 – 94,76] |

Vai trò của augmentation cũng đo được: kịch bản **không** augment đạt clean 99,20% nhưng
dark chỉ **80,16%**; kịch bản có augment (ColorJitter) đạt dark **97,05%** — **+16,9 pp**.

**Chưa xử lý được:** nền lá/đất phức tạp và góc chụp không cố định — hai yếu tố cần ảnh thực
địa. Bộ nhiễu chỉ mô phỏng suy giảm **quá trình thu nhận**, không mô phỏng **chuyển miền nền
cảnh**: một ảnh nền trắng bị làm mờ thì vẫn là ảnh nền trắng.

Chúng tôi ghi rõ ranh giới này trong phần Giới hạn thay vì tuyên bố quá phạm vi đo được.

### Đoạn Giới hạn đề xuất (bản VN)

> **Giới hạn — khoảng cách miền.** ChiliEdge-DS được chụp trên nền trắng có kiểm soát, khác
> với điều kiện chụp thực tế ngoài đồng. Để đánh giá một phần khoảng cách này, chúng tôi bổ
> sung benchmark bốn điều kiện suy giảm áp ở test-time (không xuất hiện lúc huấn luyện), mô
> phỏng các yếu tố thường gặp khi chụp bằng điện thoại: nhoè do rung tay (Gaussian blur k=5),
> thiếu sáng dưới tán lá (giảm sáng ×0,4), nhiễu cảm biến (ISO noise 0,1) và nén ảnh (JPEG
> q=30). Mô hình giữ 91,7–97,1% trên bốn điều kiện này so với 98,7% trên ảnh sạch.
>
> Tuy nhiên, bốn phép trên chỉ mô phỏng **suy giảm quá trình thu nhận**, **không** mô phỏng
> **chuyển miền nền cảnh**. Cụ thể, hai yếu tố quan trọng của ảnh thực địa **chưa** được đánh
> giá: (i) nền lá và đất phức tạp thay cho nền trắng đồng nhất, và (ii) góc chụp thay đổi.
> Do đó **không nên** suy ra độ chính xác 98,7% sẽ giữ nguyên khi triển khai ngoài đồng. Việc
> thu thập một tập ảnh thực địa có nền tự nhiên — dù quy mô nhỏ — để đo mức suy giảm khi
> chuyển miền là hướng nghiên cứu tiếp theo cần thiết trước khi triển khai thực tế.

### Đoạn Giới hạn đề xuất (bản EN)

> **Limitation — domain gap.** ChiliEdge-DS was captured against a controlled white background,
> which differs from real in-field capture. To partially probe this gap, we add a test-time
> benchmark of four acquisition degradations never seen during training, each modelling a
> common phone-capture factor: motion/defocus blur (Gaussian, k=5), low light under foliage
> (brightness ×0.4), sensor noise (ISO noise 0.1), and lossy compression (JPEG q=30). The model
> retains 91.7–97.1% across these conditions versus 98.7% on clean images.
>
> These four transforms, however, simulate **acquisition degradation only** and do **not**
> simulate **scene-level domain shift**. Two important properties of real field imagery remain
> **unevaluated**: (i) cluttered leaf-and-soil backgrounds instead of a uniform white backdrop,
> and (ii) unconstrained viewpoints. Our 98.7% clean accuracy should therefore **not** be read
> as an estimate of in-field deployment accuracy. Collecting even a small natural-background
> test set to quantify the transfer drop is a necessary next step before field deployment.

---

## 6. Tập test nhỏ, không CI/kiểm định — đã bổ sung ✅

- **Wilson score interval 95%** (đúng phương pháp Thầy gợi ý, n = 373) cho mọi con số
  accuracy — xem bảng ở mục 5.
- **McNemar ghép cặp từng ảnh** cho mọi phép so sánh giữa các mô hình, thay vì so accuracy trần.

Kết quả này **củng cố cảnh báo của Thầy**: CI của clean là [96,90 – 99,43], nên mọi mô hình
trong dải 98–99% đều **chồng CI lên nhau** — không thể xếp hạng bằng clean. Vì vậy bài đã
chuyển trọng tâm sang trục **độ bền nhiễu**, nơi các khoảng tin cậy **tách rời**:
92,23 [89,06 – 94,53] so với 88,20 [84,4 – 91,2] khi bỏ Transformer.

Chưa có k-fold; tập chia cố định vẫn được giữ để so sánh được với 27 cấu hình baseline.

---

## 7. Related Work sơ lược — sửa bản thảo 🖊

Sẽ mở rộng: mô tả ý tưởng chính của các nghiên cứu tiêu biểu trong nhóm baseline, và nêu rõ
tiêu chí lựa chọn 15 nghiên cứu (phạm vi tìm kiếm, tiêu chí đưa vào/loại trừ, khoảng thời gian).

---

## 8. Phân bố lớp + cấu hình CPU — đã bổ sung ✅

**Phân bố lớp** (`c14_class_distribution.csv`):

| lớp | train | val | test | tổng |
|---|---|---|---|---|
| anthracnose | 264 | 81 | 62 | 407 |
| healthy | 334 | 102 | 78 | 514 |
| leaf_curl | 328 | 104 | 82 | 514 |
| leaf_spot | 326 | 101 | 76 | 503 |
| yellow_leaf | 345 | 103 | 75 | 523 |
| **Tổng** | **1 597** | **491** | **373** | **2 461** |

Tỉ lệ lớp lớn nhất / nhỏ nhất = 523/407 = **1,29 < 1,5** → dữ liệu cân bằng, không dùng
class weighting, Macro-F1 và Macro-Recall không bị lớp trội làm lệch.

**Phần cứng suy luận** (`c14_latency_cpu_vs_gpu.csv`):

Môi trường đo (Kaggle, cùng một phiên chạy cho cả ba cấu hình dưới):

- **CPU: Intel® Xeon® @ 2,00 GHz, 4 lõi logic**
- **GPU: NVIDIA Tesla T4** (đơn GPU)
- PyTorch 2.10.0+cu128, Python 3.12.13

| cấu hình | ms/ảnh | ảnh/giây |
|---|---|---|
| **CPU, batch = 1** | **26,28** | **38,1** |
| 1 GPU, batch = 1 | 6,19 | 161,5 |
| 1 GPU, batch = 32 | 1,21 | 826,8 |

Cấu hình CPU, batch = 1 là con số phản ánh kịch bản triển khai trên thiết bị biên và được
dùng trong Tóm tắt/Kết luận. Số cũ **22,9 ms** trong bản thảo là của mô hình cũ (835 188 tham
số, lưới 7×7) — đã thay bằng **26,28 ms** đo lại trên mô hình đề xuất mới.

---

## 9. Hình thức — sửa bản thảo 🖊

- Đánh số lại toàn bộ Bảng/Hình theo thứ tự.
- Ghi rõ **"92× nhanh hơn"** là **so sánh độ trễ mức hệ thống** (suy luận cục bộ vs round-trip
  API đám mây, gồm độ trễ mạng và hàng đợi máy chủ), không phải so tốc độ suy luận thuần giữa
  hai kiến trúc.
- Viết lại câu "Trained end-to-end with no actual language model" theo đúng gợi ý của Thầy.
- Ghi rõ phiên bản API + ngày truy cập của mô hình Vision-LLM đối chứng.
- Các hình xuất ra đã **bỏ tiêu đề nung trong ảnh**; chú thích đặt ở bản thảo.

---

## Ghi chú trung thực về phạm vi

Bài đã **đổi hướng** theo chính các góp ý trên. Hai điều chỉnh lớn:

1. **Điểm cắt backbone 15 → 14.** `features[14]` là tầng stride-32, làm lưới đặc trưng sập từ
   14×14 (196 token) xuống 7×7 (49 token). Cắt tại 14 giữ được 196 token và cho mô hình
   **nhẹ hơn 20%** (671 732 vs 835 188) mà bền hơn trên mọi điều kiện nhiễu.

2. **Đóng góp được phát biểu lại cho khớp số đo.** Tiny Transformer đóng góp **độ bền trước
   nhiễu cảm biến** (+4 pp, tái lập 2 lần chạy), **không** đóng góp accuracy trên ảnh sạch —
   trục clean đã bão hoà và mọi mô hình đều chồng khoảng tin cậy. Ba kỹ thuật LLM cho kết quả
   **ngang khối ViT tiêu chuẩn** với **5,9% ít tham số hơn**; chúng tôi trình bày đúng như vậy,
   không tuyên bố chúng làm tăng độ chính xác.

Tất cả số liệu ở 1 seed. Chúng tôi sẽ nêu rõ giới hạn này, kèm ngưỡng Bonferroni cho các bảng
nhiều phép kiểm định.
