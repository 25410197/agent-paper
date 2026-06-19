---
name: verification-lessons
description: Kinh nghiệm từ quá trình verify + fix notebook — lỗi phổ biến khi reproduce, quy tắc ngôn ngữ báo cáo, và bug script fix
metadata:
  type: feedback
  source: notebook-verifier + manual fix session (2026-05-26)
  notebook: MetaEnsemble_MLPMixer_LSTM_SVM_Experiment.ipynb
---

# Kinh nghiệm Verification & Fix — MetaEnsemble (MLP-Mixer + LSTM + SVM)

## Lỗi hay gặp khi đọc hyperparameter từ paper

### 1. LSTM optimizer bị nhầm sang Adam
**Lỗi:** Notebook dùng `torch.optim.Adam` cho LSTM trong khi paper Table 6 ghi rõ **rmsprop**.
**Fix:** Đổi sang `torch.optim.RMSprop(lstm.parameters(), lr=LSTM_LR)`.
**Why:** Hai optimizer hội tụ rất khác nhau — nhầm optimizer làm LSTM không học được gì, SVM nhận probability gần random.
**How to apply:** Khi reproduce, đọc kỹ từng bảng hyperparameter riêng cho từng sub-model. Ensemble paper thường có nhiều bảng (Table 5 cho model A, Table 6 cho model B) với optimizer khác nhau.

### 2. LSTM epochs bị đặt = 1
**Lỗi:** `LSTM_EPOCHS = 1` — ReproduceSummary nhầm ghi "Table 6: epochs = 1", nhưng paper Table 6 ghi 50-100.
**Fix:** `LSTM_EPOCHS = 50` (lower bound của range).
**Why:** 1 epoch → LSTM cực kỳ underfitting → probability outputs vô nghĩa → SVM không học được pattern.
**How to apply:** Khi paper viết range (ví dụ "50-100 epochs"), dùng lower bound làm default. KHÔNG để = 1 trừ khi paper nói rõ.

### 3. LSTM batch size bị nhầm
**Lỗi:** `LSTM_BATCH = 128` nhưng paper Table 6 ghi **256**.
**Fix:** `LSTM_BATCH = 256`.
**Why:** Batch size ảnh hưởng gradient noise và effective learning rate — sai 2x có thể làm kết quả khác biệt.

### 4. Channel-mixing hidden dim dùng `embed_dim * 4` thay vì 64
**Lỗi:** `MixerLayer` dùng `channel_hidden = embed_dim * 4 = 512` nhưng paper Table 5 ghi "MLP Channel dimension = 64".
**Fix:** `channel_hidden = channel_hidden or 64`.
**Why:** Thay đổi số param đáng kể trong mỗi MixerLayer. Standard MLP-Mixer dùng 4× expansion nhưng paper này chọn 64 cố định.
**How to apply:** Không tự assume 4× expansion nếu paper có bảng hyperparameter rõ ràng.

### 5. Dataset distribution chart dpi=150 thay vì 300
**Lỗi:** `plt.savefig(..., dpi=150)` cho biểu đồ phân phối dataset.
**Fix:** Đổi sang `dpi=300` để đồng nhất với tất cả output ảnh khác.

---

## Lỗi logic trong script fix nhiều điểm cùng lúc

**Vấn đề:** Khi dùng Python script để sửa nhiều chỗ trong cùng một cell notebook (.ipynb JSON), nếu hai lần replace đều dùng `src = ''.join(cell['source'])` lấy từ đầu vòng lặp, lần replace thứ hai sẽ OVERWRITE kết quả của lần đầu vì cả hai đều bắt đầu từ `src` gốc.

**Fix:** Khi fix nhiều điểm trong cùng cell, phải chain replace liên tiếp trên cùng một biến `new_src`, hoặc xử lý từng cell trong script riêng biệt.

```python
# SAI — lần 2 overwrite lần 1:
if pattern1 in src:
    new_src = src.replace(p1, r1)
    cell['source'] = [new_src]
if pattern2 in src:           # src vẫn là bản gốc!
    new_src = src.replace(p2, r2)
    cell['source'] = [new_src]  # ghi đè fix lần 1

# ĐÚNG — chain replace:
new_src = src
if pattern1 in new_src:
    new_src = new_src.replace(p1, r1)
if pattern2 in new_src:
    new_src = new_src.replace(p2, r2)
if new_src != src:
    cell['source'] = [new_src]
```

---

## Pattern đọc paper tốt hơn

| Tình huống | Cần làm |
|-----------|---------|
| Ensemble model (nhiều sub-model) | Đọc từng bảng hyperparameter riêng cho MỖI sub-model |
| Epochs là range "X-Y" | Dùng lower bound X làm default |
| Framework không ghi rõ | Default PyTorch; nhưng xem Table nào có Conv2D output shape → có thể là Keras |
| "MLP Channel dimension = N" | Dùng đúng N, KHÔNG tự assume × 4 expansion |
| Normalization "pixel to [0,1]" | Nếu có pretrained CNN → dùng ImageNet stats và note lý do |

---

## Kết quả verification MetaEnsemble
- **Overall score:** ~69% (GO WITH CAUTION)
- **Constraint compliance:** 100% (24/24 rules)
- **Kiến trúc core:** Đúng (7 layers, D=128, N=9, GELU, skip, SVM grid)
- **Lỗi chính:** Cả 3 LSTM hyperparameter đều sai (optimizer, epochs, batch) — đã sửa

---

## Quy tắc ngôn ngữ Báo cáo

**Quy tắc:** Toàn bộ nội dung `[ModelName]_VerificationReport.md` phải viết bằng **TIẾNG VIỆT**.
- Giữ nguyên: tên kỹ thuật, tên biến code, tên hàm, trích dẫn trực tiếp từ paper
- Dịch sang tiếng Việt: tiêu đề, nhận xét, mô tả lỗi, ghi chú, kết luận

**Why:** Người dùng yêu cầu ngày 2026-05-26.
**How to apply:** Áp dụng cho mọi lần notebook-verifier tạo báo cáo. Đã cập nhật Rule 1 trong `.claude/agents/notebook_verifier.md`.
