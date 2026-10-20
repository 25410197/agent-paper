# Chuẩn bị bảo vệ Đồ án tốt nghiệp — ChiliEdgeNet

Tài liệu gồm 2 phần: (1) kịch bản quay video demo (chi tiết theo từng thao tác + lời thoại), (2) câu
hỏi phản biện dự kiến + gợi ý trả lời. Số liệu dùng trong tài liệu này lấy từ báo cáo/nhật ký chỉnh
sửa hiện có trong project — trước khi quay/nói trước hội đồng, đối chiếu lại đúng số liệu mới nhất
trong file `.docx` của bạn (một vài chỗ đánh dấu **[CHECK]** vì không chắc 100% đã khớp bản mới nhất).

---

## PHẦN 1 — KỊCH BẢN QUAY VIDEO DEMO (~8 phút)

### Có cần đoạn giới thiệu chung (overview) trước khi vào demo không?

**Có, nên có** — khoảng 30–45 giây đầu, trước khi chạm vào webapp. Lý do: người xem (hội đồng) cần
biết đang xem cái gì và vì sao nó quan trọng trước khi thấy các thao tác click/kéo thả, nếu không
video sẽ giống một đoạn "mò mẫm phần mềm" chứ không phải trình bày kết quả nghiên cứu. Đoạn overview
KHÔNG lặp lại toàn bộ phần thuyết trình slide (cái đó đã có ở buổi bảo vệ) — chỉ cần đúng 3 câu: đề
tài là gì, mô hình đề xuất tên gì + kết quả chính (1 con số accuracy, 1 con số tham số), và video này
sẽ cho xem những gì. Coi như một "trailer" 30 giây, không phải một bài giới thiệu đầy đủ.

Cấu trúc timing tổng thể (có thể co giãn ±30s tuỳ tốc độ nói):

| # | Đoạn | Thời lượng |
|---|---|---|
| 0 | Overview / giới thiệu chung | 30–45s |
| 1 | Model card (thông số hệ thống) | 25–30s |
| 2 | Chẩn đoán cơ bản (upload → kết quả) | 70–80s |
| 3 | XAI: 3 phương pháp + đồng thuận | 90–110s |
| 4 | Robustness slider | 35–40s |
| 5 | Hỏi đáp với Gemini | 50–60s |
| 6 | Demo camera laptop thời gian thực | 70–80s |
| 7 | Kết | 10–15s |

**Tổng ~7–8 phút.** Nếu muốn co xuống 5 phút: cắt đoạn 4 (robustness) và rút đoạn 3 chỉ còn 2/3
phương pháp XAI thay vì cả 3 + consensus.

### Chuẩn bị trước khi quay (làm hết bước này trước, đừng vừa quay vừa chuẩn bị)

1. Chạy `docker-compose up` (hoặc chạy riêng `uvicorn server:app` + `npm run dev`), đợi log hiện
   `[API] model loaded on cpu | ckpt=...` rồi mới mở trình duyệt — tránh quay cảnh trang web bị lỗi
   load model.
2. Set sẵn API key Gemini vào `.env` nếu muốn demo đoạn 5, kiểm tra `/health` trả `"gemini": true`.
3. Chuẩn bị 4 ảnh mẫu trong `frontend/public/samples/`: 1 ảnh bệnh rõ ràng để demo đoạn 2–3 (ví dụ
   `anthracnose.jpg`), 1 ảnh mờ/không phải lá cây để demo cờ uncertain/OOD ở đoạn 2, 1 ảnh healthy để
   đối chiếu nếu cần, 1 ảnh dự phòng nếu ảnh đầu bị lỗi.
4. Phóng to trình duyệt lên 100–125% zoom (chữ nhỏ sẽ khó đọc khi xuất video 1080p), đóng hết tab/
   bookmark bar không liên quan, tắt notification Windows (Focus Assist bật lên).
5. Test thử toàn bộ chuỗi thao tác 1 lần **không quay** trước, để biết chỗ nào load lâu (XAI thường
   mất vài giây tính toán) — tránh khoảng lặng chết trong video thật.
6. Cắm sẵn/kiểm tra camera laptop hoạt động nếu quay đoạn 6, dọn bàn phía sau ống kính (tránh lộ
   người/vật dụng cá nhân vào khung hình khi test ROI).
7. Chuẩn bị 1 tờ giấy in hoặc ảnh trên điện thoại của 1 lá bệnh để đưa vào khung ROI ở đoạn 6.
8. Quay bằng OBS Studio hoặc Xbox Game Bar (Win+G), 1080p, quay tách từng đoạn theo bảng trên rồi
   ghép — **không cần quay liền một mạch**, quay hỏng đoạn nào thì quay lại đúng đoạn đó.

---

### Đoạn 0 — Overview / giới thiệu chung (30–45 giây)

**Mục tiêu:** cho người xem biết ngữ cảnh trước khi thấy thao tác cụ thể.

**Hình ảnh:** có thể để 1 slide tĩnh đơn giản (tên đề tài, tên SV/MSSV, tên mô hình + 2 con số kết
quả chính) trong 5–8 giây đầu, sau đó cắt sang màn hình desktop chuẩn bị mở trình duyệt.

**Lời thoại (đọc gần đúng, có thể chỉnh giọng cho tự nhiên):**
> "Chào thầy cô, đây là video demo cho đồ án 'Xây dựng web nhận diện và phân loại bệnh lý cho cây
> ớt'. Mô hình đề xuất là ChiliEdgeNet — khoảng 671 nghìn tham số **[CHECK số liệu]**, đạt 98,93%
> độ chính xác trên tập test **[CHECK]**, chạy hoàn toàn offline trên CPU. Trong video này em sẽ cho
> xem 2 phần: web app chẩn đoán kèm giải thích bằng XAI và LLM, và một chương trình demo camera thời
> gian thực trên laptop."

**Text overlay gợi ý khi edit:** tên đề tài + MSSV góc trên, không cần giữ suốt video, chỉ 5–8s đầu.

---

### Đoạn 1 — Model card (25–30 giây)

**Mục tiêu:** chứng minh hệ thống thật sự nhẹ và chạy được, không phải chỉ chạy trên máy mạnh.

**Thao tác:**
1. Mở webapp (đã load sẵn), trỏ chuột vào khu vực hiển thị thông tin model (endpoint `/info` — có
   thể là 1 panel/card trên giao diện, hoặc mở tab DevTools gọi thử endpoint nếu giao diện không
   hiển thị sẵn).
2. Rê chuột chậm qua từng con số: tên model, số tham số, kích thước file (MB), thiết bị (`cpu`),
   test accuracy, latency đo được lúc khởi động.

**Lời thoại:**
> "Đây là model card của hệ thống. Model chỉ nặng vài megabyte, chạy trên CPU, và test accuracy đo
> được là 98,93 phần trăm. Latency ở đây là con số inference-only đo lúc khởi động server."

**Lưu ý:** không dừng lại giải thích quá kỹ từng con số — đoạn này chỉ để "khoe" nhanh, phần diễn
giải sâu để dành cho phần bảo vệ trực tiếp bằng lời + slide.

---

### Đoạn 2 — Chẩn đoán cơ bản (70–80 giây)

**Mục tiêu:** cho thấy luồng chính hoạt động: upload ảnh → nhận kết quả → có cơ chế cảnh báo khi
không chắc chắn (không phải model đoán bừa lúc nào cũng ra 1 nhãn với confidence cao).

**Thao tác chi tiết:**
1. Bấm nút upload/chọn ảnh mẫu → chọn ảnh bệnh rõ ràng (ví dụ anthracnose.jpg).
2. Chờ kết quả hiện ra (nếu load hơi lâu, có thể cắt video ở hậu kỳ để bớt khoảng chờ).
3. Trỏ chuột lần lượt vào: nhãn dự đoán lớn nhất trên cùng, % confidence, bảng ranking đầy đủ 5 lớp
   (kéo xuống nếu cần).
4. Upload tiếp ảnh thứ 2 — ảnh mờ hoặc không phải lá cây (ví dụ ảnh chụp lệch, hoặc ảnh vật thể khác)
   — để cờ `uncertain`/`ood` bật lên, trỏ vào cảnh báo đó.

**Lời thoại:**
> "Khi upload một ảnh lá bệnh, hệ thống trả về nhãn dự đoán, phần trăm tin cậy, và bảng xếp hạng đầy
> đủ cho cả 5 lớp bệnh. Bên cạnh đó, hệ thống còn có cờ cảnh báo khi ảnh không rõ ràng hoặc không
> giống bất kỳ lớp nào đã học — ví dụ với ảnh này, model báo 'uncertain' thay vì cố đoán bừa ra một
> nhãn với độ tin cậy giả tạo cao."

**Lưu ý:** nếu ảnh mờ demo không kích hoạt được cờ uncertain/OOD ngay lần đầu, thử vài ảnh khác nhau
trước khi quay chính thức — đừng để tình trạng "chọn đại ảnh, cờ không lên, phải giải thích chống
chế trên video".

---

### Đoạn 3 — XAI: 3 phương pháp giải thích + đồng thuận (90–110 giây)

**Mục tiêu:** đây là phần kỹ thuật nổi bật nhất của webapp, nên đầu tư thời lượng nhiều nhất.

**Thao tác chi tiết (dùng lại đúng ảnh ở đoạn 2, bước 1, để nhất quán):**
1. Bật bản đồ **Attention Rollout** — trỏ vào vùng nhiệt độ cao trên ảnh, đối chiếu với vị trí vết
   bệnh thật trên lá.
2. Chuyển sang **Gradient SHAP** — cùng một ảnh, so sánh vùng được tô có giống vùng của Attention
   Rollout không.
3. Chuyển sang **Occlusion Sensitivity** — tương tự.
4. Bật bản đồ **đồng thuận (consensus)**, trỏ vào chỉ số % đồng thuận giữa 3 phương pháp.

**Lời thoại:**
> "Phần XAI dùng 3 phương pháp giải thích thuộc 3 nhóm cơ chế khác nhau. Attention Rollout dựa trên
> attention của khối Transformer trong model. Gradient SHAP và Occlusion Sensitivity dựa trên
> gradient và việc che từng vùng ảnh để đo mức ảnh hưởng. Có thể thấy cả 3 phương pháp đều tập trung
> vào đúng vùng có vết bệnh trên lá. Bản đồ đồng thuận này tổng hợp cả 3 lại và cho biết mức độ
> đồng thuận giữa các phương pháp — đồng thuận càng cao thì lời giải thích càng đáng tin."

**Lưu ý quan trọng:** chọn TRƯỚC một ảnh mà cả 3 phương pháp cho ra vùng tập trung tương đối giống
nhau (test trước khi quay, ở bước "test thử không quay" đã nêu ở phần chuẩn bị) — nếu vô tình chọn
ảnh mà 3 phương pháp lệch nhau nhiều, đồng thuận thấp, sẽ khó giải thích mượt trên video và có thể
bị hội đồng hỏi xoáy vào sau này.

---

### Đoạn 4 — Robustness slider (35–40 giây)

**Mục tiêu:** cho thấy hệ thống được kiểm tra dưới điều kiện chụp không lý tưởng, không chỉ test
trên ảnh đẹp.

**Thao tác:**
1. Kéo từ từ thanh trượt **blur** từ 0 lên mức trung bình, quan sát confidence giảm dần theo thời
   gian thực (đã có debounce nên không giật khi kéo nhanh — kéo chậm để video mượt).
2. Kéo thanh **dark** (tối đi) tương tự.
3. Kéo thanh **noise** tương tự, rồi kéo hết các thanh về 0 để kết thúc đoạn.

**Lời thoại:**
> "Người dùng có thể mô phỏng điều kiện chụp không lý tưởng ngay trên giao diện — ảnh mờ, thiếu
> sáng, hoặc nhiễu — để xem model còn giữ được độ tin cậy đến mức nào. Có thể thấy confidence giảm
> dần khi mức nhiễu tăng lên, đúng như kỳ vọng, chứ không giữ nguyên một cách bất thường."

---

### Đoạn 5 — Hỏi đáp với Gemini (50–60 giây)

**Mục tiêu:** cho thấy tính năng UX bổ sung, đồng thời làm rõ ranh giới để tránh hội đồng hiểu lầm
đây là phần đóng góp học thuật chính.

**Thao tác:**
1. Bấm nút sinh giải thích tự nhiên → chờ đoạn văn xuất hiện, đọc lướt qua trên màn hình (không cần
   đọc to hết cả đoạn).
2. Gõ 1 câu hỏi follow-up vào khung chat, ví dụ: "Bệnh này có lây sang cây khác không?" hoặc "Nên xử
   lý thế nào?" → chờ trả lời.

**Lời thoại:**
> "Ngoài kết quả kỹ thuật, hệ thống còn sinh một đoạn giải thích bằng ngôn ngữ tự nhiên thông qua
> Gemini API, và cho phép hỏi thêm theo dạng hội thoại. Cần lưu ý là phần LLM này chỉ nhận mô tả
> văn bản từ kết quả chẩn đoán và vùng XAI, không tự nhìn ảnh — đây là một tính năng UX tùy chọn,
> tách biệt với phần model chính, và hệ thống vẫn chạy bình thường nếu không có API key này."

---

### Đoạn 6 — Demo camera laptop thời gian thực (70–80 giây)

**Mục tiêu:** minh hoạ latency đo được trên phần cứng thật (laptop CPU), không phải môi trường máy
ảo Kaggle như lúc train/benchmark.

**Thao tác chi tiết:**
1. Chạy `python realtime_webcam_demo.py`, cửa sổ webcam mở lên, đưa ảnh lá bệnh (in giấy hoặc trên
   điện thoại) vào khung ROI ở giữa màn hình.
2. Bấm phím số (1–5) gán ground-truth đúng với lớp bệnh của ảnh đang test.
3. Bấm **SPACE** — chỉ vào dòng chữ flash "ĐÃ ĐO #..." và chỉ số latency vừa hiện lên.
4. Có thể lặp lại với 1 ảnh khác (ví dụ healthy) để cho thấy demo hoạt động ổn định qua nhiều lần đo,
   không phải ăn may 1 lần.

**Lời thoại:**
> "Đây là chương trình demo bổ sung, chạy trực tiếp bằng camera laptop, đo latency thực tế đầu-cuối
> — bao gồm cả bước tiền xử lý ảnh — khác với con số inference-only trong benchmark chính đo trên
> máy ảo Kaggle. Model chỉ tính khi người dùng chủ động bấm SPACE, tránh gây cảm giác kết quả 'nhảy
> số' liên tục do rung tay hoặc thay đổi ánh sáng giữa các khung hình."

**⚠️ Lưu ý quan trọng — KHÔNG được nói trong đoạn này:** không nhắc đến con số real-world accuracy
của phiên demo (56,45%) — số liệu này đã bị loại khỏi báo cáo chính thức theo yêu cầu GVHD, chỉ giữ
lại số liệu độ trễ. Chỉ nói về latency, không nói về độ chính xác của phiên demo camera. Nếu lỡ tay
gán sai ground-truth và model đoán sai ngay trên video, không sao — không phải cắt bỏ, chỉ đơn giản
là đừng chủ động tổng kết tỷ lệ đúng/sai trong lời thoại.

---

### Đoạn 7 — Kết (10–15 giây)

**Lời thoại:**
> "Trên đây là toàn bộ demo hệ thống. Em cảm ơn quý thầy cô đã theo dõi."

**Hình ảnh:** có thể fade về slide tên đề tài như đoạn mở đầu, hoặc để màn hình webapp đứng yên.

---

## PHẦN 2 — CÂU HỎI PHẢN BIỆN DỰ KIẾN + GỢI Ý TRẢ LỜI

Nguyên tắc trả lời trước hội đồng: ngắn gọn, thừa nhận thẳng giới hạn nếu có (hội đồng đánh giá cao
sự trung thực hơn là chống chế), không bịa số liệu không nhớ chính xác — có thể nói "con số cụ thể
em có trong báo cáo ở mục X, nhưng ý chính là...".

### Nhóm 1 — Kiến trúc mô hình

**Q1. Vì sao chọn kiến trúc lai CNN + Transformer thay vì thuần CNN hoặc thuần Vision Transformer?**
Gợi ý trả lời: CNN (MobileNetV2) mạnh về trích xuất đặc trưng cục bộ với chi phí tính toán thấp,
phù hợp thiết bị biên; Transformer thêm khả năng mô hình hoá quan hệ toàn cục giữa các vùng ảnh mà
CNN thuần khó làm được ở tầng nông. Dùng CNN làm backbone rồi thêm một khối Transformer rất nhỏ ở
cuối (embed_dim=128, 1 lớp, 4 head) giữ được ưu điểm nhẹ của CNN mà vẫn có thêm attention toàn cục,
thay vì dùng ViT thuần cần nhiều dữ liệu và tính toán hơn nhiều để đạt hiệu năng tương đương.

**Q2. "Tiny-LLM Transformer" nghĩa là gì — có dùng LLM (GPT/Gemini) trong lúc train model chính không?**
Gợi ý trả lời: Đây là tên tự đặt cho khối Transformer nhỏ trong kiến trúc (dùng RoPE-2D, Grouped-
Query Attention, RMSNorm — đều là các kỹ thuật vay mượn từ thiết kế LLM hiện đại để làm nhẹ khối
attention), **không phải** một mô hình ngôn ngữ lớn thật và không liên quan gì đến Gemini API dùng ở
phần webapp. Nên nói rõ ngay câu đầu để tránh hội đồng hiểu lầm là dùng LLM để phân loại ảnh.

**Q3. Vì sao cắt MobileNetV2 ở layer 14 (backbone_cut=14) mà không phải layer khác?**
Gợi ý trả lời: đây là kết quả từ ablation study so sánh nhiều điểm cắt khác nhau — cắt sớm giữ được
lưới không gian lớn hơn (nhiều token hơn cho Transformer) nhưng đặc trưng chưa đủ trừu tượng; cắt
muộn thì ngược lại. Layer 14 là điểm cân bằng tốt nhất giữa số tham số, độ chính xác và kích thước
lưới token trong thực nghiệm. **[CHECK: nếu báo cáo có bảng ablation so sánh các điểm cắt, nhớ số
liệu cụ thể trước khi vào bảo vệ.]**

### Nhóm 2 — Kết quả và ý nghĩa thống kê

**Q4. Cải thiện so với RTR_Lite_MobileNet chỉ khoảng 0,8 điểm phần trăm — có ý nghĩa gì không?**
Gợi ý trả lời: đúng là mỗi kiến trúc chỉ chạy một lần nên hai khoảng tin cậy (Wilson 95%) chồng lấn
nhau — không nên khẳng định đây là một xếp hạng thống kê chắc chắn, chỉ nên đọc như một xu hướng.
Ngược lại, khoản tiết kiệm khoảng 35% tham số **[CHECK con số]** là một phép tính số học chính xác,
không phụ thuộc nhiễu ngẫu nhiên — đây là đóng góp chắc chắn hơn của đồ án so với việc chỉ nhìn vào
con số accuracy tuyệt đối. Trả lời thẳng điểm này cho thấy hiểu đúng bản chất thống kê, hội đồng
thường đánh giá cao hơn là cố "chống chế" số liệu.

**Q5. Vì sao dùng kiểm định McNemar + hiệu chỉnh Holm thay vì t-test/ANOVA thông thường?**
Gợi ý trả lời: McNemar phù hợp khi so sánh hai bộ phân loại trên **cùng một tập test, ghép cặp theo
từng ảnh** (đúng/sai của model A và model B trên từng ảnh cụ thể) — đây là paired test cho biến nhị
phân (đúng/sai), khác với t-test thường dùng so sánh trung bình liên tục và giả định độc lập. Hiệu
chỉnh Holm cần thiết vì đồ án thực hiện **nhiều phép so sánh cặp** (nhiều model/nhiều scenario), nếu
không hiệu chỉnh sẽ tăng nguy cơ sai số loại I (false positive) do kiểm định nhiều lần.

**Q6. Ablation study đã retrain lại từ đầu hay chỉ tắt/bật thành phần trên model đã train?**
Gợi ý trả lời: retrain lại từ đầu cho từng biến thể (nêu rõ trong báo cáo) — đây là cách làm đúng
chuẩn, vì tắt thành phần trên một model đã hội tụ với thành phần đó không phản ánh đúng khả năng học
thực sự của biến thể thiếu thành phần.

### Nhóm 3 — Đánh giá thực tế / hạn chế

**Q7. Demo camera cho ra real-world accuracy khoảng 56% trong khi test-set accuracy 98,93% — chênh lệch lớn vậy có đáng lo không? (Câu hỏi rủi ro cao nhất — hội đồng có thể tự tính hoặc đã đọc log)**
Gợi ý trả lời: đây chính xác là lý do phần accuracy của phiên demo không đưa vào báo cáo chính thức
(theo góp ý của GVHD) — chỉ giữ lại số liệu độ trễ vì đó là con số đáng tin cậy và có ý nghĩa (đo
latency đầu-cuối thực tế trên CPU laptop). Về nguyên nhân chênh lệch accuracy: phiên đo đó bị lệch
lớp nặng (phần lớn mẫu là 1 lớp), ảnh chụp lại qua màn hình điện thoại làm mất chi tiết, và nền
ngoài lá lọt vào khung ROI — đều là vấn đề về **giao thức đo trong 1 phiên thử nghiệm nhỏ**, không
phải bằng chứng model kém trên dữ liệu thật đại diện. Nếu hội đồng hỏi sâu, có thể nói thẳng: đây là
proxy thực nghiệm cho "phần cứng biên", không phải benchmark chính thức, và đồ án đã nêu rõ giới hạn
này ở mục Hạn chế.

**Q8. Model đã được test trên ảnh nền tự nhiên phức tạp (ngoài đồng ruộng thật) chưa, hay chỉ trên ảnh đã crop nền?**
Gợi ý trả lời: trung thực về pipeline tiền xử lý — nếu dataset chính dùng ảnh đã crop nền (smartcrop,
như tên thư mục `dataset_smartcrop_plit_300` gợi ý), nên nói rõ đây là giới hạn đã ghi trong mục Hạn
chế/Hướng phát triển, và phần demo camera + few-shot domain adaptation trên tập độc lập là hai nỗ lực
để phần nào kiểm chứng khả năng tổng quát hoá ngoài điều kiện phòng thí nghiệm. **[CHECK: xác nhận
lại đúng quy trình tiền xử lý dataset chính trong báo cáo trước khi trả lời câu này.]**

**Q9. Few-shot domain adaptation trên tập dữ liệu độc lập cho kết quả thế nào?**
Gợi ý trả lời: chuẩn bị sẵn 2–3 con số chính (accuracy trước/sau adaptation, số shot dùng) — đây là
phần hội đồng khả năng cao sẽ hỏi vì nó trực tiếp trả lời câu "model có tổng quát hoá được không".
**[CHECK: lấy đúng số liệu từ chương Thực nghiệm/Kết quả trong docx trước khi vào bảo vệ — mình
không có số liệu chính xác của phần này.]**

### Nhóm 4 — Ứng dụng / triển khai

**Q10. Phần giải thích bằng Gemini có phụ thuộc Internet không — nếu mất mạng thì sao?**
Gợi ý trả lời: có — đây là tính năng tùy chọn (optional), gọi API bên ngoài nên cần Internet và API
key; toàn bộ phần chẩn đoán + XAI chính (phần lõi của đồ án) chạy hoàn toàn offline trên CPU, không
phụ thuộc gì vào Gemini. Nếu thiếu key hoặc mất mạng, hệ thống vẫn hoạt động bình thường, chỉ không
hiện đoạn giải thích tự nhiên.

**Q11. Vì sao dùng Gemini (LLM có sẵn) thay vì tự huấn luyện một mô hình sinh giải thích riêng?**
Gợi ý trả lời: mục tiêu chính của đồ án là mô hình chẩn đoán nhẹ chạy offline (ChiliEdgeNet), không
phải xây dựng một LLM mới; dùng Gemini như một lớp tiện ích UX tùy chọn để "dịch" kết quả kỹ thuật
(nhãn, confidence, vùng XAI) sang ngôn ngữ tự nhiên dễ hiểu cho nông dân — đúng phạm vi đã nêu ở mục
Phạm vi nghiên cứu, không lạm dụng LLM vào phần đóng góp học thuật chính.

**Q12. Hướng phát triển tiếp theo là gì?**
Gợi ý trả lời (khớp với mục Hạn chế & Hướng phát triển trong báo cáo): mở rộng test trên ảnh đồng
ruộng thật không qua crop nền, thử nghiệm trên thiết bị biên vật lý thật (Raspberry Pi/Jetson thay
vì proxy laptop CPU), tăng số lần chạy độc lập để có khoảng tin cậy chặt hơn cho so sánh accuracy,
và mở rộng tập lớp bệnh/loại cây trồng khác.

---

## Checklist trước khi bảo vệ

- Đối chiếu lại mọi con số **[CHECK]** ở trên với bản `.docx` mới nhất (đặc biệt: số tham số, %
  accuracy, % tiết kiệm tham số, kết quả few-shot domain adaptation, ablation backbone_cut).
- Thuộc lòng 1 câu trả lời ngắn cho Q4 và Q7 — đây là 2 câu nhiều khả năng bị hỏi nhất và dễ bị
  "đuối lý" nếu trả lời vòng vo.
- Quay video xong thì tự xem lại 1 lần với âm thanh, kiểm tra không lộ API key/thông tin nhạy cảm
  nào trên màn hình lúc quay (ví dụ terminal log, biến môi trường).
- Test trước (không quay) toàn bộ chuỗi thao tác ở Phần 1 ít nhất 1 lần để biết chỗ nào load lâu và
  chọn sẵn ảnh mẫu cho ra kết quả đẹp (đồng thuận XAI cao, cờ uncertain kích hoạt đúng lúc cần).
