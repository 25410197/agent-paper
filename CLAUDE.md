# MULTI-AGENT WORKSPACE: PLANT DISEASE DETECTION
You are an AI system managing multiple specialized agents in this workspace.

**THƯ MỤC CHUẨN (Standard Paths):**
- Input (PDF bài báo): `.claude/inputs/`
- Output (notebook sinh ra): `outputs/`
- Agent definitions: `.claude/agents/`

**DISPATCHING LOGIC (Cách thức hoạt động):**
1. Khi người dùng gửi yêu cầu, bước ĐẦU TIÊN là đọc danh sách AVAILABLE AGENTS bên dưới.
2. Xác định Agent phù hợp nhất dựa trên trigger của từng agent.
3. Đọc TOÀN BỘ file agent tương ứng trong `.claude/agents/` trước khi làm bất cứ điều gì.
4. Thực thi ĐÚNG VÀ CHỈ các rules, skills, workflow được khai báo trong file agent đó.
5. Nếu không có agent nào phù hợp, hỏi người dùng để làm rõ yêu cầu trước khi tiếp tục.

**AVAILABLE AGENTS:**
| Alias   | File agent           | Kích hoạt khi                                                                 |
|---------|----------------------|-------------------------------------------------------------------------------|
| `Kaggle`| `kaggle_reproducer`  | Người dùng gõ "Kaggle", upload PDF, hoặc yêu cầu reproduce/SOTA              |
| `CNN`   | `gencnn`             | Người dùng gõ "CNN", mô tả kiến trúc cụ thể, hoặc gọi "build CNN"           |
| *(thêm agent mới vào đây)* | | |

**ALIAS DISPATCH (ƯU TIÊN CAO NHẤT):**
- Nếu tin nhắn người dùng **bắt đầu bằng hoặc chứa alias** (không phân biệt hoa thường), dispatch **ngay lập tức** tới agent tương ứng — không cần phân tích thêm.
- Ví dụ: "Kaggle, phân tích bài báo này" → dispatch `kaggle_reproducer`.
- Ví dụ: "CNN 3 lớp, filter 32-64-128" → dispatch `gencnn`.

**DEFAULT BEHAVIOR:**
- Nếu người dùng upload PDF mà không chỉ định agent → dùng `kaggle_reproducer`.
- Nếu người dùng mô tả kiến trúc CNN mà không có PDF → dùng `gencnn`.