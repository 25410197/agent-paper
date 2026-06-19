# MULTI-AGENT WORKSPACE: PLANT DISEASE DETECTION
You are an AI system managing multiple specialized agents in this workspace.

## Subagents

Spawn subagents to isolate context, parallelize independent work, or offload bulk mechanical tasks. Don't spawn when the parent needs the reasoning, when synthesis requires holding things together, or when spawn overhead dominates.

Pick the cheapest model that can do the subtask well:
- Haiku: bulk mechanical work, no judgment
- Sonnet: scoped research, code exploration, in-scope synthesis
- Opus: subtasks needing real planning or tradeoffs

If a subagent realizes it needs a higher tier than itself, return to the parent.

Parent owns final output and cross-spawn synthesis. User instructions override.

## Preferred Tools

### Data Fetching

1. **WebFetch**: free, text-only, works on public pages that don't block bots.
2. **agent-browser CLI**: free, local Rust CLI + Chrome via CDP. For dynamic pages or auth walls that WebFetch can't handle. Returns the accessibility tree with element refs (@e1, @e2). ~82% fewer tokens than screenshot-based tools. Install: `npm i -g agent-browser && agent-browser install`. Use `snapshot` for AI-friendly DOM state, element refs for interaction.
3. **Notice recurring fetch patterns and propose wrapping them as dedicated tools.** When the same fetch/parse logic comes up more than once, suggest wrapping it as a named tool (e.g. a skill file or a .py script that calls `agent-browser` with the snapshot and extraction steps baked in for that source). Add the entry to `## Dedicated Tools` below and reference it by name on future calls.

### PDF Files

1. **Text:** Use `pdftotext -layout`, not the `Read` tool, for body text — Read loads PDFs as images and burns tokens unnecessarily for plain prose.
2. **Figures/diagrams that carry reproduction-critical info** (architecture diagrams, layer/parameter tables rendered as images, etc.): pdftotext cannot see these — it only sees the caption text (e.g. "Fig. 9 shows the architecture..."). When the paper text references a figure/table for details needed to reproduce a model (filters, kernel size, stride, block structure...) and those details are NOT also present as text, you MUST render that PDF page to PNG and visually read it with `Read` — do not guess/assume the values and do not wait for the user to ask. This is standard pipeline behavior now, not an opt-in.
3. **Tables with architecture/parameter info — always render as PNG, no exception.** pdftotext frequently garbles multi-column tables (columns merge, rows misalign, numbers shift). Do NOT trust pdftotext output for any table that mentions: "layer", "filter", "kernel", "stride", "output shape", "channel", "architecture", "proposed", "model", "params", "block". Even if pdftotext extracted *some* text from the table, treat it as potentially corrupted and render the page to PNG to visually verify the actual values. This is the most common source of missed architecture details.
   - Render with `python .claude/scripts/render_pdf_page.py <pdf_path> <out_dir> <page_num> [page_num...] [--zoom N]` (PyMuPDF-based, no poppler binary needed). Default zoom 3.0 (~288 DPI); use a tighter `clip` + higher zoom via a one-off script call if the figure's text is still illegible at default zoom.
   - To find the right page number: pdftotext preserves page breaks as form-feed characters (`\f`) in the extracted text by default — count `\f` occurrences before the figure reference to get the 1-indexed page number, then render a small page range around it to confirm visually.
   - Save rendered figures under `papers/<tên-bài-báo>/inputs/figures/`.
   - After reading the figure, write what you saw back into `paper_content.md` (a `## EXTRACTED FIGURES` section: figure number, page, file path, and a literal transcription of the values/labels visible) so the figure doesn't need to be re-rendered on subsequent reads, and so `paper-brainstorm` (which doesn't render PDFs itself) can use it directly.

## Dedicated Tools

**`render_pdf_page.py`** — `.claude/scripts/render_pdf_page.py`. Renders PDF page(s) to high-res PNG via PyMuPDF (no poppler dependency) so figures/diagrams that `pdftotext` can't capture can be visually read with `Read`. Used by `pdf-extractor` whenever a paper references a figure for architecture/parameter details not present as text. See `### PDF Files` above for the full procedure.

**THƯ MỤC CHUẨN (Standard Paths):**
- Thư mục gốc các bài báo: `papers/`
- Input (PDF bài báo): `papers/<tên-bài-báo>/inputs/`
- Output (notebook sinh ra): `papers/<tên-bài-báo>/outputs/`
- Agent definitions: `.claude/agents/`

**CẤU TRÚC THƯ MỤC:**
```
papers/
└── <tên-bài-báo>/       ← tên folder = tên file PDF (không có .pdf)
    ├── inputs/           ← PDF + ảnh trích xuất
    └── outputs/          ← notebook + summary + report
```
Mỗi bài báo mới → tạo subfolder riêng trong `papers/` theo tên bài báo.

**AVAILABLE AGENTS** (Claude Code auto-dispatches dựa theo `description` trong frontmatter của mỗi file):

| Agent name (frontmatter) | File                    | Trigger keywords                                          |
|--------------------------|-------------------------|-----------------------------------------------------------|
| `kaggle-reproducer`      | `kaggle_reproducer.md`  | "Kaggle", "reproduce", "phục dựng", "SOTA", upload PDF    |
| `gencnn`                 | `gencnn.md`             | "CNN", "build CNN", "tạo CNN", "xây dựng CNN"             |
| `notebook-verifier`      | `notebook_verifier.md`  | "Verify", "kiểm tra notebook", "verify notebook"          |
| `aug-analyzer`           | `aug_analyzer.md`       | "phân tích augmentation", "augmentation hợp lý không", "tune augmentation", dataset mới cần aug |

**DEFAULT BEHAVIOR:**
- Nếu người dùng upload PDF mà không chỉ định agent → dùng `kaggle-reproducer`.
- Nếu người dùng mô tả kiến trúc CNN mà không có PDF → dùng `gencnn`.
- Nếu không rõ → hỏi người dùng trước khi tiếp tục.