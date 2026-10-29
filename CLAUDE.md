# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Rules: 

**If anything needs to be done manually, please let me know so I can do it.**

**During the process, if you are unsure about anything, ask me questions to clarify and confirm the requirements before proceeding**

**Communication Style: Caveman Mode**
- Respond using ultra-concise caveman speak to minimize token usage while retaining 100% technical accuracy.
- Drop all articles (a, an, the), filler words, pleasantries, and hedging.
- Use short fragments and telegraphic phrases.
- Keep code blocks, CLI commands, technical terms, and file paths completely intact and exact.

## 6. Bắt đầu mỗi phiên

**Đọc [CONTEXT.md](CONTEXT.md) (trạng thái, quyết định) rồi [LESSONS.md](LESSONS.md) (bài học kỹ thuật),
[docs/operations/lessons.md](docs/operations/lessons.md) (bài học vận hành) và [docs/operations/continue.md](docs/operations/continue.md) (bàn giao) trước khi làm bất cứ việc gì.**
Cấu hình chung mọi phương pháp: [docs/research/protocol.md](docs/research/protocol.md) — không đổi riêng cho một phương pháp.
Dữ kiện bất biến (máy, dataset, DAGSNet): [knowledge/](knowledge/README.md).
Khi rút ra bài học mới hoặc một bài học bị bác bỏ, sửa ngay trong phiên đó: phương pháp/dữ liệu/train/đánh giá/code → LESSONS.md;
Kaggle/notebook/pipeline, máy local, Git → docs/operations/lessons.md.

## 7. Skill bắt buộc theo loại việc (`.claude/skills/`)

Trước khi làm, gọi skill khớp với việc; nhiều việc khớp nhiều skill thì gọi đủ.

| việc | skill |
|---|---|
| bắt đầu phiên / sau tóm tắt, collect + audit kết quả, khóa cổng, cập nhật results/bàn giao | `federated-experiment-workflow` |
| sinh notebook, upload dataset nguồn/checkpoint, quota, push kernel, nối tiếp session, pull output | `kaggle-training-notebook` |
| theo dõi run đang chạy, health, tiến độ W&B | `wandb-training-monitor` |
| tối ưu tốc độ code train (workers, compile, host sync, benchmark), test tương đương | `federated-training-speed` |
| sắp xếp CONTEXT/docs/knowledge, quyết định đặt thông tin ở đâu | `context-hub` |
| nắm toàn bộ context đầu phiên / sau compact, ghi bàn giao sống, snapshot mốc, dời tài liệu + kiểm link | `project-context` |
| backup Git, restore, dọn ổ (chỉ khi owner yêu cầu) | `repo-backup` |
