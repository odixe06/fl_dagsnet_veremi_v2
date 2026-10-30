# Continue — bàn giao sống, 30/10/2026 ~11:15 (giờ VN)

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) →
[bài học vận hành](lessons.md) → trang này. Sửa tại chỗ. Lịch sử 29–30/10: [snapshot](../archive/continue-2026-10-30.md).

## Đang chạy (Kaggle; W&B `21522798-uit/fl-dagsnet-veremi-v2`)

| kernel | account | SHA | việc | ghi chú |
|---|---|---|---|---|
| `fl2-nilm-r2` | accanniez | 903ac42 | NILM-FL 50 round | ~1.017 s/round ⇒ budget 11,5 h dừng ~r38 (state theo rank) |
| `fl2-tinyproto-r3` | anhtruong1909 | 55d13cc | TinyProto chạy lại (view prototype theo Eq. 12) | push 30/10 ~10:50, ~4,5 h |

Kiểm: `python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <acct> -- kaggle kernels status <acct>/<slug>`
(API Kaggle chập chờn: status rỗng ⇒ thử lại). W&B: `wandb.Api().runs("21522798-uit/fl-dagsnet-veremi-v2")`.

## Đã xong (30/10) — kết quả: [results](../research/results.md), chi phí: [cost](../research/cost.md)

- 14 run FL 50 round + centralized 50 epoch + CANPerFL đã pull về `runs/pulls/<slug>/` (git-ignored, bỏ `state*.pt`) và audit
  PASS (đủ round, CMs, không NaN, HP đúng protocol). Bảng 10 metric tính lại từ CMs (`scripts/report.py`).
- Lỗi đã xử lý: APFed ERROR sau r50 khi ghi state (RAM host; metric đủ; sửa `bda3c07`, không chạy lại); TinyProto r2 dùng sai
  prototype suy luận (sửa `55d13cc`, chạy r3); SCAFFOLD không ổn định r1–r17 (NaN r15–16) rồi hồi phục — ghi chú trong results.
- Kernel cũ không lưu model mỗi round đều đã dừng (`dinhan25/fl2-v2x-htfl` cũ tự hết; bỏ qua).

## Bước tiếp

1. `fl2-nilm-r2` dừng budget (~r38) ⇒ session 2, cùng `--run`:
   `python scripts/gen_notebook.py --owner accanniez --slug fl2-nilm-r2s2 --method nilm --run fl2-nilm-r2 --resume-kernel accanniez/fl2-nilm-r2 --max-hours 11.5`
   rồi `embed_wandb_key.py` + `chmod 600` + `ast.parse` mọi cell + push `-t 43200`.
2. `fl2-tinyproto-r3` xong ⇒ pull (`pull_kernel_output.py ... --pattern '^(?!.*state(_rank\d+)?\.pt$)'`), audit, thêm vào results
   (headline `client_proto`); kiểm view prototype ≈ view head (smoke 100 xe r2: 0,819 vs 0,825).
3. Viết `docs/methods/` cho APFed, V2X-HtFL, CANPerFL; sửa bảng "còn thiếu" trong [papers/README](../../papers/README.md).
4. MobiHet tiếp ở repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (bàn giao riêng `docs/operations/continue.md` của repo đó);
   arm cuối 50 round so với bảng results ở đây.

## Lưu ý vận hành

- `--set` qua `json.loads` ⇒ chuỗi cần dấu `"` JSON (xem [lessons](lessons.md) 30/10). Notebook push phải nhúng key W&B
  (`embed_wandb_key.py`; quên ⇒ `SystemExit: W&B credential not embedded`, bản 1 fl2-canperfl).
- Commit không có trailer `Co-Authored-By` (owner 30/10).

## Việc tay (owner)

- Không có việc gấp. minhtrit06 hết GPU quota tới 31/10. dinhan25 chỉ có token (chạy được qua `kaggle_as.py`).
