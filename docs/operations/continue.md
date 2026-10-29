# Continue — bàn giao sống, 29/10/2026 ~22:30

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy (Kaggle, push 29/10 ~22:25, source `fca1bae`)

| kernel | account | việc | kỳ vọng |
|---|---|---|---|
| `fl2-cgate-w1` | odixeuit | centralized W = 1, 20 epoch, fold train → val | cổng độ rộng |
| `fl2-cgate-w05` | accanniez | W = 0,5 | |
| `fl2-cgate-w0375` | khanhmay0304 | W = 0,375 | |
| `fl2-cgate-w025` | minhtran0601 | W = 0,25 | |
| `fl2-probe-fedavg` | odixe0502 | FedAvg W = 1, lịch 50 round, dừng theo budget 0,8 h | thời gian round trên 2×T4, kiểm torchrun/NCCL/W&B |

W&B project `fl-dagsnet-veremi-v2`. Notebook sinh bằng `scripts/gen_notebook.py` (thư mục `notebooks/`, gitignore — có key W&B).
Kiểm: `python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <acct> -- kaggle kernels status <acct>/<slug>`;
log live: `... kaggle kernels logs <acct>/<slug> --follow`.

## Smoke local đã đạt (29/10)

FedAvg 400 client × 3 round, W = 1, RTX 3050: r1 0,378 → r2 0,835 → r3 0,844 macro-F1 toàn cục; 54 s/round (group_rows 2048);
budget stop + resume chạy (không trùng bit: RNG dropout không lưu). Centralized gate 400 client W 0,375: val 0,888 sau 2 epoch.

## Bước tiếp

1. Gate xong ⇒ W nhỏ nhất có F1 val ≥ F1(W=1) − 0,01 ⇒ ghi vào CONTEXT, khóa.
2. Probe xong ⇒ thời gian round T4 ⇒ ước GPU-giờ cho 16 run FL; chạy FedAvg/FedProx/SCAFFOLD 50 round với W đã khóa.
3. Build từng phương pháp còn lại theo [danh mục](../research/methods.md) (đọc bài → `docs/methods/<alias>.md` → code → test toy → smoke).
4. Centralized báo cáo (train + val → test, 50 epoch) cho W = 1 và W đã chọn.

## Việc tay (owner)

- Tải 3 PDF còn thiếu vào `papers/` (tên file trong [papers/README](../../papers/README.md)): APFed, CANPerFL, V2X-HtFL.
- Đăng nhập Kaggle CLI cho 10 account còn thiếu (`kaggle_account.py list`: anhtruong1909, annhannh331, annhtnt, catbaochau, dinhan25,
  minhtriethihi, nguyenminhtran060113, takids1909, thuyanh316, trietbackup):
  `python .claude/skills/kaggle-training-notebook/scripts/kaggle_account.py add-account <user>`.
