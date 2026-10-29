# Continue — bàn giao sống, 29/10/2026

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy

- Local: smoke FedAvg 400 client × 3 round (`runs/smoke_fedavg`), kiểm học được + đo thời gian round.

## Bước tiếp

1. Smoke FedAvg local đạt (loss giảm, F1 tăng, resume chạy) → test đơn vị `core/` (optim ≡ torch AdamW, eval, metrics).
2. Kaggle notebook chung (clone repo public ở commit pin, dataset v4, 2×T4 torchrun, W&B) + probe 2 round đo T4.
3. Cổng độ rộng centralized trên Kaggle (W ∈ {1; 0,5; 0,375; 0,25}, fold train → val) ⇒ khóa W.
4. Build từng phương pháp theo [danh mục](../research/methods.md): đọc bài → `docs/methods/<alias>.md` → `methods/<alias>.py` → smoke local
   → run 50 round Kaggle. Ưu tiên: FedAvg, FedProx, SCAFFOLD (khung toàn cục), rồi personalized, rồi MobiHet.
5. Centralized báo cáo (full + W) trên train+val → test.

## Việc tay (owner)

- Tải 3 PDF còn thiếu vào `papers/` (tên file trong [papers/README](../../papers/README.md)): APFed, CANPerFL, V2X-HtFL.
- Đăng nhập Kaggle CLI cho 10 account còn thiếu (`kaggle_account.py list`: anhtruong1909, annhannh331, annhtnt, catbaochau, dinhan25,
  minhtriethihi, nguyenminhtran060113, takids1909, thuyanh316, trietbackup):
  `python .claude/skills/kaggle-training-notebook/scripts/kaggle_account.py add-account <user>`.
