# Continue — bàn giao sống, 30/10/2026 ~00:05

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy (Kaggle; W&B project `21522798-uit/fl-dagsnet-veremi-v2`)

| kernel | account | việc | trạng thái 30/10 00:05 |
|---|---|---|---|
| `fl2-cgate-w1` | odixeuit | centralized W = 1, 20 epoch, fold train → val | ep11 val 0,962 |
| `fl2-cgate-w075` | odixeuit (session 2) | W = 0,75 (push 29/10 23:45, source `50b7fd2`) | đang nạp |
| `fl2-cgate-w05` | accanniez | W = 0,5 | ep12 0,946 |
| `fl2-cgate-w0375` | khanhmay0304 | W = 0,375 | ep17 0,942 |
| `fl2-cgate-w025` | minhtran0601 | W = 0,25 | **xong**: ep20 0,9296 |
| `fl2-probe-fedavg` v3 | odixe0502 | FedAvg W = 1, lịch 50 round, budget 1,5 h (run `fl2-probe-fedavg-v3`) | r6 test 0,865; 223 s train + 16 s eval / round |

Kiểm: `python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <acct> -- kaggle kernels status <acct>/<slug>`; log live
`... kernels logs <acct>/<slug> --follow` (dòng `^ep..`/`^r..`); W&B API (xem lịch sử phiên 29/10).

## Đã xong (29/10)

- 16 phương pháp code + test toy CPU (22 test) + smoke GPU compile (100 xe × 2 round): fedavg fedprox scaffold afpha fd_ids lwfednids
  pfedes fedproto fedtgp tinyproto nilm perfedskd mobihet. Chưa có: **APFed, V2X-HtFL, CANPerFL (chờ PDF)**.
- Quyết định owner 29/10 (đã áp): gộp model theo n_k cho mọi phương pháp; MobiHet n_k; AFPHA 8 cụm scenario×run + μ thích nghi;
  TinyProto-FP; headline prototype = luật của bài.
- Tốc độ T4 đo thật: FedAvg W = 1 toàn 5.343 xe 223 s/round train (2×T4). Notebook dựng npy một lần (400 s) rồi memmap.

## Bước tiếp

1. Gate xong ⇒ W nhỏ nhất có F1 val ep20 ≥ F1(W=1) − 0,01 ⇒ ghi CONTEXT + protocol, khóa.
2. Sinh notebook 50 round cho mọi phương pháp với W đã khóa (`scripts/gen_notebook.py --method <m> --width W --max-hours 11.5`),
   push theo quota (`kaggle_account.py quota`). Phương pháp 2 model (mobihet, nilm, pfedes) ~3× FedAvg ⇒ có thể cần session 2
   (`--resume-kernel <acct>/<slug>`; cùng `--run`).
3. Centralized báo cáo (`run_central.py` không `--gate`, 50 epoch) cho W = 1 và W đã khóa.
4. Khi có PDF: APFed, V2X-HtFL, CANPerFL (đọc bài → docstring `methods/<alias>.py` → test toy → smoke GPU).
5. Pull output (`pull_kernel_output.py`), `scripts/report.py runs/pulls/*` → `docs/research/results.md`.

## Việc tay (owner)

- Tải 3 PDF còn thiếu vào `papers/` (tên file trong [papers/README](../../papers/README.md)): APFed, CANPerFL, V2X-HtFL.
- Đăng nhập Kaggle CLI cho 10 account còn thiếu (`kaggle_account.py list`: anhtruong1909, annhannh331, annhtnt, catbaochau, dinhan25,
  minhtriethihi, nguyenminhtran060113, takids1909, thuyanh316, trietbackup):
  `python .claude/skills/kaggle-training-notebook/scripts/kaggle_account.py add-account <user>`.
