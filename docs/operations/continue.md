# Continue — bàn giao sống, 30/10/2026 ~01:30

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy (Kaggle; W&B project `21522798-uit/fl-dagsnet-veremi-v2`)

| kernel | account | việc | trạng thái 30/10 00:05 |
|---|---|---|---|
| `fl2-cgate-w1` | odixeuit | centralized W = 1, 20 epoch, fold train → val | ep11 val 0,962 |
| `fl2-cgate-w075` | odixeuit (session 2) | W = 0,75 (push 29/10 23:45, source `50b7fd2`) | đang nạp |
| `fl2-cgate-w05` | accanniez | W = 0,5 | ep12 0,946 |
| `fl2-cgate-w0375` | khanhmay0304 | W = 0,375 | ep17 0,942 |
| `fl2-cgate-w025` | minhtran0601 | W = 0,25 | **xong**: ep20 0,9296 |
| `fl2-mobihet` | trietbackup | FL 50 round W = 1, source `6617261`, budget 11,5 h | push 30/10 ~00:30 |
| `fl2-fedavg` | trietbackup (session 2) | như trên | push |
| `fl2-fedprox` | catbaochau | như trên | push |
| `fl2-scaffold` | catbaochau (session 2) | như trên | push |
| `fl2-fedproto` | minhtriethihi | như trên | push |
| `fl2-afpha` | minhtriethihi (session 2) | như trên | push |
| `fl2-central-w1` | minhtran0601 | centralized báo cáo W = 1, 50 epoch, train+val → test | push 29/10 |
| `fl2-probe-fedavg` v3 | odixe0502 | FedAvg W = 1, lịch 50 round, budget 1,5 h (run `fl2-probe-fedavg-v3`) | r6 test 0,865; 223 s train + 16 s eval / round |

Kiểm: `python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <acct> -- kaggle kernels status <acct>/<slug>`; log live
`... kernels logs <acct>/<slug> --follow` (dòng `^ep..`/`^r..`); W&B API (xem lịch sử phiên 29/10).

## Dở dang phiên 30/10 (làm tiếp NGAY)

1. **Code mới chưa kiểm GPU** (commit WIP 30/10, chưa có run nào dùng):
   - `scripts/run_fl.py`: mỗi rank ghi state riêng (`state.pt` rank 0 + `state_rank<r>.pt`); chỉ ghi khi dừng vì budget
     (hoặc `--save-every N`); resume nạp xong thì xóa file đã copy vào (đĩa `/kaggle/working` ~20 GB, bank W = 1 = 4,2 GB/rank);
     quyết định dừng budget all_reduce MAX (tránh 2 rank lệch round → treo).
   - **Bug cũ (SHA `6617261`, đang chạy):** chỉ rank 0 ghi `state.pt` ⇒ phương pháp per-client (fedproto, mobihet) **resume sai**
     (rank 1 nạp bank của rank 0). Nếu `fl2-mobihet`/`fl2-fedproto` hết 11,5 h trước r50 ⇒ không resume, chạy lại bằng SHA mới.
     Global (fedavg, fedprox, scaffold, afpha) resume đúng.
   - `methods/apfed.py` mới (APFed = FedALA-like: θ ⊙ trên base, head toàn cục, θ⁰ = 1, α = 1, 1 lượt θ/round; θ ở host pinned,
     checkpoint fp16 của 1 − θ). Test toy pass (23/23). Smoke GPU local bị dừng giữa chừng — chạy lại:
     `python scripts/run_fl.py --method apfed --subset 60 --rounds 3 --width 0.25 --out <tmp> --set group_rows=2048 eval_rows=4096`
     (kiểm `theta_mean` < 1 từ r2, F1 hợp lý), rồi kiểm resume: `--max-hours` nhỏ để dừng sau r1, chạy lại, so r3.
   - Thiếu doc `docs/methods/` cho APFed (deviation đã ghi trong docstring).
2. Smoke xong ⇒ commit + push ⇒ sinh notebook bằng SHA mới, push **song song mọi account** (owner 30/10: đủ CLI trừ dinhan25):
   fd_ids, lwfednids, pfedes, fedtgp, tinyproto, nilm, perfedskd, apfed (+ V2X-HtFL, CANPerFL khi build xong). Kiểm quota trước
   (`kaggle_account.py quota`); mỗi account 2 session.
3. **V2X-HtFL** (owner 30/10: bỏ phần generator StyleGAN-XL): ETF cố định + projection d×d weight-norm + loss CurricularFace;
   server prototype toàn cục trainable (FedTGP-like) margin thích nghi Eq 16–17; L_s thay bằng kéo feature chiếu về prototype
   toàn cục, trọng số μλ_k (Eq 22, λ_k = |D|/(|D|+|D_k|)). Bài không nêu s, m, γ, τ, μ ⇒ chọn + ghi deviation.
4. **CANPerFL**: PDF thật đã có (`papers/canperfl.pdf`, owner 30/10) — chưa đọc.

## Đã xong (29/10)

- 16 phương pháp code + test toy CPU (22 test) + smoke GPU compile (100 xe × 2 round): fedavg fedprox scaffold afpha fd_ids lwfednids
  pfedes fedproto fedtgp tinyproto nilm perfedskd mobihet. Chưa có: **APFed, V2X-HtFL, CANPerFL (chờ PDF)**.
- Quyết định owner 29/10 (đã áp): gộp model theo n_k cho mọi phương pháp; MobiHet n_k; AFPHA 8 cụm scenario×run + μ thích nghi;
  TinyProto-FP; headline prototype = luật của bài.
- Tốc độ T4 đo thật: FedAvg W = 1 toàn 5.343 xe 223 s/round train (2×T4). Notebook dựng npy một lần (400 s) rồi memmap.

## Bước tiếp

1. **W = 1 đã khóa** (owner 30/10); gate w075/w05/w1 còn chạy chỉ để ghi nhận.
2. Còn chưa push (chờ slot): fd_ids, lwfednids, pfedes, fedtgp, tinyproto, nilm, perfedskd. Sinh notebook 50 round với W = 1 (`scripts/gen_notebook.py --method <m> --width W --max-hours 11.5`),
   push theo quota (`kaggle_account.py quota`). Phương pháp 2 model (mobihet, nilm, pfedes) ~3× FedAvg ⇒ có thể cần session 2
   (`--resume-kernel <acct>/<slug>`; cùng `--run`).
3. Centralized báo cáo (`run_central.py` không `--gate`, 50 epoch) cho W = 1 và W đã khóa.
4. Khi có PDF: APFed, V2X-HtFL, CANPerFL (đọc bài → docstring `methods/<alias>.py` → test toy → smoke GPU).
5. Pull output (`pull_kernel_output.py`), `scripts/report.py runs/pulls/*` → `docs/research/results.md`.

## Việc tay (owner)

- 3 PDF đã đủ (30/10): apfed, v2x_htfl, canperfl. Sửa bảng "còn thiếu" trong [papers/README](../../papers/README.md).
- Kaggle CLI: owner báo 30/10 đã đủ trừ **dinhan25** (kiểm lại `kaggle_account.py list`). minhtrit06 hết GPU quota tới 31/10:
  `python .claude/skills/kaggle-training-notebook/scripts/kaggle_account.py add-account <user>`.
