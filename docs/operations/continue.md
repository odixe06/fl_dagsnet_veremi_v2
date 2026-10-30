# Continue — bàn giao sống, 30/10/2026 ~02:05

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

## Trạng thái 30/10 ~06:50 (giờ VN; W&B live)

Kernel cũ (không lưu model mỗi round) đã CANCEL hết; `dinhan25/fl2-v2x-htfl` cũ owner không dừng được ⇒ bỏ qua (tự hết 12 h).
Quota GPU refresh 31/10 00:00 UTC.

| kernel | trạng thái | F1-macro (pooled) | s/round train |
|---|---|---|---|
| fl2-central-w1 | **xong ep50** | 0,9593 (trần centralized) | |
| fl2-fedavg | **xong r50** | global 0,9042 (p10 0,754) | 224 |
| fl2-fedprox | **xong r50** | global 0,8998 (p10 0,752) | |
| fl2-fd-ids-r2 | **xong r50** | global 0,8992 (p10 0,755) | |
| fl2-afpha-r2 | **xong r50** | global 0,8990 (p10 0,752) | |
| fl2-scaffold | **xong r50** | global 0,8911 (p10 0,774) | |
| fl2-lwfednids-r2 | **xong r50** | global 0,8904 (p10 0,729) | 72 |
| fl2-fedproto | **xong r50** | client 0,8726 / proto 0,8698 (p10 0,575) | 230 |
| fl2-tinyproto-r2 | r48 | client 0,7809 / proto 0,2265 | 303 |
| fl2-fedtgp-r2 | r45 | client ~0,871 / proto ~0,865 | 319 |
| fl2-perfedskd-r2 | r44 | client ~0,866 | 354 |
| fl2-v2x-htfl-r2 | r44 | client ~0,863 / proto ~0,852 | 332 |
| fl2-apfed-r2 | r30 | client ~0,897 | 493 |
| fl2-pfedes-r2 | r30 | client ~0,841 | 483 |
| fl2-nilm-r2 | r14 | client ~0,852 / proxy ~0,853 | 980 ⇒ ~38 round/session, **cần session 2** (`--resume-kernel accanniez/fl2-nilm-r2`, cùng `--run`) |
| fl2-canperfl | push 30/10 04:40, trietbackup, SHA e8ad753, 10 epoch từ state r50 fl2-fedavg, `-t 10800` | | |

Quote `--set` chuỗi: xem [lessons](lessons.md) 30/10.

## Trạng thái 30/10 ~01:30 (giờ VN; W&B UTC 18:30 29/10)

W = 1 giữ (owner 30/10) dù cổng W = 0,75 đạt (−0,0063). SHA `6617261` = code cũ (resume per-client SAI: chỉ rank 0 ghi state);
SHA `924bdcd`/`73bb96d` = code mới (state theo rank, ghi khi dừng budget).

| kernel | account | SHA | tiến độ |
|---|---|---|---|
| fl2-fedavg | trietbackup | 6617261 | r18 0,887; 240 s/round ⇒ xong ~3,3 h |
| fl2-fedprox | catbaochau | 6617261 | r15 0,875 |
| fl2-scaffold | catbaochau | 6617261 | r15 0,248 — **dao động mạnh** (0,23 ↔ 0,85, trùng ‖c‖ vọt); để chạy tới r50 |
| fl2-fedproto | minhtriethihi | 6617261 | r11 client 0,863 / proto 0,854; 271 s/round ⇒ vừa 1 session |
| fl2-mobihet | trietbackup | 6617261 | **r1 1.335 s + eval 398 s; r2 > 36 min** ⇒ ≥ 30 h, không resume được ⇒ đề nghị owner dừng tay; điều tra (FLOPs chỉ 3× FedAvg, eval 10× FedProto cùng hàm ⇒ nghi nghẽn bộ nhớ GPU / recompile) rồi chạy lại SHA mới |
| fl2-central-w1 | minhtran0601 | 6617261 | ep18 |
| fl2-afpha | minhtriethihi | 6617261 | ERROR (OOM norm (K,P)) — đã sửa, chạy lại = fl2-afpha-v2 |
| ~~fl2-afpha-v2, fd-ids, lwfednids, pfedes, fedtgp, tinyproto, nilm, perfedskd, apfed, v2x-htfl~~ | (cùng account dưới) | 924bdcd/73bb96d | **owner dừng tay** (không lưu model mỗi round) |
| **fl2-<m>-r2**: afpha, fd-ids, lwfednids, pfedes, fedtgp, tinyproto, nilm, perfedskd, apfed, v2x-htfl | annhannh331, annhtnt, nguyenminhtran060113, takids1909, thuyanh316, anhtruong1909, accanniez, khanhmay0304, odixeuit, dinhan25 | **903ac42** | push 30/10 ~02:00, RUNNING; lưu `models/rXXX.pt` mỗi round + state đầy đủ r50 |
| fl2-mobihet | trietbackup | 6617261 | **owner dừng tay**; MobiHet chuyển sang repo `odixe06/mobihet` |

Bước tiếp:
1. MobiHet: tìm nguyên nhân chậm (W&B system metrics GPU memory; profile local subset W = 1), sửa, chạy lại SHA mới.
2. CANPerFL: khi fl2-fedavg xong r50 ⇒ `gen_notebook.py --method canperfl --rounds 10 --input-kernel trietbackup/fl2-fedavg
   --set init='"/kaggle/input/**/fl2-fedavg/state.pt"'` (state.pt cũ ghi mỗi round ⇒ có r50).
3. Chi phí: `scripts/cost.py` → `docs/research/cost.json` (FLOPs/dòng/round đo bằng FlopCounterMode, payload). Còn viết
   `docs/research/cost.md` (Big-O + bảng: byte/client/round × K × 50, FLOPs = n_k × hệ số, server FLOPs, giây GPU/round thật).
4. Doc `docs/methods/` cho APFed, V2X-HtFL, CANPerFL; sửa bảng "còn thiếu" trong papers/README.
5. Theo dõi mọi run; pull output; `scripts/report.py`.

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
