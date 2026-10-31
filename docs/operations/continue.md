# Continue — bàn giao sống, 31/10/2026 (giờ VN)

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) →
[bài học vận hành](lessons.md) → trang này. Sửa tại chỗ. Lịch sử 29–30/10: [snapshot](../archive/continue-2026-10-30.md).

## Đang chạy (Kaggle; W&B `21522798-uit/fl-dagsnet-veremi-v2`)

### MobiHet v4 mới — owner yêu cầu bỏ W&B, 30/10/2026

- Phiên này sở hữu `minhiw/mobihet-v4-weighted-nowb/1`; push thành công, status **RUNNING**.
  [Notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-weighted-nowb).
- Log trực tiếp xác nhận torch 2.10.0+cu128, Python 3.12.13, **2 Tesla T4**;
  `test_method_runs[mobihet]` + `test_next_round_resume`: **2 passed**, 15 deselected.
  Lần kiểm mới nhất: **RUNNING, SMOKE PASS**; NPY hoàn tất sau 1.433s. Smoke r1 train774s, r2 train856s;
  đã restart/resume thành công. Smoke compile gate FAIL → eager; **production compile gate cả hai rank PASS**
  (gradient finite, grad_rel ~0,043). Production bắt đầu 30/10 13:30:50 UTC (20:30:50 VN), budget còn8,68h;
  Kiểm 31/10: **COMPLETE ở production r25/50**, log `budget stop before round 26`.
  Phiên dừng bình thường khoảng04:54 giờ VN ngày31/10, chưa hoàn tất mục tiêu50round.
  R25 private/proxy macro-F1:0,883182/0,887442; accuracy:0,914857/0,915196;
  client F1 mean:0,824924/0,849356. Applied586396, skipped9 (0,001535%).
  Cần kiểm artifact/checkpoint hai rank và resume r26–50; chưa khởi chạy session2.
  Các số dưới đây là log các round đầu:
  Macro-F1 private/proxy r1:0,8011/0,4371; r2:0,7095/0,7888; r3:0,8183/0,8357; r4:0,8355/0,8478.
  Client F1 mean r4:0,6481/0,8137. R1 train1341s/eval402s; r3–4 train1143s/eval36s (~19,7 phút/round).
  Nếu tốc độ giữ nguyên, 50 round ~16,5h train/eval; phiên này budget8,68h sẽ cần nối tiếp, chưa đủ r50.
  Smoke 32 xe: r1 private/proxy macro-F1 0,7920/0,6065; r2 0,7546/0,8036,
  client F1 mean r2 0,5975/0,7443. Không phải kết quả full dataset.
- Dataset owner chọn: `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`. Không chuẩn hóa lại.
  Recipe MobiHet hiện hữu, aggregation sample-weighted; W=1, batch64, seed42, cosine50.
- Notebook chạy test CPU → decode/audit 5.343 client và số dòng → smoke 32 client r1,
  khởi động process mới resume r2 → kiểm applied/skipped/finite → train đủ client tới r50 hoặc budget.
  Smoke không dùng điểm test để chọn HP. Session timeout 11h, budget nội bộ 10h kể cả setup, chừa 0,4h finalize.
- Builder: `scripts/build_mobihet_v4.py --owner minhiw`; notebook tại `notebooks/mobihet-v4-weighted-nowb/`.
  Snapshot nhúng 37 file local; base SHA `36ac7ffafcb45974ed736b4d838279728d00e884` **không chứa patch local**.
  SHA256 notebook: `c760a77f38b9756535a93e7ffe9a9f265f2726a94e65cb6d614de027e642fef8`.
  Output có `source_snapshot.json`, `source_manifest.json`, `SMOKE_PASS.json` khi smoke thành công,
  `smoke/` và `mobihet-v4-weighted-nowb/` (metrics/CM/models/state theo rank).
- Quota trước launch: minhiw 20,9296h free, reserved0; hoangdang1711 13,8727h; dangthiet 15,6100h.
  Host Windows này không có torch; kiểm động thực hiện trên Kaggle. CLI2.2.4, Python311 có SDK/nbformat.
- Tiếp: đọc log account-scoped `kaggle kernels logs minhiw/mobihet-v4-weighted-nowb --follow`;
  status/output dùng `/1`. Log stream có `/1` trả404; stream không version hoạt động.
  Khi xong: pull/audit đủ round, CM, config, checkpoint; budget stop thì dựng session resume cùng source snapshot,
  không dùng generator mặc định clone base SHA (thiếu patch). Chưa có bằng chứng hiệu năng cải thiện.

- **Session 2 COMPLETE, r50 audit PASS:** `minhiw/mobihet-v4-weighted-nowb-s2/1`, push 31/10;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-weighted-nowb-s2).
  Worker xác nhận torch2.10.0+cu128, Python3.12.13, 2 Tesla T4. Đã restore byte-hash đúng 37 file source;
  resume preflight PASS: checkpoint round25 đủ cả hai rank, tensor finite, RNG MobiHet có mặt, metric liên tục r1–25.
  R50 private/proxy macro-F1 0,89530/0,89954; client mean 0,84678/0,86605; class13 0,11591/0,11727.
  Pull 104 file bỏ state lớn exit0; audit 50 CM ×2 view, 9.347.679 dòng/view/round, metric/source/finite/skipped PASS.
  W&B tắt. Quota minhiw đầu session2:30h free.
  Builder `scripts/build_mobihet_v4_resume.py`; notebook SHA256
  `b9a49cee2bd3ae6d5e53c1dd4520445bdc658d41fcbd9ba3da1ea431f3a929cd`.

- **P0 eval-only r25 kết thúc ERROR sau khi đã ghi artifact; audit khoa học PASS, gate FAIL:** `minhiw/mobihet-v4-p0-local-prior-r25/1`;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-p0-local-prior-r25). Không train lại: nạp checkpoint hai rank r25,
  kiểm `private_raw` khớp artifact cũ, rồi chấm `private = z + log π_k − log π_g`. Gate khóa trước launch:
  local-prior làm macro-F1 0,88319→0,88122, client mean 0,82493→0,81169, p10 0,67724→0,64526.
  ERROR do raw eager-vs-compiled lệch nhỏ vượt assertion 1e-12, không phải eval crash; không rerun. Builder
  `scripts/build_mobihet_v4_p0_eval.py`; notebook SHA256
  `3d1c64fba5d623293a813d84c0582b0a8ad2a6d2b28b20bd9cfd3a60fd8fff50`; W&B tắt.

- **P1a identifiability COMPLETE, audit PASS:** `minhiw/mobihet-v4-p1-identifiability/1`;
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-p1-identifiability). Chỉ đọc `train/`, HGB + MLP 3×256
  fit fold train và chấm full fold val cho binary benign/timeDelay trên 77 feature; thêm exact/rounded collision audit.
  Gate khóa: best val F1 lớp13 ≥0,50 ⇒ method branch; <0,30 ⇒ raw dataset P1b; giữa hai mức ⇒ P1b.
  HGB/MLP val F1 lớp13 =0,54138/0,54233 ⇒ gate `REPRESENTATION_SUFFICIENT_METHOD_BRANCH` PASS.
  Predictions đủ1.389.784 dòng val; confusion, counts, source hash và footer fingerprint audit PASS. Notebook SHA256
  `0e60027104ccb19459fd8fb599fc66f7c52e8282679ac49cf0eb25dfc044f730`; W&B tắt.

- **RelKD train→val paired arm v2 đang RUNNING:** `minhiw/mobihet-v4-relkd-val-r10/2`,
  [notebook](https://www.kaggle.com/code/minhiw/mobihet-v4-relkd-val-r10); local notebook `notebooks/mobihet-v4-relkd-val-r10/`, builder
  `scripts/build_mobihet_relkd_val.py`, SHA256 `c095997e9c1c4e75b21fd59aa2007db63f992648b7b0efba9f090e5d5c3018e9`.
  Chỉ decode train/, chạy matched `mobihet` và `mobihet_rel` tới r10 trên horizon50; unit/resume + smoke2-round là hard gate
  trong notebook. V1 train+resume smoke r1–2 khỏe nhưng checker đòi metadata `method` ở rank1 ⇒ ERROR sau smoke;
  sửa assertion rank-aware, không đổi khoa học. V2 startup 2×T4; unit/method/exact-resume RelKD 3 PASS; đang decode train-only.
  Gate khoa học ghi ở [MobiHet improvement](../methods/mobihet-improvement.md). W&B tắt.

### Các phiên kế thừa từ bàn giao trước (chưa xác minh trên host này)

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

1. Khi RelKD xong: pull/audit control+candidate r1–10 và `RELKD_GATE.json`; PASS mới dựng final train+val→test,
   FAIL thì chuyển specialist 0↔13, không tune RelKD. Không mở dataset v5/P1b.
2. `fl2-nilm-r2` dừng budget (~r38) ⇒ session 2, cùng `--run`:
   `python scripts/gen_notebook.py --owner accanniez --slug fl2-nilm-r2s2 --method nilm --run fl2-nilm-r2 --resume-kernel accanniez/fl2-nilm-r2 --max-hours 11.5`
   rồi `embed_wandb_key.py` + `chmod 600` + `ast.parse` mọi cell + push `-t 43200`.
3. `fl2-tinyproto-r3` xong ⇒ pull (`pull_kernel_output.py ... --pattern '^(?!.*state(_rank\d+)?\.pt$)'`), audit, thêm vào results
   (headline `client_proto`); kiểm view prototype ≈ view head (smoke 100 xe r2: 0,819 vs 0,825).
4. Viết `docs/methods/` cho APFed, V2X-HtFL, CANPerFL; sửa bảng "còn thiếu" trong [papers/README](../../papers/README.md).
5. MobiHet tiếp ở repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (bàn giao riêng `docs/operations/continue.md` của repo đó);
   arm cuối 50 round so với bảng results ở đây.

## Lưu ý vận hành

- `--set` qua `json.loads` ⇒ chuỗi cần dấu `"` JSON (xem [lessons](lessons.md) 30/10). Notebook push phải nhúng key W&B
  (`embed_wandb_key.py`; quên ⇒ `SystemExit: W&B credential not embedded`, bản 1 fl2-canperfl).
- Commit không có trailer `Co-Authored-By` (owner 30/10).

## Việc tay (owner)

- Không có việc gấp. minhtrit06 hết GPU quota tới 31/10. dinhan25 chỉ có token (chạy được qua `kaggle_as.py`).
