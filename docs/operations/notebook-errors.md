# Sổ lỗi kernel — bắt buộc đọc trước tạo notebook

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Trạng thái ERROR không quyết định rerun. Pull/verify round đã commit, source đã chạy và nguyên nhân trước.
Đủ output thì giữ kết quả; checkpoint hợp lệ thì nối boundary, không khởi động r1.

| kernel / build | lỗi đã biết và bằng chứng | ngăn lặp khi tạo notebook mới | quyết định |
|---|---|---|---|
| Chấm nhanh r27 `dinhan25/…-e29-48-s42ze…` 21/10 | `AssertionError ('fast gate', 0.2508)` ở r34 50c: gate tuyệt đối 0,25 hiệu chỉnh r1–4, logit round muộn lớn ~3× ⇒ sai số fp16 tăng theo; model không sai (r34 train đã audit) | Gate tương đối (tag r29, ≤ 2 % max\|logit\|); hiệu chỉnh tolerance trên round muộn nhất | r29–33 audit bằng `collect_eval.py --partial`; r34–48 chấm lại r29 |
| Push 06:31–08:35Z và 12:38Z 21/10 (6/26 lần push trong ngày): chấm `minhtran0601/…-e28-50-s42zc02…-f1`, `dinhan25/…-e09-28-s42ze…-f1`, `khanhmay0304/…-e03-12-s42zk02…-f1`, `minhtran0601/…-e45-50-s42za02…-f1`; train S2 `thuyanh316/…-s42ze03…`, `takids1909/…-s42zk03…` | Status RUNNING 2,5–4h nhưng không log live (`kernels logs -f` rỗng), không lên W&B, rồi ERROR log `[]`, `failure_message` rỗng, 0 output, executed 404 ⇒ **chưa cell nào chạy**. Đo trễ push → cell đầu (metadata executed notebook): thường 0,3–6 phút; 16:39Z 20/10 bốn account cùng chờ 48–49 phút; annhtnt za r28–50 chờ 131 phút rồi chạy bình thường ⇒ Kaggle hiện RUNNING khi chưa cấp máy 2×T4; kernel lỗi = chờ quá hạn. Không theo account (cùng account trước/sau chạy được), không theo dataset/cache/image (giống hệt bản chạy được), không có lỗi code nào để sửa | Không sửa được trong notebook. Watchdog sau push: 20–25 phút không log live ⇒ coi như chưa có máy, push lại nguyên trạng sang account khác (API không cancel được: nhờ owner Stop session kernel cũ trên UI để nó không chạy sau và tốn quota đôi). Tránh push dồn nhiều kernel cùng lúc | Push lại nguyên trạng: zc r28–50 minhtriethihi `-f2`, ze r9–28 minhtrit06 `-f2`, zk r3–12 accanniez `-f2`, S2 ze04 annhannh331, zk04 takids1909, za r45–50 odixe0502 `-f2` |
| FullDrift final r24 zc/zi 20/10 | Gaussian constructor hash cứng pilot manifest; final-only runtime không có file ⇒ cả2rank crash trước round1; collector 54file/kernel, 0round; notebook childrc1 nhưng KaggleCOMPLETE | Test filesystem chỉ final, `smoke=False`, cả2arm×20/50/100c×2rank; propagate subprocess error. Không sửa 40module legacy/source đã push | Owner dừng za/ze/zg/zk; source mới r25 qua local gates; owner yêu cầu push đủ6, kiểm remote startup từng kernel; [recovery](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/final-recovery-2026-10-20.md) |
| H1 full1 q1 accanniez15/10 | Host maintenance/Docker daemon sau16299,7s; API0artifact/executed404; W&B summary25/22 nhưng không weights | Status ERROR không là numericalFAIL; xác minh output và restore checkpoint trước. W&B không là checkpoint. Dùng audited parent16/14 nếu output mất; logging project mới tách nhánh recovery | Giữ kernel lỗi; q2 nối16/14, không r1; [proof](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/full1-unavailable-2026-10-15-afternoon.json) |
| V2-A seed42 minhtran0601; seed43 dinhan25 | EF sketch khuếch đại residual, loss/parameter tăng, BN calibration nonfinite; [chẩn đoán](https://github.com/odixe06/mobihet/blob/main/archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json) | Guard finite update/state/BN, đo residual contraction/norm ở đúng epoch/batch; preflight codec không đủ. Không dùng lại adaptation numerical FAIL | Không rerun y nguyên; các arm hoàn tất vẫn có giá trị |
| V2-B seed42 minhtran0601; seed43 dinhan25 | K512/own512 proxy loss không hữu hạn r1, trước commit; K1024 hoàn tất nhưng teacher gate FAIL | Preflight đúng batch/epoch production; smoke4step/batch32 không thay batch512/full epoch. Gating độc lập từngK, không mở KD/ZO khi teacher FAIL | Không rerun512; không dùng1024 che lỗi512 |
| V6-A seed42 minhtran0601; seed43 dinhan25 | G59 bearing=dx/dist bùng nổ trên synthetic gần dist0, 12arm lỗi; control/G61 vẫn hoàn tất | Toy missing/coincident + sampled synthetic support/norm/nonfinite gate trước SGD; không gọi interval rộng là faithful reconstruction | G59 numerical FAIL, G61 quality/byte FAIL; không rerun y nguyên |
| Hybrid full calibration hybf1 minhtrit06; hybf3 trietbackup | Workers rc0, budget stop trước r3, controller đòi mọiarm đúng3 nên kernel ERROR; source/checkpoint r2/r3 hợp lệ | Controller phân biệt session boundary, số round mới và target50. Budget từngarm/worst/pairmax +setup/compile/I/O +15%; không assert mọiarm phải có cùng round | Nối actual r2/r3; không train lại r1 |
| Hybrid cont01 minhtrit06 14/10 | Push trả `not valid kernel sources` cho parent ERROR dù owner-scoped output listing đọc được; notebook vẫn được push | Push parser phải phát hiện refused input; require-resume trước data decode. ERROR output cần dataset private checkpoint +hash gate khi không attach được; listing không thay mount proof | Giữ notebook đã PUSHED, tạo tag mới dùng bridge; trietbackup dùng bridge trước push |

Exact IDs của tám ERROR lịch sử tại [status snapshot](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/kernel-status-2026-10-12.json) và
[status hiện có](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/kernel-status-current.json); status có thời điểm, không thay audit. Các bảng/report từng build tại
[đề xuất](https://github.com/odixe06/mobihet/blob/main/docs/proposals.md), [Hybrid full audit](https://github.com/odixe06/mobihet/blob/main/docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/feasibility-20261014.md).

## Những lỗi preflight/runtime đã gặp

- Dataset/checkpoint upload: `-r zip -t`, manifest SHA/size và marker-last; không CSV conversion, không bỏ thư mục.
- Create dataset trả success trước xử lý xong; ListDatasetFiles có thể403 trong thời gian đó. Đọc status/đợi ready,
  retry read có giới hạn, không upload/push trùng. Vẫn kiểm owner/access;403 riêng lẻ không chứng minh nguyên nhân.
- Private kernel source có thể bị service từ chối mà push vẫn exit0. Kiểm response, actual mount/import guard;
  mất response thì status trước retry, không push trùng.
- W&B resume: preflight log không được tự tăng step; BoundaryCells dùng adapter `commit=False` cho preflight-only.
  Kiểm history monotonic từ round checkpoint; W&B resume không tự nạp weights.
- Zero GPU hoặc sai runtime: metadata cả notebook và JSON, pinned image, assert2T4/sm75/torch trước train;
  không bỏ assert hoặc reinstall torch để cứu run.
- Verifier load GPU tensor: `map_location='cpu'` trước so với packetCPU. Lỗi checker thì sửa checker, kiểm output cũ,
  không train lại model đang đúng.
- CUDA deterministic: CUBLAS config trước CUDA, deterministic algorithms, stable sort/tie; không warn-only để gọi exact-resume PASS.
- Fingerprint tất cả `mh/*.py`: đóng băng source trong full-vs-resume, kể cả module chưa dùng.
- ClassTail local15/10: `scan_seconds` trong algorithm/diag làm full-vs-resume khác dù weights đúng.
  Timing chỉ ở round diagnostics; loại khỏi algorithm state. Local4arm exact-resume đã PASS.
- Dataset checkpoint15/10 hiện giữ `package/` và `runs/` trong listing; bản cũ từng bỏ một level.
  So actual all-page names/sizes, recursive discovery và CPU production import; không hardcode việc bỏ prefix.
- Notebook private-inline có key trong platform history; không quét platform notebook để kết luận driver training lỗi.
  Pull nền tảng vào ignored private_push0600, audit bản redacted; không đưa key vào dataset/log/docs.
- Network reset khi quota/listing: retry **read-only** có giới hạn, giữ UNKNOWN nếu chưa đọc được. Không suy auth hỏng,
  không lặp push có tác động bên ngoài.

Guard runtime phát hiện lỗi; không biến adaptation FAIL thành PASS. Nếu thấy training đang active có nonfinite/OOM/worker kill,
báo chủ dự án và yêu cầu dừng; không tự đổi HP hoặc bỏ client. Stable thì ngừng health polling, vẫn collect terminal.

[CONTEXT](../../CONTEXT.md) · [Notebook preflight batch](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/resume-batch-preflight-2026-10-14.json) ·
[BoundaryCells helper/controller preflight](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/boundary-utility-preflight-2026-10-14.json).

15/10 CrossClientRank utility local emitted-controller fixture: `if not active and pending: break`
đã dừng queue ngay khi pair vừa hoàn tất dù còn budget. Sửa trước GPU push: chỉ break khi còn thời gian
không đủ worst-round+margin; normal/costFAIL/workerFAIL/budget fixtures kiểm đủ8arms.
Dataset scientific source chưa đổi; controller chưa PUSHED khi phát hiện.

15/10 W&B config JSON chuyển integral float1,0 thành int1; tái hash config từ API có thể báo fingerprint
mismatch giả. Inspector khôi phục declared argparse float types trước hash, rồi so frozen source/fixture/fork.
Không sửa fingerprint run hoặc rerun training vì lỗi checker.

15/10 CrossClientRank utility seed42: CE checkpoint r3 có `wire=b''`; PyTorch weights-only loader
từ chối `GLOBAL bytes`. Minimal empty-byte fixture tái hiện cả local2.13, không phải lỗi riêng2.10.
Fixture trước chỉ chạy pooled arm, không chứa empty CE wire; preflight phải cover cả CE/control state.
Scoped `torch.serialization.safe_globals([bytes])` giữ `weights_only=True`, không sửa frozen source/model.
Actual r3 weights/state/payload/predictions audit PASS; cost1,2556/1,2180×CE.
Recovery dùng tag mới và actual r3, CPU mount/audit trước GPU, không SGD lại r3.
Helper dataset verifier cũng đã sửa root-level manifest filename bị `split('/')[1]` IndexError;
đây là checker bug, không dataset corruption hoặc lý do upload/train lại.

## Phiên tối 15/10 — lỗi build và cách tránh

| sự cố | bằng chứng | quy tắc cho build sau |
|---|---|---|
| CCR seed43 v1 cùng lỗi emptybytes như seed42, nằm ERROR ~1,5h không ai biết | lỗi sau167s; owner Console "vẫn chạy" là hàng đợi/UI cũ | Lỗi tất định từ code/serializer của một seed ⇒ kiểm ngay status mọi seed cùng build; chuẩn bị recovery song song. Platform RUNNING/Console không thay status API + log |
| Hybrid-full chạy lại arm đã có ở full-validation | h_dml/h_dml_own bit-exact 8/8 round; local/Gaussian/G-own lệch ≤0,008; hybf3 100% trùng, owner dừng | Trước khi lập build/batch: so config (bỏ fingerprint source) với arm đã audit ở build khác; trùng thì tái dùng reference, chỉ chạy arm mới (Hybrid: h_hybrid, h_hybrid_own) |
| Budget CCR 2,25h không đủ cho reference | rank arm ~1 phút/round, cr_dml ~4,75 phút/round ⇒ dml dừng r16 | Ước budget theo **loại arm chậm nhất** (DML/Hybrid = 2 model + eval), không trung bình; xếp arm chậm sớm hoặc session riêng |
| Session full 4h chỉ được 1 round | hybf1 q2: worst-round 7.020s, session kết thúc 1,95h | Round full 1,6–2h/arm: session phải ≥ setup + k×worst-round; 4h chỉ ra 1 round, lãng phí boundary |
| Collector CCR báo SHA lệch giả ở run resume | `metrics.csv` (và result.json) được driver ghi lại khi chạy tiếp | Checker miễn file tổng hợp được ghi lại; vẫn bắt buộc round/config/payload khớp. Đã sửa `collect_rank_utility.py` |
| Verifier dataset fail vì `__pycache__` | code test local chạy trong thư mục input sau khi lập danh sách | Lập manifest/danh sách file **trước**, chạy test với `PYTHONDONTWRITEBYTECODE=1` hoặc ngoài thư mục stage |
| `kaggle datasets create` lỗi `Expecting value` | 110/110 file upload thành công, status `ready` | Lỗi parse response sau upload: kiểm `datasets status` + all-page names/sizes, không upload lại |
| `pkill -f <pattern>` giết chính shell đang chạy | exit144 hai lần | Dừng tiến trình nền bằng PID cụ thể, không pattern trùng dòng lệnh hiện tại |
| Push session thứ 3 bị từ chối | `Kernel push error: Maximum batch GPU session count of 2 reached`; push_nb không tạo PUSHED, status 404 | Mỗi account tối đa **2 batch GPU session**; lập kế hoạch arm theo 2 session/account, phần dư xếp sau khi session kết thúc |
| 14 arm/seed không vừa 7h | AntiMem round 191–259s (eval 4 role mỗi round) so ước128s | Đo round đầu thật trước khi chốt số arm/session; eval nhiều role tốn thêm ~50%; xếp arm quan trọng (đối chứng/QProxy) sớm |
| `kaggle datasets create` 403 ở account khác | odixeuit/minhtrit06: upload file xong rồi `CreateDataset` 403 (OAuth và KGAT token đều vậy); list/đọc vẫn được, dung lượng 2–12GB | Account bị chặn tạo dataset mới: đừng chép dataset sang account đó. Dùng **dataset gốc share cho account chạy** (Kaggle web → dataset Settings → Sharing), kiểm `kaggle datasets files` bằng account chạy trước khi push. `gen_antimem.py --dataset_owner` tách owner dataset/kernel. **21/10:** cả hai tạo được dataset private nhỏ (probe 27 B ready rồi xóa) ⇒ lỗi cũ gắn với dataset GB hoặc hạn chế đã gỡ; dataset lớn phải thử lại trước khi dựa vào |
| FullDrift v1: mọi arm chết ở `Run.__init__` sau setup | `TypeError: int + list`: gate production cộng `manifest['counts']['train_k']` (histogram 16 lớp) như số | Smoke local bỏ qua nhánh `if not smoke` ⇒ tách gate production thành hàm và **test trên manifest thật** (`test_fd_local.py --manifest`, pull `source/data_manifest.json` của session đầu). Session v1 chỉ tốn ~5–10 phút/session |
| Collect chậm, collector Hybrid không thoát; audit FD song song bị guard dừng (~4GB/tiến trình) | Collector hardcode `pull_kernel_output -j 6` (q3 full 8,8GB/346s, 25MiB/s); `collect_hybrid_full.py` lặp vô hạn vì hbf1-cont01/hybf3-q2 không bao giờ AUDITED (16/10) | Theo skill [pull-outputs](../../.claude/skills/kaggle-training-notebook/references/pull-outputs.md): tốc độ theo tổng kết nối, ~32 bão hòa. Collector có `--jobs` (mặc định 16); nhiều session thì **một tiến trình mỗi session** (`collect_fd.py --only`), chia ~32 kết nối; Hybrid dùng `--once`. **Pull song song, audit tuần tự**: `collect_fd.py` giữ lock `papers/.artifact-verification.lock` khi audit |
| Collector FD vòng 3 assert sai sau khi thêm arm vòng 4 | `collect_fd.py` so `fd.py` pulled với `source/proj/fd.py` **hiện tại**; fd.py đã đổi (hkd, gaussian_m4b) sau stage r3 (16/10) | So với bản **stage của vòng đó** (`artifacts/input/fd-src-<tag>/fd.py`), không với source đang sửa; nhiều vòng song song thì source sống luôn đi trước |
| fd-s43ya-20261015-r23-v1 minhtran0601 (20/10) | `Prepared real pilot` ở 9.929s (các kernel trước 198–262s, ~45× chậm), sau đó >30 phút chưa tới `start` ⇒ W&B chưa có run 3h+ sau khi chạy; cùng source/dataset kernel khác bình thường ⇒ máy Kaggle chậm I/O | Run W&B chưa lên sau ~20 phút kể từ lúc kernel thật sự chạy ⇒ owner xem log Kaggle; bước chuẩn bị > ~600s ⇒ không đủ budget 15 epoch, dừng sớm và repush tag mới (không đợi mốc 7h) | owner dừng tay 20/10 ~07:30Z |
| Dòng "ERROR: pip's dependency resolver" + cảnh báo wandb trong log mọi kernel FD (74/81 log) | `wandb==0.29.0` cần `opentelemetry-api>=1.43`, image Kaggle có 1.38 (ghim bởi google-adk/otel-sdk) ⇒ pip nâng và báo xung đột; `wandb.login(key=..., anonymous=...)` báo "api key in code"/"anonymous has no effect"/tạo netrc. Vô hại với train/kết quả. "Debugger warning: frozen modules" và dòng lặp đôi là của runtime Kaggle (trước cell đầu), không sửa được từ notebook | `gen_fd.py` từ 20/10: `wandb==0.28.0` (không phụ thuộc opentelemetry) + `--progress-bar off`; đặt `WANDB_API_KEY` rồi `wandb.login(verify=True)` (không key/anonymous/relogin — test HOME sạch PASS). `fd.py` vẫn login kiểu cũ (source r23 đóng băng) ⇒ sửa ở tag `--final`. Skill: [wandb.md](../../.claude/skills/kaggle-training-notebook/references/wandb.md) | áp dụng từ notebook sinh sau 20/10 |

## Log live (kiểm lại 20/10/2026)

CLI hiện tại có SSE `kaggle kernels logs <owner/slug> --follow`; đã đọc được W&B auth và `Prepared real pilot` của r24 RUNNING.
`kernels logs` không follow vẫn có thể rỗng. Bọc `timeout 35s`, chạy qua `kaggle_as.py <owner>`, log private0600 rồi redact trước công bố;
exit124 là hết thời gian đọc, không phải lỗi kernel. [Bằng chứng](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/final-r24-2026-10-20/final-startup-2026-10-20.json).
Trước khi yêu cầu owner mở UI hoặc dừng vì W&B chưa xuất hiện, kiểm stream này để phân biệt setup/mount/decode với treo hoặc traceback.
QUEUED là chờ GPU, giữ kernel. Log live vẫn không thay source/checkpoint/prediction audit khi terminal.
| fd-s42ze-20261015-r25-v1 thuyanh316 (50c, `SESSION_HOURS=2`) và fd-s42zi-20261015-r25-v1 odixe0502 (100c, 3h), ERROR 20/10 18:44Z / ~19:10Z | Chuẩn bị dữ liệu final 4.420 s (ze) và 8.792 s (zi); cùng cấu hình chạy được: za 1.019 s, zc 1.566 s, zg 1.114 s, zk 5.090 s. Lúc vào vòng train còn 0,77h / 0,56h < ngưỡng 1h của notebook ⇒ bỏ qua arm, `summary={}` ⇒ assert ERROR; 0 round, không checkpoint. Không lỗi code/dữ liệu; session ngắn + chuẩn bị dữ liệu dao động 17–147 phút theo máy (ya r23: 165 phút) | (1) Không đặt session ngắn hơn ~(chuẩn bị xấu nhất 2,5h + phần train); session train final dùng gần trần 12h, để `fd.py --max_hours` tự dừng trước round không kịp. (2) Bước chuẩn bị chạy trong session GPU vẫn tính quota GPU ⇒ chuyển sang kernel CPU (cache fixture, kiểm SHA bằng manifest) trước khi nhân số notebook. (3) Notebook nên báo rõ "hết budget trước train" thay vì assert chung | repush với session dài (chờ owner duyệt phương án) |
| Push notebook chấm `mh-fd-e01-…` (20/10 ~19:50Z) | `kaggle kernels push` trả **409 Conflict** cả 4 notebook: slug kernel trùng slug dataset checkpoint `mh-<rid>` cùng owner | Kernel và dataset không được trùng slug trong một account: kernel chấm dùng `mobihet-<rid>`, dataset `mh-<rid>` (`gen_eval.py` đã sửa). 409 khi push = chưa tạo kernel, sửa slug rồi push lại an toàn | đã push lại 4/4 |
| Kernel cache fixture CPU (20–21/10) | Chuẩn bị trên CPU dao động 852 s (annhtnt K20) → 18.854 s (odixe0502 K20, 5,2h); 4/14 kernel chạy > 5h | Không tốn quota GPU nhưng có thể chậm nhiều giờ ⇒ dựng cache **trước** khi account cần (ngay khi phân bổ), không để session GPU chờ cache; notebook GPU không có cache vẫn tự chuẩn bị (an toàn) | theo dõi tới COMPLETE |
