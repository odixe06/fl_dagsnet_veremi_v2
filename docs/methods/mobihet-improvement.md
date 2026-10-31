# MobiHet v4 — chẩn đoán round 25 và hướng cải tiến

Trạng thái 31/10/2026: MobiHet v4 r50 audit **PASS**; private/proxy macro-F1 0,8953/0,8995. P0 r25 **FAIL**;
P1a 77-feature identifiability **PASS**. RelKD train→val đang chạy; không dùng official test để chọn feature, loss, round hay ngưỡng.

## Bằng chứng hiện có

Nguồn: `rounds.jsonl` của `minhiw/mobihet-v4-weighted-nowb/1`, đủ r1–25; số dưới đây chưa được audit lại từ confusion matrix.

| view r25 | macro-F1 | accuracy | client-F1 mean | p10 | F1 lớp 13 `timeDelayAttack` | macro-F1 bỏ lớp 13 |
|---|---:|---:|---:|---:|---:|---:|
| private | 0,8832 | 0,9149 | 0,8249 | 0,6767 | **0,0559** | **0,9383** |
| proxy | 0,8874 | 0,9152 | 0,8494 | 0,7266 | **0,0368** | **0,9441** |

So FedAvg r50 0,9042, private thiếu 0,0210 và proxy thiếu 0,0168. Nếu chỉ lớp 13 thay đổi, private cần F1 lớp 13
0,392 và proxy cần 0,305 để bằng FedAvg. Các lớp yếu kế tiếp là benign (0), positionMirroring (7) và
suddenConstantSpeed (11); 12/16 lớp private đã ≥ 0,919.

Ba dấu hiệu không ủng hộ việc ưu tiên tăng bit hoặc tăng lượng Gaussian:

- Q3 residual tương đối chỉ 0,00364 ở r25; chưa chứng minh lượng tử là nút thắt.
- Proxy hơn private cả pooled macro-F1 (+0,0043) lẫn client-F1 mean (+0,0244); cơ chế tri thức hiện tại chưa tạo lợi cá nhân hóa.
- Lớp 13 có 1.924.480 dòng train, nên lỗi không giải thích bằng thiếu support toàn cục. Dataset hiện ghi nhận
  benign ↔ timeDelay là cặp nhầm chính vì attack dịch cả thời gian gửi và nhận.

## P0 — kiểm đúng luật quyết định, không train lại

**Vấn đề.** Private train bằng `CE(z + log π_k − log π_g, y)` nhưng `evaluate()` chấm logits thô `z`.
Logits thô là view đã hiệu chỉnh về prior toàn cục; headline lại chấm model k trên test của chính client k.

**Can thiệp.** Từ checkpoint r50, báo ba view trên cùng predictions:

1. `private_raw = z` (view hiện tại);
2. `private_local = z + log π_k − log π_g` (đúng logits đi vào CE local);
3. hỗn hợp logits private/proxy với hệ số khóa trên fold `val`, không fit trên test.

**Dấu hiệu bác bỏ.** `private_local` không tăng client-F1 mean ít nhất 0,005 hoặc làm pooled macro-F1 giảm quá 0,005;
hỗn hợp không tăng ít nhất 0,005. Khi đó không đổi luật headline.

**Kết quả r25 — FAIL.** Artifact CM đủ 5.343 client/9.347.679 dòng và metric tính lại khớp JSON. So raw,
`private_local` làm macro-F1 0,88319 → 0,88122, client mean 0,82493 → 0,81169 và p10 0,67724 → 0,64526.
F1 lớp 13 tăng 0,05587 → 0,07312 nhưng không bù thiệt hại các lớp khác. Giữ logits raw làm headline; không rerun P0.

## P1 — cổng định danh lớp 13 trước khi sửa thuật toán

**Vấn đề.** F1 lớp 13 gần 0 ở cả private lẫn proxy, trong khi 15 lớp còn lại đạt 0,94. Có thể 77 feature hiện tại không chứa
đủ tín hiệu quan sát để tách timeDelay khỏi benign.

**Đo trên train/val, không official test:**

- audit exact/near duplicate feature giữa nhãn 0 và 13, theo client và sender;
- binary oracle 0-vs-13 bằng HGB và MLP trên 77 feature, chấm fold `val` theo thời gian/client;
- lặp oracle trên raw timing/provenance hợp lệ: receive time độc lập, send time, message sequence và cửa sổ causal theo sender;
- cấm đưa `source_run`, world/attack archive hoặc đường dẫn file vào model vì chúng tiết lộ loại tấn công.

**Dấu hiệu quyết định.** Nếu oracle raw đạt F1 lớp 13 ≥ 0,50 còn oracle 77 feature < 0,30, lỗi nằm ở dataset representation.
Nếu cả raw oracle < 0,30, lớp 13 không định danh được dưới observation model hiện tại: không tune loss để tạo điểm giả;
hoặc bổ sung sensor/timestamp độc lập, hoặc công bố task 15 lớp sau khi gộp observationally-equivalent class và chạy lại mọi baseline.

**P1a kết quả — PASS method branch.** Fit fold train, chấm full fold val; official test không đọc. HGB F1 lớp13 0,54138
(recall 0,7864), MLP 3×256 đạt 0,54233 (recall 0,8506); best vượt gate 0,50. Exact collision trên mẫu cân bằng
200k/lớp = 0; rounded 0,1 chỉ 1/200k. Artifact predictions 1.389.784 dòng, confusion, source hash và dataset footer
`16d3475e...` audit PASS. Kết luận giới hạn: 77 feature có tín hiệu cho binary 0↔13; chưa chứng minh specialist sẽ nâng
multiclass MobiHet. Không mở dataset v5/P1b lúc này; chuyển P3, ưu tiên reliability-gated KD rồi specialist.

## P2 — dataset v5 nếu raw còn tín hiệu

Chỉ mở khi P1 chứng minh raw có tín hiệu. Dựng feature causal, fit bằng fold train:

- giữ `rel_delay`, `ses_dt`, `ses_dt_send`, `ses_dt_skew` ở cả physical/log scale và normal-score; tránh mất ngưỡng vật lý do clipping/tie;
- thêm chuỗi 8–16 message theo `(client, sender)`: inter-arrival, inter-send, clock-skew drift, message-age residual,
  position prediction tại thời điểm nhận và neighbor-consensus residual;
- thêm nhánh temporal 1D nhỏ hoặc specialist 0-vs-13; classifier 16 lớp chỉ gọi specialist khi top candidates thuộc {0,13};
- scaler, rolling state và feature selection chỉ fit trên train. Fold val dùng chọn thiết kế; sau khi khóa mới retrain train+val một lần.

**Cổng val:** F1 lớp 13 ≥ 0,35; benign F1 giảm ≤ 0,005; macro-F1 tăng ≥ 0,015; client p10 không giảm quá 0,005.
Nếu dataset đổi, phải chạy lại tối thiểu centralized, FedAvg, APFed và MobiHet trên cùng v5; số v4 không còn là đối chứng công bằng.

## P3 — sửa phương pháp nếu representation đã đủ

Ưu tiên theo thứ tự, mỗi arm đổi một yếu tố:

1. **KD có reliability theo lớp/client.** Trên calibration từ train/val, client chỉ nhận KD từ proxy ở lớp/cặp lỗi mà teacher
   tốt hơn private; tắt KD cho timeDelay khi proxy đang kém. Không thêm byte. Falsifier: private class13/client mean không tăng ≥ 0,005.
2. **Tách head local và head trao đổi.** Head local chỉ CE trên dòng thật và dùng cho own-client; proxy/global head nhận KD và Gaussian.
   Backbone có thể chung, head local không rời client nên vẫn hỗ trợ kiến trúc dị thể. Falsifier: private không vượt proxy client mean
   ít nhất 0,005 hoặc p10 giảm.
3. **Control proxy CE-only q3.** Bỏ `KL(private→proxy)` nhưng giữ Q3/error feedback để tách tác động KD khỏi nén.
   Chạy thêm dense/q3 matched control chỉ khi CE-only q3 vẫn hụt. Residual Q3 hiện nhỏ nên chưa ưu tiên tăng bit.
4. **Specialist benign–timeDelay** từ P2, loss pairwise/focal chỉ trên cặp 0/13. Nhận khi đạt cổng P2; không dùng class weight
   toàn cục vì lớp 13 không hiếm và có thể chỉ đổi recall lấy false positive benign.

### Arm P3.1 đã build — `mobihet_rel`

Chỉ teacher→private KD đổi: với từng client/batch/class có mặt, tính mean true-label NLL của proxy snapshot và private;
KD real và Gaussian của lớp đó chỉ bật khi teacher NLL thấp hơn private. Lớp vắng mặt tắt gate. Proxy CE/KD, Gaussian,
Q3/error feedback, aggregation và byte không đổi. Không thêm HP tune (`margin = 0`). Baseline module `mobihet` giữ gate tắt.

Notebook paired `minhiw/mobihet-v4-relkd-val-r10/2` đang chạy. V1 chỉ lỗi checker metadata rank1 sau smoke r2;
v2 sửa assertion, không đổi source phương pháp/gate. Cả control và candidate fit fold train, chấm full fold val,
dùng horizon cosine 50 và dừng r10; official test không decode. Gate khóa: ΔF1 lớp13 ≥0,005, Δclient mean ≥0,005,
Δmacro-F1 ≥−0,005 và Δp10 ≥−0,005. Candidate FAIL thì không kéo dài/tune gate; chuyển specialist 0↔13.
Startup 2×T4 và 3 test unit/method/exact-resume PASS; W&B tắt.

Không ưu tiên: tăng Gaussian batch, copula/GMM khác, tăng nhiệt độ, cap KD đơn giản, tăng Q3→Q4. Các build MobiHet cũ đã cho
thấy những hướng này thường chỉ tăng tốc đầu hoặc cải thiện proxy mà không chuyển ổn định sang private; v4 chỉ mở lại khi control mới
chỉ ra đúng cơ chế.

## Trình tự thí nghiệm

1. Hoàn tất và audit r50 hiện tại; không thay run đang chạy.
2. P0 hậu kỳ trên checkpoint r50.
3. P1 identifiability trên train/val.
4. Chọn đúng một nhánh: P2 nếu raw có tín hiệu; P3 nếu 77 feature đã đủ.
5. Smoke 2 round → arm 10 round seed42 với cổng khóa → arm 50 round; arm PASS mới lặp seed43.
