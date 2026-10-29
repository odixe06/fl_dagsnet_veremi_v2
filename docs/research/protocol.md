# Protocol chung — mọi phương pháp chạy đúng cấu hình này

Owner chốt 29/10/2026. Một phương pháp chỉ được khác ở **cơ chế của chính nó** (loss, cái gì trao đổi, cách gộp, siêu tham số
riêng lấy từ bài gốc). Mọi thứ dưới đây dùng chung, code chung (`core/`), không sửa riêng cho một phương pháp.
Đổi bất kỳ dòng nào ⇒ chạy lại mọi phương pháp.

## Dữ liệu và client

| | |
|---|---|
| Dataset | v4 `odixe0502/veremi-nextgen-v4-scaled-per-vehicle` ([knowledge/dataset](../../knowledge/dataset.md)); 77 đầu vào theo [input_order.json](../../knowledge/input_order.json); không chuẩn hóa lại |
| Client | **5.343 xe nhận**, partition tự nhiên của dataset (không Dirichlet) |
| Train local | **toàn bộ** dữ liệu train của client = fold `train` + `val` (37.361.273 dòng). Không có tập chọn mô hình: headline = round cuối |
| Test | 9.347.679 dòng (20 % cuối timeline từng xe) |
| Tham gia | **mọi client mỗi round**, trừ khi chọn client là cơ chế của bài gốc (vd PerFed-SKD) — khi đó giữ đúng cơ chế, ghi rõ |

## Train

| | |
|---|---|
| Round | **50**, mỗi round mỗi client **1 epoch local** trên hết dữ liệu của nó |
| Model | DAGSNet ([knowledge/dagsnet](../../knowledge/dagsnet.md)); độ rộng chung `W` chốt bằng cổng centralized (dưới) — mọi phương pháp, mọi client cùng `W` |
| Batch local | **64** (Table 1 bài DAGSNet). Client nhỏ hơn 64 dòng: 1 bước với toàn bộ dữ liệu |
| Optimizer | AdamW, β (0,9; 0,999), ε 1e-8, weight decay 1e-4; trạng thái optimizer khởi tạo lại mỗi round (client không giữ moment qua round) |
| LR | cosine theo round: 1e-3 ở r1 → 1e-5 ở r50 (cố định trong một round) |
| Khác | grad-norm clip 1,0 theo từng client; AMP fp16 (T4); dropout 0,1 |
| Seed | 42 (một seed) |
| Init | mọi model khởi tạo từ cùng θ⁰ (seed 42) |

**Gộp model:** mọi trung bình tham số/BN buffer giữa client theo số mẫu n_k (owner 29/10), kể cả bài gốc gộp không trọng số
(SCAFFOLD, NILM-FL, PerFed-SKD, LwFedNIDS, MobiHet) — với 5.343 xe 76–95k dòng, trung bình 1/K kéo model về gần θ⁰ (smoke 100 xe, 2 round:
SCAFFOLD 1/K 0,17 so FedAvg n_k 0,69). Gộp prototype giữ đúng cơ chế của bài (TinyProto APS, FedTGP Q^c, FedProto Eq. 6).

Siêu tham số riêng của phương pháp (μ FedProx, nhiệt độ KD, λ prototype, …): **giá trị trong bài gốc**; bài không nêu ⇒ giá trị mặc định
của code chính thức; không có ⇒ chọn, ghi vào `docs/methods/<method>.md` là deviation. Không quét siêu tham số trên test.

## Đánh giá (mỗi round)

- **Model toàn cục** (FedAvg-like): chấm trên **toàn bộ** test.
- **Model riêng từng client** (personalized): model của client k chấm trên **test của chính xe k**; gộp 5.343 phần = đúng toàn bộ test.
  (Owner 29/10: chấm 5.343 model trên cả test là ~5·10¹⁰ forward/round, không khả thi.)
- Báo **đủ 10 metric** (accuracy; precision/recall/F1 macro, micro, weighted) trên ma trận nhầm lẫn **gộp**, cộng F1 từng lớp.
  Kèm phân phối F1-macro theo client (trung bình, p10, trung vị, min; macro trên các lớp có trong test của client đó).
- Headline: **round 50**, không chọn round theo test.
- Chi phí: byte truyền mỗi client mỗi round (lên + xuống, fp32 = 4 B/tham số trừ khi phương pháp nén), tổng 50 round; GPU-giây/round.

## Cổng chọn độ rộng DAGSNet (centralized)

Owner 29/10: được rút gọn số tham số mỗi client nếu đạt tiêu chí. Quy trình:
1. Train centralized DAGSNet full (W = 1, 395.408 tham số) và các bản thu hẹp đều theo kênh (W ∈ {0,5; 0,375; 0,25}) trên fold `train`,
   cùng lịch, chấm fold `val`.
2. Chọn W nhỏ nhất có macro-F1 val ≥ F1(full) − **0,01**. Khóa W trước mọi run FL.
3. Run centralized báo cáo (train + val → test) cho cả full và W đã chọn.

**Đã khóa: W = 1** (owner 29/10, trước khi cổng xong). Cổng hoàn tất (val macro-F1 ep20): W 1 0,9648 · 0,75 0,9585 · 0,5 0,9481 ·
0,375 0,9417 · 0,25 0,9296 ⇒ W = 0,75 **đạt** (−0,0063), nhưng owner 30/10 giữ W = 1 (không chạy lại). Số tham số: W 1 395.408 · 0,75 224.560 · 0,5 101.712 · 0,375 58.288 · 0,25 26.864.

Centralized dùng batch lớn (4.096, như build DAGSNet cũ) vì 37M dòng ở batch 64 không chạy được trong ngân sách; đây là deviation của
riêng kịch bản centralized.

## Hạ tầng (không ảnh hưởng kết quả)

- `core/grouped.py`: G model client chạy song song trong một bước (bmm), tương đương từng bit với DAGSNet tham chiếu ở fp64
  ([tests/test_grouped.py](../../tests/test_grouped.py)); client nhóm theo kích thước (`core/local.py`).
- 2×T4: mỗi GPU giữ nửa số client (xen kẽ theo kích thước) + dữ liệu của chúng; process group chỉ để cộng payload gộp và ma trận nhầm.
- Số W = 1 và W = 2 GPU không trùng bit (luồng RNG theo nhóm) — mọi run báo cáo dùng 2×T4.
