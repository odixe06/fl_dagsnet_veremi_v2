# LESSONS — bài học kỹ thuật (dữ liệu, model, train, đánh giá, code)

Đọc sau [CONTEXT](CONTEXT.md), cùng [bài học vận hành](docs/operations/lessons.md). Bài học kế thừa:
`../mobihet-features/LESSONS.md` (dataset v4), `../mobihet/LESSONS.md` (grep theo chủ đề; FL/KD/đánh giá trên DAGSNet).

## Kế thừa — áp dụng thẳng cho repo này

- Model client chỉ học nhãn của chính nó đạt đỉnh trên test có prior toàn cục sau 1–2 epoch rồi giảm (mobihet 13/09). Ở đây model client
  chấm trên test của chính xe ⇒ prior khớp hơn, nhưng Sybil test 17,7 % vs train 8,0 % vẫn lệch theo thời gian.
- Luôn báo F1 train cạnh F1 test; sụp đột ngột khi đổi cấu hình là dấu hiệu tối ưu hỏng (HGB L2 = 0, mobihet-features 28/10).
- Val "xe chưa thấy" ≈ val cùng xe khi train gộp nhiều xe; không lộ được học thuộc theo xe (mobihet-features 29/10).

## Dữ liệu

- 31/10 MobiHet P0 r25: thêm `log π_k − log π_g` lúc own-client inference tăng F1 timeDelay 0,0559 → 0,0731 nhưng
  giảm pooled macro-F1 0,8832 → 0,8812, client mean 0,8249 → 0,8117 và p10 0,6772 → 0,6453. Giữ logits raw;
  prior correction lúc train không suy ra rằng phải cộng lại lúc inference.
- 31/10 P1a train→val: trên đúng 77 feature, binary HGB/MLP tách benign–timeDelay đạt F1 lớp13 0,541/0,542;
  exact collision mẫu 200k/lớp bằng0. Representation có tín hiệu, nên không đổ lỗi chính cho dataset hoặc mở v5 trước;
  lỗi còn lại là chuyển tín hiệu binary vào multiclass/private qua objective/KD. Đây là oracle binary, không phải điểm MobiHet dự kiến.

- 30/10 MobiHet v4: resume cần RNG riêng của Gaussian (`gen_g`), stochastic quantization (`gen_q`),
  lịch mẫu (`env.gen`) và torch CPU/CUDA theo rank. Chỉ lưu weights/version không tái hiện round kế tiếp.
  Test CPU train r1 → serialize → r2 so với restore → r2 đã PASS bit-exact trên Kaggle;
  chưa chứng minh bit-exact CUDA/compile. Smoke hai GPU kiểm riêng đường restart.
- 30/10: tính loss bằng `loss * ok` vẫn cho NaN khi loss NaN và ok=0; dùng `where`, chia số bước thực sự áp dụng.
  Thay đổi thống kê loss, không thay gradient/update. Theo dõi applied/skipped để tránh model đứng yên nhưng loss giả thấp.

- `pyarrow.dataset(<v4>/train)` lỗi "Unable to merge: Field scenario has incompatible types: large_string vs string" (cột trong file là
  `large_string`, partition hive suy `string`) ⇒ đọc từng file bằng `pq.read_table` (`core/data.py`).
- 77 đầu vào = 11 × 7 ⇒ DAGSNet giữ lưới 11 vị trí (`view(B, 11, 7).transpose(1, 2)`).

## Compile

- **`torch.compile` + AMP fp16 làm gradient NaN ở nhánh AlexNet grouped** (29/10): forward khớp eager (Δlogit 2e-3) nhưng
  backward của `amax` (maxpool 2, ceil) cho NaN ở mọi bước ⇒ optimizer bỏ mọi bước, model đứng ở init, FedAvg smoke acc 0,047 /
  loss "0,028" (chỉ là tổng các bước được áp = 0). fp32 compile và eager không bị. Sửa: `torch.maximum(x[..., 0::2], x[..., 1::2])`.
  Chặn tái diễn: `local.compile_gate` (một bước thật, so gradient compile vs eager, FAIL ⇒ eager) + cảnh báo `skipped_frac` mỗi round.
  Test tương đương fp64 eager không bắt được lỗi chỉ có ở đường compile + fp16.
- Shape tĩnh cho train (nhóm đệm tới đủ G client rỗng), shape động cho eval; mỗi shape mới = một lần compile ~2 phút.

## Tốc độ (đo 29/10, RTX 3050 4 GiB local)

- DAGSNet đơn, AMP, batch 8.192: full 46k mẫu/s eager, 59k compile; W 0,375: 78k / 113k.
- Grouped (G model/bước, bmm channels-last) compile: ~45k slot/s full width ≈ 77 % model đơn; eager chậm hơn ~2,5×.
  Lần compile đầu ~2 phút. Hiệu suất thực = slot thật / slot (padding): 0,85 ở batch 64 / 8.192 dòng mỗi bước grouped trên phân bố
  client v4 (nhóm theo kích thước); phần mất chủ yếu do client lớn (95k dòng = 1.487 bước tuần tự).
- **Stride của tensor đầu vào cũng là guard của `torch.compile` tĩnh** (29/10): `mask = valid[:, s*b:(s+1)*b]` có stride phụ thuộc số bước
  của nhóm ⇒ mỗi nhóm một lần recompile, quá 8 lần dynamo **im lặng chạy eager** (probe Kaggle FedAvg W 1: r1 1.243 s thay vì vài phút).
  Sửa: `.contiguous()` mỗi bước, `recompile_limit = 64`; kiểm bằng `TORCH_LOGS=recompiles` (còn 2 lần hợp lệ: dropout 0 của gate, no_grad eval).
- Nạp parquet trên Kaggle: mỗi rank đọc cả 410 file từ `/kaggle/input` mất 10–30 phút ⇒ một lần `data.build_npy` trong notebook rồi
  `V4_NPY` memmap theo rank.
- 30/10: luật suy luận prototype phải theo đúng bài từng phương pháp: FedProto/FedTGP dùng prototype toàn cục, **TinyProto Eq. 12 dùng
  prototype cục bộ dense c̄^L_ij**. Target thưa × μ của TinyProto chỉ là đích regularizer (0 ngoài mask) ⇒ dùng làm prototype suy luận
  cho F1 0,24 vs head 0,79. Audit: view prototype thấp xa view head là dấu hiệu sai luật suy luận.
