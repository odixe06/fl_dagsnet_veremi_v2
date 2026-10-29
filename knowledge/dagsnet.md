# DAGSNet cho v4 (77 đầu vào) — đặc tả để dựng model client

Nguồn: Khan et al. 2025, *Scientific Reports*, DOI `10.1038/s41598-025-94445-9` (`papers/afpha_edl_cmso.pdf`), §4.10, Eq. (38)–(48).
Code tham chiếu: [core/dagsnet.py](../core/dagsnet.py) (`DAGSNet`, `FULL`, `width`). Bản G model song song: [core/grouped.py](../core/grouped.py),
cùng tên `state_dict`, tương đương kiểm bằng [tests/test_grouped.py](../tests/test_grouped.py). Đặc tả bản 66 feature của repo cũ:
`../mobihet-features/knowledge/legacy-66/architecture.md` (đã kiểm từng lớp; bản này chỉ khác số kênh vào).

## 1. Hợp đồng vào/ra

```
x      : (B, 77) float — 77 cột đúng thứ tự knowledge/input_order.json (71 feature + 6 cờ NaN), đã chuẩn hóa sẵn
Fm     = x.view(B, 11, 7).transpose(1, 2)        # (B, 7, 11): 7 kênh × 11 vị trí — view RỒI MỚI transpose
logits : (B, 16) — CHƯA softmax; loss = CrossEntropyLoss; lớp theo knowledge/label_mapping.json
```

- 77 = 11 × 7 ⇒ giữ đúng lưới 11 vị trí của bản 66 cột (11 × 6). Patch = 7 cột liên tiếp theo thứ tự đầu vào.
  `view(B, 7, 11)` thẳng sẽ trộn sai cột mà vẫn chạy — lỗi im lặng.
- Không chuẩn hóa lại (dataset đã chuẩn hóa train lẫn test bằng scaler fit trên fold train, tính được kiểu federated).

## 2. Kiến trúc (Eq. 38–48)

```
Fm (B,7,11) ─┬─ stem₀ 1×1 → DenseNet1d  (3 lớp, growth g)          → (B, s+3g, 11)
             ├─ stem₁ 1×1 → GoogleNet1d (2 inception, nhánh c)       → (B, 4c, 11)
             ├─ stem₂ 1×1 → AlexNet1d   (3 conv a, 2 maxpool ceil)   → (B, a, 3)
             └─ stem₃ 1×1 → SqueezeNet1d(3 fire, squeeze q, expand e)→ (B, 2e, 11)
global average pool trục vị trí → concat (Eq. 47) → LayerNorm → Dropout → Linear→h → ReLU → Dropout → Linear→16 (Eq. 48)
```

- Mọi conv là `Conv1d(bias=False) → BatchNorm1d → ReLU`, `padding = k//2`. Inception: 1×1 / (1×1→3×3) / (1×1→5×5) / (maxpool3 → 1×1).
  Fire: squeeze 1×1 → [expand 1×1, expand 3×3]. AlexNet: `MaxPool1d(2, ceil_mode=True)` hai lần, 11 → 6 → 3 (bắt buộc ceil).
- Bốn stem độc lập (mỗi nhánh một stem). Init: mặc định PyTorch, seed 42 trước khi dựng.

## 3. Độ rộng

`width(W)` nhân mọi số kênh với W (làm tròn, ≥ 4); số lớp, kernel, cấu trúc giữ nguyên.

| W | s (stem) | g | c | a | q / e | h (head) | tham số | buffer BN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (FULL) | 96 | 32 | 32 | 128 | 32 / 48 | 256 | **395.408** | 3.295 |
| 0,75 | 72 | 24 | 24 | 96 | 24 / 36 | 192 | 224.560 | 2.479 |
| 0,5 | 48 | 16 | 16 | 64 | 16 / 24 | 128 | 101.712 | 1.663 |
| 0,375 | 36 | 12 | 12 | 48 | 12 / 18 | 96 | 58.288 | 1.255 |
| 0,25 | 24 | 8 | 8 | 32 | 8 / 12 | 64 | 26.864 | 847 |

395.408 = 395.024 của bản 66 cột + 4 stem × 96 kênh × 1 cột vào thêm. W dùng cho FL do cổng centralized chọn
([protocol](../docs/research/protocol.md#cổng-chọn-độ-rộng-dagsnet-centralized)); ghi W đã chốt vào CONTEXT, không vào đây.

## 4. Chỗ bài báo không quy định (deviation phải công bố, giữ như build cũ)

| chỗ | bài báo | bản này |
|---|---|---|
| stem 1×1 trước mỗi nhánh | không nhắc | 4 stem độc lập, s kênh |
| gộp 4 nhánh khác chiều dài (11/11/3/11) | "Concat" | global average pool rồi concat |
| head | một phép chiếu σ(WF+b) | LayerNorm → Dropout 0,1 → Linear(comb→h) → ReLU → Dropout 0,1 → Linear(h→16) |
| σ ở Eq. 48 | softmax | không có trong module (CrossEntropyLoss chứa log-softmax) |
| growth / số module / kênh | không nhắc | bảng §3 |
| bài toán | nhị phân | 16 lớp (Eq. 50 nhánh đa lớp) |
| §4.8 DWT→ViT→GAT, §4.9 CMSO, pruning/quantization | có | **không dùng** (DAGSNet trần) |
| huấn luyện (Table 1) | Adam lr 1e-3, batch 64, 100 epoch, CCE, BN, dropout | FL: AdamW lr 1e-3 cosine, batch 64, 50 round × 1 epoch ([protocol](../docs/research/protocol.md)) |

## 5. Hidden layer cho phương pháp prototype / representation

`DAGSNet.hidden(x)` = `ReLU(Linear(Dropout(LayerNorm(F_combined))))` (h chiều, trước Dropout + Linear cuối) — không thêm lớp, không đổi số tham số.
`grouped.forward(..., hidden=True)` trả cùng tensor cho G model. Phương pháp cần "feature extractor / classifier head" tách đôi dùng ranh giới này:
extractor = mọi thứ trừ `head.5`, classifier = `head.5` (`Layout.mask(["head.5"])`).
