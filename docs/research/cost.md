# Chi phí tính toán và băng thông — mọi phương pháp

Đo, không ước tay: [scripts/cost.py](../../scripts/cost.py) → [cost.json](cost.json) (W = 1, 30/10/2026). Thời gian GPU thật
mỗi round: `t_train`, `t_eval` trong `rounds.jsonl` / W&B của từng run (2×T4). Byte MobiHet: đo mỗi round (`bytes_up`,
`bytes_down`, Q3 + LZMA2 trên mẫu 64 xe/rank).

## Ký hiệu

| | | W = 1 |
|---|---|---|
| P, B | tham số, buffer BN của DAGSNet | 395.408; 3.264 |
| F | FLOPs một lượt forward / dòng (FlopCounterMode, matmul + conv) | 4,45 MFLOP |
| n_k | dòng train của xe k; Σ n_k = 37.361.273 | |
| K | số xe = 5.343 | |
| d, C | chiều hidden (prototype), số lớp | 256; 16 |
| N_p | số prototype (xe, lớp) upload mỗi round ≤ K·C | |
| S | epoch server (FedTGP, V2X-HtFL) | 100 |

Chi phí client mỗi round = n_k × c_m × F, c_m đo bằng `cost.py` (một round thật của method trên dữ liệu toy không đệm;
FedAvg = 2,99 ≈ forward + backward = 3, kiểm chéo). Một epoch dữ liệu đầy đủ của mọi xe ở FedAvg = 37,4 M × 13,3 MFLOP ≈ 5,0·10¹⁴ FLOP/round.

## Bảng

| phương pháp | client / dòng / round (c_m × F) | Big-O client / round | byte lên / xuống / xe / round | Big-O server / round |
|---|---|---|---|---|
| FedAvg | 2,99 × F = 13,3 MFLOP | O(3 n_k F) | 1,595 MB / 1,595 MB = 4(P+B) | O(K P) |
| FedProx | 2,99 × F | O(3 n_k F + s_k P) (proximal) | 4(P+B) / 4(P+B) | O(K P) |
| SCAFFOLD | 2,99 × F | O(3 n_k F + s_k P) (control variate) | 3,176 MB / 3,176 MB = 4(2P+B) | O(K P) |
| AFPHA | 2,99 × F | O(3 n_k F + s_k P) | 4(P+B) / 4(P+B) | O(K P) (8 cụm + khoảng cách) |
| FD-IDS | 3,99 × F | O(4 n_k F) (teacher forward) | 4(P+B) / 4(P+B) | O(K P) |
| LwFedNIDS | 0,29 × F = 1,3 MFLOP (model tỉa 70 %, 39.494 tham số) | O(3 n_k F′) | 0,162 MB / 0,162 MB | O(K P′); tỉa một lần O(P) |
| pFedES | 7,98 × F | O(8 n_k F) (2 lượt F_k + bước G) | 1,4 KB / 1,4 KB (extractor proxy) | O(K P_G) |
| FedProto | 2,99 × F | O(3 n_k F) | 16,4 KB / 16,4 KB = 4 C d | O(N_p d) |
| FedTGP | 3,99 × F (+ lượt thu prototype) | O(4 n_k F) | 16,4 KB / 16,4 KB | O(S N_p C d + S d²) |
| TinyProto | 3,99 × F | O(4 n_k F) | 1,7 KB / 1,7 KB (prototype thưa) | O(N_p d) |
| NILM-FL | 5,97 × F | O(6 n_k F) (2 model, KD hai chiều) | 4(P+B) / 4(P+B) | O(K P) |
| PerFed-SKD | 4,99 × F | O(5 n_k F) | 4(P+B) / 4(P+B), chỉ xe được chọn | O(K P) |
| MobiHet | 8,97 × F | O(9 n_k F) (private trên 1,5 dòng + proxy + teacher) | đo: ~117 KB / ~114 KB (r1, Q3 + LZMA2; dense 1,595 MB ⇒ ~14×) + bootstrap một lần 197 KB | O(K P) |
| APFed | 5,97 × F | O(6 n_k F) (lượt học θ + train) | 4(P+B) / 4(P+B) | O(K P) |
| V2X-HtFL | 4,11 × F (P = 461.456: + projection d×d) | O(4 n_k F + n_k d²) | 16,4 KB / 16,4 KB | O(S N_p C d + S C² d) |
| CANPerFL | pha 1 = FedAvg × 50 round; pha 2 = 10 epoch × 2,99 × F, một lần | O(3 n_k F) mỗi round + O(30 n_k F) một lần | pha 1 = FedAvg; pha 2 = 0 | O(K P) |

s_k = số bước local của xe k (⌈n_k / 64⌉). Byte = fp32 trừ khi phương pháp nén (MobiHet). Tổng băng thông của run =
byte/xe/round × K × 50 (PerFed-SKD: × tỉ lệ xe được chọn, `selected_frac` mỗi round; MobiHet: tổng `bytes_*` đo được).

Lưu ý:
- FLOPs đếm matmul/conv (quy ước chuẩn của FlopCounterMode); phép elementwise, lượng tử hóa, LZMA không tính.
- Server FLOPs trong `cost.json` (`server_flops_per_round_toy`) đo trên 8 xe toy; ở quy mô thật tỉ lệ với N_p (Big-O ở bảng).
- Thời gian GPU thật không tỉ lệ đúng với FLOPs (engine grouped, đệm, bộ nhớ): đọc `t_train` từ run, không suy từ c_m.
