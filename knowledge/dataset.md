# Dataset v4 — VeReMi NextGen, client = xe nhận

Nguồn dựng: repo `odixe06/mobihet-features` (`src/v2/`), README gốc `../mobihet-features/data/v4-scaled/README.md`.
Kaggle public: **`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`** (bản chuẩn hóa, dùng để train). Bản raw đã chia: `odixe0502/veremi-nextgen-v3-split-per-vehicle`.
Local: `/home/odixe/project/mobihet-features/data/v4-scaled/` (không chép vào repo này).

## Cấu trúc

`train/scenario=<s>/world=<w>/part-*.parquet` và `test/...` (410 file mỗi phía), `clients.parquet` (mốc cắt từng client),
`scaler.json` (bảng CDF), `label_mapping.json`. Cột `scenario`, `world` kiểu `large_string` ở file nhưng partition hive suy ra `string`
⇒ `pyarrow.dataset(dir)` lỗi "Unable to merge"; đọc **từng file** bằng `pq.read_table`.

## Số liệu (đếm trên file local 29/10)

| | dòng | ghi chú |
|---|---:|---|
| train/ fold `train` | 32.685.166 | |
| train/ fold `val` | 4.676.107 | 12,5 % cuối phần train của mỗi xe |
| **train/ tổng (dùng để train FL/centralized)** | **37.361.273** | owner 29/10: client train hết dữ liệu local = train + val |
| **test/** | **9.347.679** | 20 % cuối timeline từng xe |

- **5.343 client** (`client_id` 0…5342) = `scenario × run × receiver_id`; 4 scenario (highway_2/7, urban_2/7). Mọi client có train và test, mọi client có benign.
- Dòng train/client: min 76 · p10 178 · **trung vị 1.368** · p90 18.177 · p99 86.116 · max 95.287.
- Dòng test/client: min 20 · p10 46 · trung vị 343 · p90 4.546 · max 23.822.
- Số lớp có mặt/client: train trung vị 15 (p10 8, min 2); test trung vị 12 (p10 5, min 2).

| id | lớp | train (train+val) | test |
|---:|---|---:|---:|
| 0 | benign | 9.773.821 | 2.172.713 |
| 1 | accelerationMultiplication | 630.778 | 155.909 |
| 2 | constantPositionOffset | 1.795.239 | 415.570 |
| 3 | constantSpeedOffset | 1.756.278 | 408.219 |
| 4 | dataReplay | 1.943.831 | 432.195 |
| 5 | dosAttack | 6.463.385 | 1.404.653 |
| 6 | feignedBraking | 483.556 | 108.792 |
| 7 | positionMirroring | 1.924.303 | 409.354 |
| 8 | randomPositionOffset | 1.915.642 | 434.231 |
| 9 | randomSpeedOffset | 2.214.237 | 495.126 |
| 10 | reversedHeading | 1.108.314 | 240.279 |
| 11 | suddenConstantSpeed | 225.294 | 63.314 |
| 12 | suddenStop | 820.377 | 236.178 |
| 13 | timeDelayAttack | 1.924.480 | 482.549 |
| 14 | trafficCongestionSybil | 3.287.699 | 1.652.147 |
| 15 | zeroSpeedReport | 1.094.039 | 236.450 |

## Cột

- Định danh/nhãn (KHÔNG đưa vào model): `scenario, world, run, client_id, fold, label (int8), receiver_id, sender_id, message_id, rcv_time_ns`.
- Đầu vào model: **77 cột float32** theo đúng thứ tự [input_order.json](input_order.json) = 71 feature (`scaler.json["order"]`) + 6 cờ NaN.
  Nhóm: `msg_*` 4, `rel_*` 8, `ses_*` 14, `trf_*` 6, `win_*` 9, `nbr_*` 2, `map_*` 4, `p4_std_*` 24; cờ `*_nan` 6.
- **Đã chuẩn hóa sẵn cả train lẫn test** (z = Φ⁻¹(CDF mid-rank của fold train), NaN → 0, |z| ≤ 3,72). **Không chuẩn hóa lại.**
  Scaler tính được kiểu federated (tổng histogram từng client) ⇒ dùng chung cho mọi phương pháp FL không vi phạm giả định FL.
- Giá trị nằm trong ±3,72 ⇒ lưu fp16 trên GPU an toàn (sai số làm tròn ≤ 2e-3).

## Caveat (đọc khi diễn giải kết quả)

- benign ↔ timeDelayAttack là cặp nhầm chính (timeDelay dời cả thời gian gửi và nhận ⇒ gần như không để lại mâu thuẫn).
- Mỗi world chỉ có một loại tấn công ⇒ benign giữa luồng Sybil/DoS dễ nhận hơn benign ở world yên tĩnh (shortcut bối cảnh).
- Sybil dồn nửa sau vòng đời xe nhận: 8,0 % train / 17,7 % test ⇒ test có prior khác train.
- Tham chiếu centralized của bên dựng dataset (fold val, xe chưa thấy): HistGradientBoosting 0,959, MLP 3×256 0,944 macro-F1.
