# Danh mục phương pháp — 15 baseline + MobiHet + centralized

Nguồn: [bảng owner](https://docs.google.com/spreadsheets/d/1v2tfkEUUfD9wUbOUp4m8ID-H_WK_hgERiC8jq4MSCLU/edit?gid=0) — owner 29/10 chọn
nhóm "Chọn" = 1 và 2 (bỏ FedCAPS, OptiFLIDS, GRU_FedAvg, FedDF). Cấu hình chung: [protocol](protocol.md). Bài báo: [papers/](../../papers/README.md).
Build cũ (dataset Dirichlet 20/50/100 client) chỉ để đối chiếu cơ chế: `odixe06/fl_dagsnet_veremi` (A), `dangvanhoang1711/FL_DAGSNet` (B).

"Model chấm": **toàn cục** = một model, chấm toàn test; **client** = model riêng mỗi xe, chấm test của xe đó (bảng: "đại diện" / "trung bình").

| # | alias | bài | model chấm | trao đổi mỗi round | build cũ | trạng thái v2 |
|---:|---|---|---|---|---|---|
| 0 | centralized | DAGSNet (Khan 2025) | toàn cục | — | A `edl_cmso` | chờ cổng độ rộng |
| 1 | FedAvg | McMahan 2017 | toàn cục | θ | B `fedavg` | code xong (`methods/fedavg.py`), smoke |
| 2 | FedProx | Li 2020 | toàn cục | θ | B `fedprox` | |
| 3 | SCAFFOLD | Karimireddy 2020 | toàn cục | θ + control variate | B `scaffold` | |
| 4 | AFPHA_EDL_CMSO | Khan 2025 (Sci. Rep.) | toàn cục | θ | A `afpha` | |
| 5 | FD-IDS | Peng 2025 (Sensors) | toàn cục | θ (+ KD) | A `fd_ids` | |
| 6 | APFed_LDwCBN | Sci. Rep. 2025 (DOI 10.1038/s41598-025-17699-3) | toàn cục | θ | B `apfed` | **thiếu PDF** |
| 7 | LwFedNIDS | Bouayad 2024 (Access) | toàn cục | θ đã tỉa | A `lwfednids` | |
| 8 | V2X-HtFL | OJVT 2025 | toàn cục (đại diện) | prototype + feedback sinh | B `v2x_HtFL` | **thiếu PDF** |
| 9 | CANPerFL | Hoang 2023 (Appl. Sci.) | client | θ (pretrain FL) → fine-tune local | B `canperfl` | **thiếu PDF** |
| 10 | TinyProto | Lee 2026 (AAAI) | client | prototype thưa | A `tinyproto` | |
| 11 | pFedES | Yi 2025 (AAAI) | client | extractor proxy nhỏ | A `pfedes` | |
| 12 | NILM-FL | Li 2025 (TSG) | client | proxy | A `nilm_fl` | |
| 13 | FedTGP | Zhang 2024 (AAAI) | client | prototype ↔ prototype toàn cục huấn luyện được | B `fedtgp` | |
| 14 | PerFed-SKD | Singh 2024 (TCE) | client | θ (client được chọn) | A `perfedskd` | |
| 15 | FedProto | Tan 2022 (AAAI) | client | prototype | B `fedproto` | |
| 16 | **MobiHet** (đề xuất) | repo `mobihet`, `gkd + la + kdpx`, proxy q3_zl | client (private) + proxy | proxy 3-bit + LZMA2 | `mobihet/methods/full-drift` | |

Mỗi phương pháp khi build: `methods/<alias>.py` (chỉ cơ chế) + `docs/methods/<alias>.md` (tóm tắt bài, công thức, siêu tham số bài gốc,
deviation, khác gì so với build cũ). Kết quả: `docs/methods/<alias>-results.md` từ artifact đã audit.
