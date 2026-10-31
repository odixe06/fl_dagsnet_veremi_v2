# Danh mục phương pháp — 15 baseline + MobiHet + centralized

Nguồn: [bảng owner](https://docs.google.com/spreadsheets/d/1v2tfkEUUfD9wUbOUp4m8ID-H_WK_hgERiC8jq4MSCLU/edit?gid=0) — owner 29/10 chọn
nhóm "Chọn" = 1 và 2 (bỏ FedCAPS, OptiFLIDS, GRU_FedAvg, FedDF). Cấu hình chung: [protocol](protocol.md). Bài báo: [papers/](../../papers/README.md).
Build cũ (dataset Dirichlet 20/50/100 client) chỉ để đối chiếu cơ chế: `odixe06/fl_dagsnet_veremi` (A), `dangvanhoang1711/FL_DAGSNet` (B).

"Model chấm": **toàn cục** = một model, chấm toàn test; **client** = model riêng mỗi xe, chấm test của xe đó (bảng: "đại diện" / "trung bình").

| # | alias | bài | model chấm | trao đổi mỗi round | build cũ | trạng thái v2 |
|---:|---|---|---|---|---|---|
| 0 | centralized | DAGSNet (Khan 2025) | toàn cục | — | A `edl_cmso` | chờ cổng độ rộng |
| 1 | FedAvg | McMahan 2017 | toàn cục | θ | B `fedavg` | code + smoke local (r3 0,844 / 400 xe) |
| 2 | FedProx | Li 2020 | toàn cục | θ | B `fedprox` | code + test toy |
| 3 | SCAFFOLD | Karimireddy 2020 | toàn cục | θ + control variate | B `scaffold` | code + test toy |
| 4 | AFPHA_EDL_CMSO | Khan 2025 (Sci. Rep.) | toàn cục | θ | A `afpha` | code + test toy (8 cụm scenario×run, μ thích nghi — owner 29/10) |
| 5 | FD-IDS | Peng 2025 (Sensors) | toàn cục | θ (+ KD) | A `fd_ids` | code + test toy |
| 6 | APFed_LDwCBN | Sci. Rep. 2025 (DOI 10.1038/s41598-025-17699-3) | toàn cục | θ | B `apfed` | **thiếu PDF** |
| 7 | LwFedNIDS | Bouayad 2024 (Access) | toàn cục | θ đã tỉa | A `lwfednids` | code + test toy (tỉa 70 % zero-shot L1 nhóm) |
| 8 | V2X-HtFL | OJVT 2025 | toàn cục (đại diện) | prototype + feedback sinh | B `v2x_HtFL` | **thiếu PDF** |
| 9 | CANPerFL | Hoang 2023 (Appl. Sci.) | client | θ (pretrain FL) → fine-tune local | B `canperfl` | **thiếu PDF** |
| 10 | TinyProto | Lee 2026 (AAAI) | client | prototype thưa | A `tinyproto` | code + test toy (biến thể FP — owner) |
| 11 | pFedES | Yi 2025 (AAAI) | client | extractor proxy nhỏ | A `pfedes` | code + test toy |
| 12 | NILM-FL | Li 2025 (TSG) | client | proxy | A `nilm_fl` | code + test toy |
| 13 | FedTGP | Zhang 2024 (AAAI) | client | prototype ↔ prototype toàn cục huấn luyện được | B `fedtgp` | code + test toy |
| 14 | PerFed-SKD | Singh 2024 (TCE) | client | θ (client được chọn) | A `perfedskd` | code + test toy |
| 15 | FedProto | Tan 2022 (AAAI) | client | prototype | B `fedproto` | code + test toy |
| 16 | **MobiHet** (đề xuất) | repo `mobihet`, `gkd + la + kdpx`, proxy q3_zl | client (private) + proxy | proxy 3-bit + LZMA2 | `mobihet/methods/full-drift` | r50 audit PASS + [kế hoạch cải tiến](../methods/mobihet-improvement.md) |

Đặc tả từng phương pháp (công thức theo số phương trình của bài, siêu tham số bài gốc, deviation) nằm ở **docstring đầu file**
`methods/<alias>.py` — một nguồn duy nhất, đọc cùng code. Toàn cục: [global-baselines](../methods/global-baselines.md).
Model chấm headline của FedProto/FedTGP/TinyProto = luật prototype của bài (`client_proto`); head phân loại báo kèm (owner 29/10).
Kết quả: `docs/methods/<alias>-results.md` từ artifact đã audit.
