# CONTEXT — fl_dagsnet_veremi_v2 / FL baselines + MobiHet trên VeReMi NextGen v4 (5.343 xe)

Đọc CONTEXT → [LESSONS](LESSONS.md) + [bài học vận hành](docs/operations/lessons.md) → [bàn giao](docs/operations/continue.md).

## Bài toán

IDS V2X 16 lớp, học cộng tác: **mỗi xe nhận là một client** (5.343 client, dataset v4 của repo `mobihet-features`).
Dựng lại **15 phương pháp** trong bảng owner + **MobiHet** (đề xuất của owner: model client khác nhau được, băng thông hạn chế) +
**centralized**, tất cả trên DAGSNet, **cùng một cấu hình** ([protocol](docs/research/protocol.md)). Build cũ (Dirichlet 20/50/100 client,
`odixe06/fl_dagsnet_veremi`, `dangvanhoang1711/FL_DAGSNet`) chưa sát bài gốc và không đồng nhất ⇒ build lại từ đầu, chỉ tham khảo.

## Quyết định owner (29/10/2026)

- Phạm vi: bảng "Chọn" = 1 + 2 ⇒ 15 phương pháp ([danh mục](docs/research/methods.md)), + MobiHet `gkd + la + kdpx` proxy q3_zl, + centralized.
- 50 round; **mọi client tham gia mỗi round, 1 epoch local trên hết dữ liệu train (fold train + val)**; HP chung cố định, headline r50.
- Đánh giá: model toàn cục trên toàn test; model riêng client trên **test của chính xe đó** (gộp = toàn test). Đủ 10 metric.
- DAGSNet được thu nhỏ (giảm kênh, giữ 4 nhánh) nếu centralized macro-F1 val kém full ≤ 0,01; độ rộng chung mọi phương pháp.
- Một seed (42). W&B online. Repo GitHub **public** `odixe06/fl_dagsnet_veremi_v2`; không đẩy data, PDF bài báo, credentials.
- Task < 1 h chạy local; task dài chạy Kaggle (16 account; CLI mới đăng nhập 6, owner bổ sung sau). Env conda `nckh`.
- Tốc độ train phải tối ưu ⇒ `core/grouped.py`: nhiều client một bước (bmm), dữ liệu thường trú GPU.

## Trạng thái — 29/10 tối

- Khung repo, skills (chép từ `mobihet-features`), knowledge (máy, dataset, DAGSNet 77 đầu vào), protocol xong.
- Lõi `core/` + FedAvg, FedProx, SCAFFOLD xong; 9 test PASS (grouped ≡ DAGSNet fp64, AdamW ≡ torch, metric ≡ sklearn, eval ≡ tham chiếu).
- 29/10 ~22:25: 4 notebook cổng độ rộng centralized + probe FedAvg đang chạy Kaggle ([continue](docs/operations/continue.md)). W chưa chốt.
- Smoke local FedAvg 400 client: macro-F1 0,378 → 0,835 → 0,844 (r1–r3); tốc độ ~10× sau sửa (LESSONS).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Cấu hình chung, cách chấm, cổng độ rộng | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, trạng thái | [methods](docs/research/methods.md) |
| Dataset, cột, số liệu, caveat | [knowledge/dataset](knowledge/dataset.md) |
| Dựng DAGSNet cho client | [knowledge/dagsnet](knowledge/dagsnet.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Bài báo, cái nào thiếu | [papers/README](papers/README.md) |
| Phương pháp MobiHet gốc, kết quả 20/50/100 client | repo `../mobihet`: `docs/methods/full-drift/design.md`, `results/final/report.md` |
