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
- Override owner 30/10 cho lần chạy MobiHet v4 mới: **không dùng W&B**, dùng log/artifact Kaggle; dataset `odixe0502/veremi-nextgen-v4-scaled-per-vehicle`. Chi tiết phiên mới ở bàn giao.
- Task < 1 h chạy local; task dài chạy Kaggle (16 account; CLI mới đăng nhập 6, owner bổ sung sau). Env conda `nckh`.
- Tốc độ train phải tối ưu ⇒ `core/grouped.py`: nhiều client một bước (bmm), dữ liệu thường trú GPU.

## Trạng thái — 31/10 (giờ VN)

- 19/19 phương pháp code + test. **W = 1 khóa** (DAGSNet 395.408 tham số).
- 14 run FL 50 round + centralized + CANPerFL **xong, audit PASS** ⇒ [results](docs/research/results.md) (centralized 0,9593;
  FL tốt nhất APFed client 0,9119; FedAvg 0,9042). Còn chạy: NILM-FL (cần session 2), TinyProto r3 (sửa luật prototype).
- Chi phí FLOPs/băng thông mọi phương pháp: [cost](docs/research/cost.md). MobiHet: repo riêng (bên dưới).
- MobiHet v4 r50 audit PASS: private/proxy macro-F1 0,8953/0,8995, class13 ~0,116. P0 local-prior FAIL;
  P1a 77-feature oracle PASS (F1 lớp13 0,5423),
  mở method branch; paired RelKD train→val r10 v2 đang chạy song song, startup tests PASS (v1 lỗi checker sau smoke r2).
  Chưa cần dataset v5/raw. Không W&B.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Đang làm gì, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Kết quả 50 round, audit, lưu ý từng run | [results](docs/research/results.md) |
| Cấu hình chung, cách chấm, cổng độ rộng | [protocol](docs/research/protocol.md) |
| Danh sách phương pháp, trạng thái | [methods](docs/research/methods.md) |
| Dataset, cột, số liệu, caveat | [knowledge/dataset](knowledge/dataset.md) |
| Dựng DAGSNet cho client | [knowledge/dagsnet](knowledge/dagsnet.md) |
| Máy local, Kaggle, lỗi kernel | [local-env](knowledge/local-env.md), [kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md) |
| Git backup gần nhất | [backup](docs/operations/backup.md) |
| Bài báo, cái nào thiếu | [papers/README](papers/README.md) |
| Phương pháp MobiHet: phát triển tiếp, kết quả v1 20/50/100 client | repo [`odixe06/mobihet`](https://github.com/odixe06/mobihet) (v2; v1 ở `history/v1/docs/methods/full-drift/`) |
