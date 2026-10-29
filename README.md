# fl_dagsnet_veremi_v2

Federated IDS cho V2X trên VeReMi NextGen v4: **5.343 client = 5.343 xe nhận**, 16 lớp, DAGSNet. Dựng lại 15 phương pháp FL
(FedAvg, FedProx, SCAFFOLD, AFPHA, FD-IDS, APFed, LwFedNIDS, V2X-HtFL, CANPerFL, TinyProto, pFedES, NILM-FL, FedTGP, PerFed-SKD,
FedProto), phương pháp đề xuất MobiHet và centralized, **cùng một protocol** ([docs/research/protocol.md](docs/research/protocol.md)).

- Dataset: Kaggle [`odixe0502/veremi-nextgen-v4-scaled-per-vehicle`](https://www.kaggle.com/datasets/odixe0502/veremi-nextgen-v4-scaled-per-vehicle) — [mô tả](knowledge/dataset.md).
- Model: [DAGSNet cho 77 đầu vào](knowledge/dagsnet.md).
- Bắt đầu: [CONTEXT.md](CONTEXT.md). Env: conda `nckh` ([environment.yml](environment.yml)).

```
core/      DAGSNet tham chiếu + bản grouped (nhiều client/bước), dữ liệu thường trú GPU, AdamW theo client, eval, metric
methods/   một file mỗi phương pháp (chỉ cơ chế)
scripts/   run_fl.py (chạy FL), centralized, benchmark, notebook Kaggle
tests/     test tương đương / đơn vị
knowledge/ dữ kiện bất biến (máy, dataset, DAGSNet)
docs/      protocol, danh mục phương pháp, kết quả, vận hành
```
