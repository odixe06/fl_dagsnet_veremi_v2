# Kết quả — 5.343 xe, DAGSNet W = 1, 50 round (30/10/2026)

Tính lại từ confusion matrix đã pull (`scripts/report.py runs/pulls/*/`), không chép từ log. Protocol: [protocol](protocol.md).
Headline mỗi phương pháp = model được đánh giá theo luật của bài (global; model client trên test của chính nó; prototype cho
FedProto/FedTGP). Byte = lên + xuống mỗi xe mỗi round. Một seed (42). Chi phí: [cost](cost.md).

Audit 30/10: mọi run đủ 50 round (CANPerFL: 10 epoch fine-tune sau FedAvg r50), không giá trị không hữu hạn, HP đúng protocol.

Chưa có / lưu ý:
- **TinyProto**: view prototype của run r2 dùng sai prototype (toàn cục thưa × μ, không phải prototype cục bộ dense của Eq. 12) ⇒
  bỏ khỏi bảng; chạy lại `fl2-tinyproto-r3` (SHA `55d13cc`).
- **NILM-FL**: đang chạy (`fl2-nilm-r2`, ~980 s/round ⇒ cần session 2).
- **APFed**: metric r50 đầy đủ; bank model client r50 mất (rank 0 hết RAM host khi ghi state cuối; sửa `bda3c07`).
- **SCAFFOLD**: không ổn định giai đoạn đầu (train_loss NaN r15–16, F1 r5 0,23, r15–16 0,25/0,23), hồi phục từ r18.
- **MobiHet**: repo riêng `odixe06/mobihet` (đang sàng lọc 10 round).
- FedAvg/FedProx/SCAFFOLD/FedProto chạy SHA `6617261` (không có snapshot model từng round; state r50 trên Kaggle).

| method | view | round | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted | client F1 mean | p10 | min | byte/client/round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized (train+val → test) | — | ep50 | 0,9580 | 0,9684 | 0,9580 | 0,9560 | 0,9531 | 0,9580 | 0,9580 | 0,9593 | 0,9580 | 0,9556 | — | — | — | — |
| afpha | global (headline) | 50 | 0,9211 | 0,9300 | 0,9211 | 0,9106 | 0,8932 | 0,9211 | 0,9211 | 0,8990 | 0,9211 | 0,9045 | 0,8652 | 0,7523 | 0,1786 | 3.189.376 |
| apfed | client (headline) | 50 | 0,9278 | 0,9377 | 0,9278 | 0,9217 | 0,9032 | 0,9278 | 0,9278 | 0,9119 | 0,9278 | 0,9169 | 0,8663 | 0,7533 | 0,1500 | 3.189.376 |
|  | global | 50 | 0,9216 | 0,9333 | 0,9216 | 0,9132 | 0,8947 | 0,9216 | 0,9216 | 0,9031 | 0,9216 | 0,9075 | 0,8645 | 0,7487 | 0,1806 | 3.189.376 |
| canperfl | client (headline) | 10 | 0,9233 | 0,9281 | 0,9233 | 0,9171 | 0,8986 | 0,9233 | 0,9233 | 0,9088 | 0,9233 | 0,9159 | 0,8779 | 0,7684 | 0,1545 | 0 |
|  | global | 10 | 0,9228 | 0,9354 | 0,9228 | 0,9147 | 0,8957 | 0,9228 | 0,9228 | 0,9042 | 0,9228 | 0,9085 | 0,8667 | 0,7540 | 0,1500 | 0 |
| fd_ids | global (headline) | 50 | 0,9220 | 0,9373 | 0,9220 | 0,9155 | 0,8914 | 0,9220 | 0,9220 | 0,8992 | 0,9220 | 0,9054 | 0,8664 | 0,7552 | 0,2176 | 3.189.376 |
| fedavg | global (headline) | 50 | 0,9228 | 0,9354 | 0,9228 | 0,9147 | 0,8957 | 0,9228 | 0,9228 | 0,9042 | 0,9228 | 0,9085 | 0,8667 | 0,7540 | 0,1500 | 3.189.376 |
| fedproto | client_proto (headline) | 50 | 0,8978 | 0,8778 | 0,8978 | 0,8923 | 0,8651 | 0,8978 | 0,8978 | 0,8698 | 0,8978 | 0,8932 | 0,7873 | 0,5911 | 0,0000 | 32.832 |
|  | client | 50 | 0,9013 | 0,8958 | 0,9013 | 0,8947 | 0,8588 | 0,9013 | 0,9013 | 0,8726 | 0,9013 | 0,8942 | 0,7788 | 0,5745 | 0,0000 | 32.832 |
| fedprox | global (headline) | 50 | 0,9214 | 0,9307 | 0,9214 | 0,9112 | 0,8941 | 0,9214 | 0,9214 | 0,8998 | 0,9214 | 0,9049 | 0,8658 | 0,7524 | 0,1786 | 3.189.376 |
| fedtgp | client_proto (headline) | 50 | 0,9020 | 0,8924 | 0,9020 | 0,8945 | 0,8610 | 0,9020 | 0,9020 | 0,8719 | 0,9020 | 0,8947 | 0,7794 | 0,5712 | 0,0000 | 32.768 |
|  | client | 50 | 0,9007 | 0,8989 | 0,9007 | 0,8950 | 0,8594 | 0,9007 | 0,9007 | 0,8740 | 0,9007 | 0,8933 | 0,7782 | 0,5691 | 0,0000 | 32.768 |
| lwfednids | global (headline) | 50 | 0,9162 | 0,9320 | 0,9162 | 0,9111 | 0,8830 | 0,9162 | 0,9162 | 0,8904 | 0,9162 | 0,8986 | 0,8476 | 0,7291 | 0,2139 | 324.000 |
| perfedskd | client (headline) | 50 | 0,8994 | 0,8903 | 0,8994 | 0,8931 | 0,8598 | 0,8994 | 0,8994 | 0,8710 | 0,8994 | 0,8928 | 0,8133 | 0,6573 | 0,1241 | 3.189.376 |
| pfedes | client (headline) | 50 | 0,8864 | 0,8692 | 0,8864 | 0,8800 | 0,8405 | 0,8864 | 0,8864 | 0,8511 | 0,8864 | 0,8803 | 0,7449 | 0,5264 | 0,0000 | 2.808 |
| scaffold | global (headline) | 50 | 0,9225 | 0,9011 | 0,9225 | 0,8842 | 0,8859 | 0,9225 | 0,9225 | 0,8911 | 0,9225 | 0,9003 | 0,8744 | 0,7738 | 0,2963 | 6.352.640 |
| v2x_htfl | client (headline) | 50 | 0,8974 | 0,8865 | 0,8974 | 0,8910 | 0,8554 | 0,8974 | 0,8974 | 0,8661 | 0,8974 | 0,8907 | 0,7642 | 0,5519 | 0,0000 | 32.832 |
|  | client_proto | 50 | 0,8947 | 0,8701 | 0,8947 | 0,8877 | 0,8597 | 0,8947 | 0,8947 | 0,8627 | 0,8947 | 0,8892 | 0,7576 | 0,5468 | 0,0000 | 32.832 |
