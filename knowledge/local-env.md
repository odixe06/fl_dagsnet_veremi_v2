# Máy local — kiểm 29/10/2026

| thành phần | giá trị |
|---|---|
| OS | Lubuntu 26.04.1 LTS (nền Ubuntu), LXQt; kernel 7.0.0-42-generic, x86_64 |
| CPU | Intel Core i5-12500H, 16 luồng |
| RAM / swap | 15,3 GiB / 8 GiB `/swapfile` (RAM trống thay đổi — owner chạy việc khác; luôn đo `free -h`) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GiB, sm_86; driver 595.99.02 |
| Đĩa | `/` 469 G (trống ~197 G ngày 29/10) |
| Conda | `/home/odixe/Tools/miniconda3`, env **`nckh`** (Python 3.12.15) — dùng chung với `mobihet`, `mobihet-features` |
| Package | torch 2.13.0+cu130 (CUDA 13.0, cuDNN 9.20), numpy 2.5.2, pyarrow 25.0.0, scikit-learn 1.9.0, wandb 0.29.0, kaggle 2.2.5 — [local-runtime.json](local-runtime.json), [environment.yml](../environment.yml) |
| Checkout | `/home/odixe/project/fl_dagsnet_veremi_v2` |
| Dataset v4 local | `/home/odixe/project/mobihet-features/data/v4-scaled` (4,3 GB parquet) — xem [dataset](dataset.md) |
| Kaggle CLI | `~/.kaggle/` (roster 16 account, token theo account); helper `.claude/skills/kaggle-training-notebook/scripts/kaggle_as.py` |
| W&B | key trong `~/.netrc` |
| GitHub | `gh` đăng nhập `odixe06` |

```bash
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh && conda activate nckh
cd /home/odixe/project/fl_dagsnet_veremi_v2
python scripts/run_local_guarded.py python <script> <args>   # watchdog dừng khi MemAvailable < 700 MiB
```

## Chạy ở đâu

- Owner 29/10: task **< 1 giờ chạy local**, task dài chạy Kaggle (2×T4, 16 account). Ước lượng thời gian/RAM trước khi chạy.
- Local không chứa nổi dataset đầy đủ: 46,7M dòng × 77 cột fp16 = 7,2 GB RAM, GPU 4 GiB. Local dùng tập con (vài trăm client)
  hoặc stream theo file. Kaggle: cả train (5,75 GB fp16) lẫn test (1,44 GB) nằm thường trú trên mỗi T4 (15 GB).
- sm_86 local không chứng minh hành vi compile/AMP trên T4 sm_75; T4 không có BF16 ⇒ AMP fp16 + GradScaler.
- Env `nckh` dùng chung: không cài/gỡ package, không ghi đè `mobihet-compat.pth` (của repo `mobihet`).
- Dựng lại env trên máy mới: `conda env create -f environment.yml` (không chạy trên máy đã có `nckh`).
