# Kaggle — account và quy tắc vận hành

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Lỗi kernel đã gặp và cách tránh: [sổ lỗi](notebook-errors.md) (đọc trước khi sinh notebook). Kernel đang chạy: [continue](continue.md).
Lịch sử cấp account/quota 15–17/10: [archive](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/kaggle-accounts-history-2026-10-15-to-17.md).

## Kaggle CLI 2.2.5 — cập nhật 27/10/2026

Theo yêu cầu owner từ phiên RCD: nâng `nckh` từ 2.2.4 lên 2.2.5, chỉ package `kaggle`
thay đổi; `kagglesdk` 0.1.38 và runtime ML giữ nguyên. `pip check` PASS.
Pin tại `environment.yml`; metadata tại [local-runtime](../../knowledge/local-runtime.json).
Helper/skill `.agents` và `.claude` cập nhật đồng thời, mỗi bộ test account/pull **29 PASS**.

CLI mới có streaming/resume/timeout cho output, nhưng một lệnh chỉ lấy một trang listing.
Helper giữ all-page traversal/parallel và kiểm file thiếu. Đã sửa helper bỏ mất phiên bản
kernel: `<owner>/<slug>/7` phải truyền `v7` qua mọi trang SDK; regression hai trang xác nhận.
Dùng version cố định khi thu bằng chứng. `kernels push --no-run` không hủy worker;
giữ GUARD và hướng dẫn stop hiện có. Không chạy train/collector/backup/Git/DVC trong tác vụ này.

Nguồn: [release chính thức](https://github.com/Kaggle/kaggle-cli/releases/tag/v2.2.5),
[changelog](https://github.com/Kaggle/kaggle-cli/blob/v2.2.5/CHANGELOG.md).
Số liệu throughput CLI cũ trong skill là lịch sử, chưa benchmark lại 2.2.5.

## Thiết lập máy hiện tại — 25/10/2026

Đã lưu 16 token trong `~/.kaggle/accounts/`, token 0600, thư mục 0700; mapping email nằm riêng
ở `~/.kaggle/account-roster.json` (0600), không đưa vào repo. SDK introspection xác nhận 16/16 token
active và username khớp tên file. Trạng thái có thể đổi; kiểm lại khi cần.
User sẽ đăng nhập OAuth CLI sau. Không ép tài khoản mặc định; chọn account theo quota và slot GPU live,
chạy qua `kaggle_as.py`, giữ global login đang có. Helper `quota/health/plan` hỗ trợ token-only;
`use/ensure` vẫn cần OAuth snapshot và quy trình chuyển global, không dùng để thay thế lệnh account-scoped.

```bash
source /home/odixe/Tools/miniconda3/etc/profile.d/conda.sh
conda activate nckh
# User tự đăng nhập khi sẵn sàng:
kaggle auth login
# Kiểm quota không đổi global login:
python .agents/skills/kaggle-training-notebook/scripts/kaggle_account.py quota
```

## Account

16 account đã lưu credential (ngoài repo, `~/.kaggle/accounts/`, thư mục 0700, token 0600; không ghi token/email vào docs):
accanniez, anhtruong1909, annhannh331, annhtnt, catbaochau, dinhan25, khanhmay0304, minhtran0601, minhtriethihi, minhtrit06,
nguyenminhtran060113, odixe0502, odixeuit, takids1909, thuyanh316, trietbackup.

| lưu ý | account |
|---|---|
| Không đổi global CLI khi làm việc account khác; user sẽ tự login trên host mới | không ép tài khoản |
| `kaggle datasets create` từng 403 với dataset 2–12 GB (≤17/10). **21/10 06:07Z tạo được** dataset private 27 B (ready, đã xóa) ⇒ dataset nhỏ (nguồn, checkpoint) upload trực tiếp được; dataset GB chưa kiểm lại | odixeuit, minhtrit06 |
| Refresh token OAuth hỏng 20/10; hook 21/10 báo 16/16 account healthy | minhtrit06 |
| **Owner dùng cho dự án khác từ 21/10** (≈11h GPU tới 08Z); owner 21/10 ~18:35Z: **được xếp chấm/train khi có slot** (cache K100 push 18:35Z). Chia quota và 2 slot GPU với dự án kia; không thu hồi token khi chưa pull xo/xu | odixeuit |
| **Owner còn dùng Kaggle cho một dự án khác (21/10)** ⇒ quota hụt bất thường (giảm nhiều hơn kernel mobihet đang chạy + phần giữ chỗ) là do dự án đó, không phải kernel mobihet chạy ngầm/lỗi. Đọc quota live trước mỗi đợt push, không dựa số cũ | mọi account |
| **Dự án kia đang chạy (owner báo 21/10 ~11:50Z):** minhtrit06 ×2 (`rcd-llm-probe-hf` v2 từ ~09:30Z, `rcd-llm-probe-vllm` v3 ~09:37Z), anhtruong1909 ×1, takids1909 ×1 (LLM 7–8B). Kernel mobihet trên các account này vẫn chạy (W&B 11:50Z) nhưng chia quota ⇒ ưu tiên account khác cho đợt mới, theo dõi quota minhtrit06 trước khi xếp zg r35–50 | minhtrit06, anhtruong1909, takids1909 |

## Quy tắc

- Mọi lệnh chạy theo account bằng `kaggle_as.py`, credential chỉ trong process:
  `conda run -n nckh python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py <user> -- kaggle kernels status <user>/<kernel>`.
- Quota: được query mọi account (owner 17/10) bằng `kaggle_account.py quota`; reset hằng tuần thứ Bảy 00:00Z (gần nhất 24/10).
  Chỉ push khi quota live ≥ 1,15 × giờ session; tối đa 2 session GPU cùng lúc mỗi account.
- Dataset source/checkpoint phải PRIVATE, upload lên **account chạy kernel**, verify đủ file/SHA trước khi push.
- Notebook public (`notebooks/`) không chứa key; bản push (`private_push/`, git-ignored, 0600) mang key W&B inline theo quyền owner đã cấp
  (notebook `is_private: true`). Đăng nhập W&B và pin phiên bản: [skill wandb.md](../../.claude/skills/kaggle-training-notebook/references/wandb.md).
- `QUEUED` = chờ GPU, không phải lỗi. `RUNNING`/`COMPLETE`/W&B finished không chứng minh đã train: phải có log/heartbeat và audit artifact.
  Log live: `kaggle kernels logs --follow` (SSE) giới hạn ~35 s; `kernels logs` thường rỗng khi đang chạy.
- Dừng session chỉ làm được trên web (owner bấm **Stop session**); API không hủy được.
- Kaggle native MCP chưa cấu hình; CLI có xác thực là kênh chính.
