# Bài học vận hành MobiHet

> Kế thừa từ repo `mobihet` / `mobihet-features` (29/10/2026): mục lịch sử là bằng chứng của các run cũ, không phải trạng thái repo này.

Đọc cùng [LESSONS kỹ thuật](../../LESSONS.md) sau [CONTEXT](../../CONTEXT.md). Lỗi kernel cụ thể (triệu chứng → cách tránh) ở
[sổ lỗi kernel](notebook-errors.md); quy trình account/quota ở [kaggle](kaggle.md), Git/DVC ở [backup (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).
Bài học ở đây là quy tắc rút ra khi chạy thí nghiệm và hạ tầng; bài học về phương pháp/train/đánh giá ghi ở LESSONS.

## Thiết lập host

- 26/10: nhận diện máy từ cả OS nền và desktop session: `/etc/os-release` báo Ubuntu nhưng
  session là Lubuntu/LXQt. RAM vật lý và swap phải đo riêng; không kế thừa hạn mức WSL cũ.
  Số đo và đường dẫn hiện tại chỉ giữ tại [local-env](../../knowledge/local-env.md).
- 25/10: lỗi DNS/NVIDIA-SMI trong sandbox không chứng minh mạng hoặc driver host hỏng. Đối chiếu terminal
  host với quyền phiên; sau khi owner bật quyền, Git clone và CUDA hoạt động.
- Đích OneDrive hiện là rclone mount. Kiểm mount trước push để tránh ghi nhầm thư mục local bên dưới;
  kiểm cloud bằng rclone độc lập với VFS cache. [Remote và bằng chứng (repo mobihet-features)](https://github.com/odixe06/mobihet-features/blob/main/docs/operations/backup.md).

## Kaggle, notebook và đường pipeline

- **20/10 final startup crash:** test `--smoke --final` vẫn đọc pilot fixture, không chứng minh đường production final-only.
  Gaussian legacy còn hash cứng pilot manifest ⇒ r24 zc/zi crash trước round1. Test bắt buộc: không tồn tại pilot,
  Gaussian đọc final manifest; đối chiếu context pilot với legacy. Notebook phải lan truyền exit code subprocess,
  không biến `rc=1, epochs=0` thành Kaggle COMPLETE. Source đã push giữ nguyên; sửa ở tag mới.
- **20/10 upload source:** `datasets create` có thể exit1/JSON parse error dù server đã tạo dataset ready.
  Đọc status, verify manifest names/sizes trước retry; không upload/push trùng từ exit code đơn lẻ.
- **Sửa 20/10: CLI hiện tại đọc được log live bằng `kaggle kernels logs <owner/slug> --follow` (SSE).** Đã đọc W&B auth và
  `Prepared real pilot … old cut: 0` khi r24 vẫn RUNNING; lệnh không `--follow` có thể vẫn rỗng. Dùng account-scoped, đọc có timeout;
  exit 124 do hết thời gian đọc không phải kernel ERROR. Các ghi chú cũ "chỉ có W&B khi đang chạy" không còn đúng ở môi trường này.
  [Bằng chứng startup](https://github.com/odixe06/mobihet/blob/main/docs/archive/operations/final-r24-2026-10-20/final-startup-2026-10-20.json). Log/status chưa thay terminal artifact audit.
- Fixture được fingerprint phải chính là fixture driver đọc. `--final` trước 20/10 chọn dữ liệu final ở Run nhưng main hash pilot; chưa có run final production. Dùng chung `fixture_path`, và evaluator từ chối manifest khác training. Smoke dùng pilot không phát hiện được sai đường dẫn production.
- Exact-resume fixture phải cover state của cả local/control và FL arm. CrossClientRank pooled replay PASS
  đã bỏ sót CE empty-byte state, khiến production verifier dừng dù r3 hợp lệ;
  [repro và recovery](notebook-errors.md) giữ nguyên weights/source, không cần SGD lại.

- **29/10: `kaggle datasets files` trả tối đa 200 file/trang**, kể cả với `--page-size 1000`, và in dòng `Next Page Token = …` lẫn vào đầu ra
  `--csv`. Đối chiếu dataset nhiều file phải lặp `--page-token` tới khi hết token, rồi so tập tên với thư mục local (v3: 823/823).

## Collect, poll và máy local

- Máy local ngủ (WSL) làm poll nền dừng, Kaggle vẫn chạy. Sau khi thức đừng để watcher mở collector cho mọi session cùng
  lúc (mỗi audit đỉnh ~4 GB): chia nhóm session rời nhau cho 2–4 tiến trình `collect_fd.py --only`, pull ~20 s/session.
- Collector có thể thoát 0 mà chưa pull nếu thiếu `PUSHED`. Kiểm `AUDITED.json` và completion cụ thể; chỉ dựng lại marker sau khi
  xác nhận đúng kernel từ status account-scoped, rồi vẫn đối chiếu executed source trước audit (ya-v2, 20/10).
- **Không tạm dừng (SIGSTOP) tiến trình pull; giới hạn audit song song theo RAM còn trống.** 22/10: dừng pull để ưu tiên kernel khác làm đứt
  stream (`ChunkedEncodingError`); cùng lúc audit lớn (~2,6–4 GB mỗi audit 100 client) với tác vụ khác của owner đẩy RAM trống còn ~600 MiB
  và bộ canh RAM dừng hai collector. Ưu tiên bằng thứ tự hàng đợi, tối đa 2 audit cùng lúc trên máy 10 GB; pull nối tiếp được từ `.part`.

## Git, DVC và lưu trữ cloud

- SHA của `.git/index` khác chưa có nghĩa nội dung staged đổi. Kiểm staged tree và HEAD riêng; ghi cả byte identity và content identity,
  không khôi phục index chỉ để ép SHA trùng snapshot cũ (20/10).
- 30/10: resume state của phương pháp per-client phải ghi **theo rank** (mỗi rank giữ bank client riêng); bản cũ chỉ rank 0 ghi ⇒
  resume sai lặng lẽ. Bank W = 1 ≈ 4,2 GB/rank, ghi atomic giữ 2 bản ⇒ vượt ~20 GB `/kaggle/working`: chỉ ghi khi dừng budget,
  xóa bản copy vào sau khi nạp. Quyết định dừng budget phải đồng thuận giữa các rank (all_reduce), không thì treo collective.
- 30/10: owner cấm trailer `Co-Authored-By:` trong commit (skill repo-backup); quy tắc này thắng reminder attribution của harness.
  Commit trước 30/10 ~02:00 còn trailer này — không sửa lịch sử (force-push đã bị từ chối).
