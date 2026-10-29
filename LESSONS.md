# LESSONS — bài học kỹ thuật (dữ liệu, model, train, đánh giá, code)

Đọc sau [CONTEXT](CONTEXT.md), cùng [bài học vận hành](docs/operations/lessons.md). Bài học kế thừa:
`../mobihet-features/LESSONS.md` (dataset v4), `../mobihet/LESSONS.md` (grep theo chủ đề; FL/KD/đánh giá trên DAGSNet).

## Kế thừa — áp dụng thẳng cho repo này

- Model client chỉ học nhãn của chính nó đạt đỉnh trên test có prior toàn cục sau 1–2 epoch rồi giảm (mobihet 13/09). Ở đây model client
  chấm trên test của chính xe ⇒ prior khớp hơn, nhưng Sybil test 17,7 % vs train 8,0 % vẫn lệch theo thời gian.
- Luôn báo F1 train cạnh F1 test; sụp đột ngột khi đổi cấu hình là dấu hiệu tối ưu hỏng (HGB L2 = 0, mobihet-features 28/10).
- Val "xe chưa thấy" ≈ val cùng xe khi train gộp nhiều xe; không lộ được học thuộc theo xe (mobihet-features 29/10).

## Dữ liệu

- `pyarrow.dataset(<v4>/train)` lỗi "Unable to merge: Field scenario has incompatible types: large_string vs string" (cột trong file là
  `large_string`, partition hive suy `string`) ⇒ đọc từng file bằng `pq.read_table` (`core/data.py`).
- 77 đầu vào = 11 × 7 ⇒ DAGSNet giữ lưới 11 vị trí (`view(B, 11, 7).transpose(1, 2)`).

## Tốc độ (đo 29/10, RTX 3050 4 GiB local)

- DAGSNet đơn, AMP, batch 8.192: full 46k mẫu/s eager, 59k compile; W 0,375: 78k / 113k.
- Grouped (G model/bước, bmm channels-last) compile: ~45k slot/s full width ≈ 77 % model đơn; eager chậm hơn ~2,5×.
  Lần compile đầu ~2 phút. Hiệu suất thực = slot thật / slot (padding): 0,85 ở batch 64 / 8.192 dòng mỗi bước grouped trên phân bố
  client v4 (nhóm theo kích thước); phần mất chủ yếu do client lớn (95k dòng = 1.487 bước tuần tự).
