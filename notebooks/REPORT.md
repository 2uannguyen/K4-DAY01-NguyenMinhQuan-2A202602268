# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** 3.13.5 / 2.14.0 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Chạy trên môi trường local thay cho Colab; cài thư viện và cấu hình kernel trong `.venv`. Giữ nguyên ba checkpoint và các threshold của notebook. Ô lưu Google Drive được bỏ qua khi chạy local.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  - {
    "sample_id": "traffic",
    "class_id": 468,
    "class_name": "cab",
    "rank": 1,
    "score": 0.510915,
    "taxonomy_name": "ImageNet-1K"
    }
- Record này mô tả toàn ảnh như thế nào?
  - Mô hình xếp lớp `cab` (taxi) ở hạng cao nhất cho toàn bộ ảnh `traffic`. Record không xác định vị trí hoặc số lượng taxi. Quan sát hình cho thấy ảnh có nhiều loại phương tiện, đặc biệt là nhiều xe buýt, nên một nhãn cấp ảnh chưa mô tả đầy đủ nội dung cảnh giao thông.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  - Class list đến từ taxonomy của bộ dữ liệu dùng để huấn luyện và cách ánh xạ lớp được lưu cùng checkpoint. Với `yolo11n-cls.pt`, đó là ImageNet-1K. Mô hình lựa chọn trong danh sách lớp đã có, không tự tạo tên lớp mới khi dự đoán.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` giúp phần mềm xử lý và đối chiếu lớp; `class_name` giúp người đọc hiểu ý nghĩa; `taxonomy_name` xác định hệ thống lớp đang dùng. Cùng một ID có thể mang ý nghĩa khác trong taxonomy khác, nên cần giữ đủ ba trường để tránh diễn giải sai.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  - Guideline cần quy định gán một nhãn hay nhiều nhãn cho mỗi ảnh. Nếu chỉ gán một nhãn, phải xác định cách chọn chủ thể chính, chẳng hạn theo mục tiêu dự án, diện tích hoặc mức độ nổi bật. Trường hợp không xác định được chủ thể chính cần có cách đánh dấu và chuyển reviewer quyết định.
- Vì sao model score không phải ground truth?
  - Score `0.510915` là điểm mô hình dùng để xếp hạng dự đoán. Nó không chứng minh nhãn `cab` đúng và không phải điểm chất lượng nhãn. Ground truth cần được con người xác định từ ảnh theo guideline và kiểm tra chất lượng.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

  - {
    "sample_id": "kitchen",
    "class_name": "bowl",
    "score": 0.719062,
    "bbox_xyxy": [32.65, 342.12, 100.16, 384.93],
    "bbox_width": 67.51,
    "bbox_height": 42.81
    }
- Diễn giải vị trí box bằng lời: Ảnh `kitchen` có kích thước 640 × 427 pixel. Box bao quanh vật chứa được dự đoán là `bowl` ở phía dưới bên trái ảnh. Góc trên trái là `(32.65, 342.12)`, góc dưới phải là `(100.16, 384.93)`. Box rộng `67.51` pixel và cao `42.81` pixel; gốc tọa độ nằm ở góc trên trái ảnh.
- So sánh số prediction ở hai threshold:  Ví dụ khi hạ threshold từ `0.60` xuống `0.35`, số prediction tăng từ 6 lên 11, gồm thêm ba prediction `bowl` và hai prediction `cup`. JSON detection được lưu ở threshold `0.35`.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

  - Threshold thấp giữ lại nhiều ứng viên hơn, tạo cơ hội phát hiện thêm vật thể nhưng cũng có thể giữ thêm dự đoán sai. Reviewer phải kiểm tra nhiều box hơn về lớp, vị trí, sự trùng lặp và tính hợp lệ. Threshold cao giảm số prediction cần xem nhưng có thể bỏ mất vật thể thật có score thấp.
  - Chưa có ground truth đối chiếu nên không thể kết luận recall hoặc precision đã tăng bao nhiêu. Threshold chỉ lọc prediction, không quyết định vật thể nào cần được gán ground truth.
- Đề xuất một quy tắc box chặt: Với quy ước gán nhãn phần nhìn thấy, mỗi vật thể có một box riêng; bốn cạnh box bám sát các điểm ngoài cùng của phần vật thể nhìn thấy, hạn chế khoảng nền thừa và không cắt mất phần thuộc vật thể. Box phải nằm trong ảnh và có chiều rộng, chiều cao dương.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

  - Guideline cần quy định chỉ bao phần nhìn thấy hay ước lượng cả phần bị che, mức độ nhìn thấy tối thiểu để gán nhãn và cách đánh dấu che khuất/cắt mép. Trong ảnh `kitchen`, prediction `person` sát mép trái chỉ bao một phần cánh tay/bàn tay nhìn thấy. Cần xác định trường hợp này có đủ điều kiện gán nhãn `person` hay không; nếu guideline chưa rõ thì chuyển reviewer hoặc Lab Coach quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  - {
    "sample_id": "kitchen",
    "instance_id": "kitchen-002",
    "class_name": "bowl",
    "score": 0.735744,
    "polygon_point_count": 67,
    "polygon_xy": [
    [53.0, 344.0],
    [52.0, 345.0],
    [50.0, 345.0],
    [49.0, 346.0],
    [48.0, 346.0]
    ]
    }
- Polygon bổ sung chi tiết gì so với box?
  - Box chỉ thể hiện vùng chữ nhật bao quanh vật thể và thường chứa cả nền. Polygon mô tả đường biên chi tiết hơn, giúp phân biệt vùng thuộc vật thể với vùng xung quanh. Với chiếc bát, polygon có thể bám theo đường cong của miệng và thân thay vì bao bằng hình chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` phân biệt từng đối tượng trong output. Ví dụ, `kitchen-002` và `kitchen-003` đều thuộc lớp `bowl` nhưng là hai instance riêng. ID này không phải `class_id`, cũng không phải tracking ID để theo dõi một vật thể qua nhiều khung hình.
- Đề xuất một quy tắc biên mask: Mask bám theo biên phần vật thể nhìn thấy, loại bỏ nền và phần thuộc vật thể khác. Hai vật thể cùng lớp phải được tách thành hai instance. Không tự vẽ tiếp phần bị che nếu guideline chưa cho phép; polygon cần tạo vùng hợp lệ, không tự giao nhau.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định cách xử lý biên không rõ, hai vật thể tiếp xúc, vùng bị che và phần vật thể bị chia thành các vùng rời nhau. Ví dụ, hình segmentation cho thấy mask `dining table` bao cả một phần khăn trải trên bàn. Cần quyết định khăn là vật thể riêng hay được tính vào vùng bàn theo quy ước dự án. Nếu không có quy định, annotator đánh dấu ca mơ hồ và chuyển reviewer quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                     | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                   | Annotator làm gì?                                                                                              | Reviewer xem gì?                                                                                   |
| --------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | Nhãn cấp ảnh theo taxonomy; một hoặc nhiều nhãn tùy guideline.  | Ảnh`traffic` có nhiều xe buýt và ô tô nhưng lớp hạng 1 là `cab`; việc chọn một nhãn đại diện cho cả ảnh còn mơ hồ. | Chọn nhãn theo quy tắc xác định chủ thể chính; không chép lớp hạng 1 của model.                    | Kiểm tra nhãn có phù hợp nội dung ảnh, taxonomy và quy tắc ảnh nhiều chủ thể không.   |
| Phát hiện vật thể | Một lớp và một box`[x_min, y_min, x_max, y_max]` cho mỗi object. | Prediction`person` ở mép trái ảnh `kitchen` chỉ bao phần cánh tay/bàn tay nhìn thấy.                                            | Áp dụng quy tắc vật thể cắt mép và mức độ nhìn thấy tối thiểu; chuyển ca chưa rõ cho reviewer. | Kiểm tra điều kiện gán nhãn, độ chặt box, sai lớp, trùng box và vật thể bị bỏ sót. |
| Instance segmentation | Một lớp và một mask/polygon riêng cho mỗi instance.               | Mask`dining table` bao một phần khăn trên bàn; ranh giới giữa bàn và vật đặt trên bàn cần làm rõ.                          | Tách vùng theo guideline, chỉnh biên hoặc đánh dấu vùng mơ hồ.                                        | Kiểm tra mask có lấn nền/vật khác, thiếu phần vật thể hoặc gộp nhiều instance không.  |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng ba ảnh COCO công khai đã được notebook cố định và kiểm tra checksum, giữ thông tin nguồn trong `IMAGE_ATTRIBUTION.md`. Không đưa ảnh cá nhân, dữ liệu khách hàng hoặc dữ liệu nội bộ lên repository công khai.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach hoặc giảng viên qua kênh hỗ trợ chính thức để được hướng dẫn.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
