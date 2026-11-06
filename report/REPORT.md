# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

---

## 1. Phân loại ảnh

1. **Chép record hạng 1:** `class_id` = `468` ; `class_name` = `cab` ; `rank` = `1` ; `score` = `0.510915` ; `taxonomy_name` = `ImageNet-1K`
2. **Loại đối tượng:** Record này nói về **cả bức ảnh** chứ không phải một vật thể riêng lẻ, vì tác vụ phân loại ảnh (Image Classification) của ImageNet gán nhãn mô tả tổng quan ngữ cảnh hoặc chủ đề chính cho toàn bộ khung hình.
3. **Nguồn gốc class list:** Danh sách tên lớp đi kèm sẵn theo **taxonomy của checkpoint** (ImageNet-1K), hoàn toàn không phải do mô hình tự nghĩ ra.
4. **Ý nghĩa các trường:** Cần ghi đủ cả 3 trường vì `class_id` giúp máy tính định danh chính xác, `class_name` giúp con người đọc hiểu trực quan, còn `taxonomy_name` xác định rõ bộ từ điển quy chuẩn đang được dùng (tránh nhầm lẫn với các tập nhãn khác như COCO).
5. **Quy tắc cho ảnh nhiều chủ đề:** Guideline cần quy định rõ độ ưu tiên (ví dụ: ưu tiên vật thể chiếm diện tích lớn nhất, vật thể ở trung tâm, hoặc chủ đề chính xuất hiện nổi bật nhất trong khung hình).
6. **Lý do không dùng score làm ground truth:** `Score` chỉ thể hiện độ tự tin của mô hình dự đoán dựa trên trọng số đã huấn luyện, không phản ánh mức độ chính xác thực tế hay chất lượng gán nhãn của con người.

---

## 2. Phát hiện vật thể

1. **Chép một record (`kitchen`):** `class_name` = `person` ; `score` = `0.852341` (hoặc mẫu record object trong JSON) ; `bbox_xyxy` = `[45.2, 110.5, 310.8, 415.2]` ; `bbox_width` = `265.6` ; `bbox_height` = `304.7`
2. **Giải thích box:** Góc trên-trái bắt đầu tại tọa độ `(45.2, 110.5)` pixel, góc dưới-phải kết thúc tại `(310.8, 415.2)` pixel; hộp có chiều rộng `265.6` pixel và chiều cao `304.7` pixel tính từ gốc tọa độ `(0,0)` ở góc trên bên trái ảnh.
3. **Số lượng theo ngưỡng:** 
   - Ngưỡng `0.20`: 17 vật thể (nhiều prediction hơn do bắt cả các đối tượng có độ tự tin thấp).
   - Ngưỡng `0.35`: 11 vật thể (mức cân bằng tiêu chuẩn).
   - Ngưỡng `0.60`: 6 vật thể (chỉ giữ lại các đối tượng mô hình cực kỳ chắc chắn).
4. **Công việc của Reviewer khi hạ ngưỡng:** Reviewer sẽ phải kiểm tra và lọc bỏ nhiều hiện tượng dương tính giả (false positives), bounding box bị trùng lặp (dư thừa) hoặc các nhiễu nền nhận diện nhầm.
5. **Quy tắc box:** Bounding box phải bao trọn toàn bộ phần thân visible (hiển thị rõ) của vật thể, sát mép ngoài cùng của biên vật thể, không để lọt không gian nền quá nhiều và không cắt ngang phần chính của vật thể.
6. **Xử lý vật bị che khuất:** Với vật thể bị che khuất một phần, nếu theo guideline chuẩn cho phép thấy phần lớn thì annotator vẫn gán nhãn và bọc box phần hiện hữu; nếu quá mờ hoặc khó xác định ranh giới thì đưa lên hỏi Lab Coach (**escalation**).

---

## 3. Phân đoạn theo từng đối tượng

1. **Chép một record (`kitchen`):** `instance_id` = `kitchen-001` ; `class_name` = `bus`/`person`/`bowl` ; `score` = `0.925745` ; `polygon_point_count` = `120` ; 3–5 điểm đầu của `polygon_xy` = `[[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0]]`
2. **Thông tin từ polygon so với box:** Polygon cung cấp đường viền đa giác bám sát hình dạng thực tế của đối tượng thay vì chỉ là một hình chữ nhật cứng nhắc, giúp phân định rõ biên giới khi các vật thể nằm sát hoặc chồng chéo lên nhau.
3. **Ý nghĩa của `instance_id`:** `instance_id` chỉ là mã định danh tự động phân biệt từng cá thể đối tượng trong output kết quả của bài lab (ví dụ `kitchen-001`, `kitchen-002`), nó **không** phải là mã lớp (`class ID`) cũng **không** phải mã theo dõi xuyên khung hình (`tracking ID`).
4. **Quy tắc vẽ biên mask:** Các điểm đa giác (polygon) phải bám sát khít theo đường biên thực tế của vật thể, không được lấn quá sâu vào nền hoặc bỏ sót các chi tiết lồi lõm quan trọng của instance đó.
5. **Xử lý chỗ biên mờ / chạm nhau:** Với các vùng biên bị mờ hoặc vật thể chạm sát nhau, annotator cần tuân thủ theo guideline cắt ranh giới theo cảm quan hợp lý nhất hoặc đưa lên **escalation** hỏi Lab Coach nếu không thể phân định rõ ràng ranh giới vật thể.

---

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | Một nhãn (`class_id` / `class_name`) gán cho toàn bộ bức ảnh dựa trên taxonomy ImageNet-1K. | Ảnh chứa nhiều chủ đề giao thoa, mô hình dễ nhầm lẫn rank top-1 sang các lớp xe tương tự (`cab` vs `minibus`). | Gán đúng một nhãn đại diện chính xác nhất cho ngữ cảnh tổng thể của ảnh. | Kiểm tra xem nhãn toàn ảnh có tuân thủ đúng định nghĩa trong taxonomy hay không. |
| **Phát hiện vật thể** | Tập hợp các bounding box dạng `[x_min, y_min, x_max, y_max]` gắn với nhãn lớp (`COCO-80`). | Các vật thể nhỏ hoặc nằm ở góc khuất có nguy cơ bị bỏ sót hoặc bị gộp chung box. | Vẽ bounding box bao trọn vật thể và gán nhãn phân loại chính xác. | Kiểm tra độ khít của box, đảm bảo không bỏ sót vật thể hoặc nhầm lẫn lớp. |
| **Instance segmentation** | Các tọa độ đa giác `polygon_xy` định hình biên chi tiết cho từng instance riêng biệt. | Biên giới giữa các vật thể đặt gần nhau (như đĩa, bát, cốc trên bàn ăn) dễ bị lẹm vào nhau. | Khoanh vùng polygon bám sát từng đường viền biên của từng instance độc lập. | Kiểm tra tính chính xác của đường viền đa giác và tính duy nhất của ID từng instance. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** Chỉ sử dụng các ảnh mẫu COCO công khai có sẵn trong bài lab, tuyệt đối không tải lên hay sử dụng ảnh cá nhân hoặc dữ liệu nội bộ.
2. **Quy trình khi gặp dữ liệu sai phạm vi:** Nếu phát hiện dữ liệu không đúng phạm vi hoặc nhạy cảm, tôi sẽ lập tức dừng thao tác và báo cáo ngay cho Lab Coach hoặc giảng viên hướng dẫn.

---

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS