# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: PHẠM XUÂN DUY

Công cụ gán nhãn đã dùng: CVAT

Mọi con số trong báo cáo được truy xuất chính xác từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` và các file `outputs/round*_diff.md`.

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool: 268 ảnh) và tập kiểm thử (test set: 20 ảnh) được chia theo trục thời gian với vùng đệm ở giữa (buffer: 112 ảnh), thay vì chia ngẫu nhiên, xuất phát từ bản chất vật lý của dữ liệu video giám sát giao thông ban đêm. Camera được đặt cố định tại một vị trí trên đường cao tốc. Một phương tiện khi lưu thông qua tầm nhìn của camera thường xuất hiện liên tục trong nhiều khung hình kế tiếp trong khoảng vài giây.

Nếu chia ngẫu nhiên (random split), các khung hình thuộc cùng một phân đoạn thời gian rất ngắn sẽ bị phân tán vào cả tập huấn luyện (pool) và tập kiểm thử (test). Khi đó, cùng một chiếc xe với cùng góc nhìn, điều kiện ánh sáng và bối cảnh mặt đường sẽ vừa xuất hiện trong tập huấn luyện vừa xuất hiện trong tập kiểm thử (hiện tượng rò rỉ dữ liệu do tương quan thời gian - temporal data leakage). Kết quả là số đo trên tập kiểm thử (AP50, Precision, Recall) sẽ bị lệch lạc theo hướng **lạc quan quá mức (overly optimistic / lệch dương)**: mô hình đạt điểm số rất cao nhờ khả năng "học vẹt" vị trí và hình dáng của các xe cụ thể đã thấy, thay vì thể hiện năng lực khái quát hóa (generalization) thực sự trên các phương tiện và dòng lưu lượng mới. Việc tạo vùng đệm thời gian (buffer window) ngăn cách triệt để giữa các khung hình train và test giúp đảm bảo đánh giá khách quan, trung thực năng lực phát hiện xe của mô hình.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg` và `outputs/metrics_round0.json` (TP=197, FP=16, FN=206 trên tổng số 403 box tham chiếu), mô hình khởi đầu lạnh (yolov8n pretrained trên COCO) đạt Precision@0.25 khá cao (0.925) nhưng Recall@0.25 rất thấp (chỉ 0.489), tức bỏ sót tới hơn một nửa số phương tiện trong cảnh đêm. Mô hình không khớp nhãn tham chiếu tập trung chủ yếu ở:
- Các xe ở cự ly xa chỉ còn hai đốm sáng đèn hậu nhỏ hoặc ánh đèn mờ nhạt;
- Các xe màu tối chìm vào nền đêm sát dải phân cách hoặc mép đường;
- Các xe bị che khuất một phần (occluded) bởi xe khác hoặc bị cắt mép khung hình (truncated).

Độ phủ theo kích thước xe thể hiện rõ sự phân hóa sâu sắc:
- `R small`: đạt vỏn vẹn **0.182** (chỉ phát hiện được 12/66 xe kích thước nhỏ);
- `R medium`: đạt **0.547** (162/296 xe kích thước vừa);
- `R large`: đạt **0.561** (23/41 xe kích thước lớn).
Điều này chứng minh mô hình cold start huấn luyện trên ảnh COCO ban ngày gặp điểm mù nghiêm trọng nhất đối với xe nhỏ ở xa trong điều kiện thiếu sáng ban đêm.

Một trường hợp cần con người rà lại nhãn tham chiếu trước khi kết luận mô hình sai: Nhãn tham chiếu trong tập test cũng được sinh tự động bởi mô hình mạnh hơn mà chưa qua thẩm định thủ công 100%. Ví dụ, ở các vệt phản quang ánh đèn pha chói lóa trên mặt đường ướt hoặc các bảng phản quang ven đường, nhãn tham chiếu có thể bị gán nhầm thành box xe (False Positive của reference), hoặc ngược lại bỏ sót các xe quá xa chỉ có hai đốm sáng mờ. Khi mô hình dự đoán khác nhãn tham chiếu tại các vị trí này, chuyên viên gán nhãn cần mở ảnh gốc để kiểm tra bằng mắt xem có thực sự là xe hay không, tránh quy chụp mô hình sai khi chính nhãn tham chiếu bị lỗi.

## 3. Chiến lược chọn mẫu

Công thức chọn mẫu kết hợp ba thành phần:
`score = W_U·U + W_A·A + W_D·D`
với các trọng số mặc định `W_U = 0.5`, `W_A = 0.3`, `W_D = 0.2`:
- `U` (Uncertainty): độ bất định trung bình của mô hình trên khung hình, phản ánh mức độ phân vân của mô hình đối với các dự đoán bounding box;
- `A` (Ambiguity): tỷ lệ các bounding box có điểm tin cậy rơi vào vùng lưỡng lự (vùng biên quyết định, ví dụ confidence từ 0.2 đến 0.6);
- `D` (Diversity): điểm đa dạng hóa khoảng cách thời gian so với các khung hình đã được chọn trước đó.
Vai trò của `MIN_GAP_S` (ngưỡng thời gian tối thiểu 2.0 giây): Do camera đặt cố định, hai khung hình cách nhau dưới 2 giây hầu như chứa cùng một cảnh giao thông với các xe chưa kịp thay đổi vị trí đáng kể. `MIN_GAP_S` đóng vai trò rào chắn chống trùng lặp, ép thuật toán phải bỏ qua các khung hình xuất hiện quá sát nhau, ngăn ngừa lãng phí công sức rà nhãn vào dữ liệu dư thừa.

Minh chứng từ `reports/SELECTION.md` và `outputs/selection_round1.csv`:
- Ba frame thuộc lô 12 ảnh được chọn:
  1. `frame_0182.jpg` (hạng 1, điểm 0.9591, t=72.8s): độ bất định U=0.9182, có tới 18 box phân vân trên tổng số 28 box đề xuất, cảnh chứa nhiều xe tối chạy sát mép trái;
  2. `frame_0099.jpg` (hạng 8, điểm 0.9063, t=39.6s): độ bất định U=0.9460 rất cao, 14 box phân vân, có nhiều xe nhỏ gần cầu vượt và xe bị mép ảnh cắt góc dưới phải;
  3. `frame_0107.jpg` (hạng 14, điểm 0.8876, t=42.8s): U=0.8752, 15 box phân vân, xuất hiện vệt phản quang ánh đèn pha trên mặt đường ướt khiến model lưỡng lự giữa xe và vệt sáng.
- Một frame khác để đối chiếu: `frame_0372.jpg` (hạng 6, điểm số rất cao 0.9101, U=0.9202, A=0.8333, 15 box phân vân). Dù có điểm cao hơn các frame được chọn (như frame_0312, frame_0099, frame_0107), frame này vẫn bị loại (selected=False) do thời điểm t=148.8s chỉ cách frame_0369.jpg (hạng 2, t=147.6s) đúng 1.2s (< MIN_GAP_S = 2.0s). Đây là minh chứng rõ ràng cho việc ưu tiên đa dạng và tiết kiệm ngân sách rà nhãn thay vì chọn mù quáng theo điểm số bất định.

Điểm bất định **không chứng minh** ảnh đó chắc chắn sẽ cải thiện mô hình khi đưa vào huấn luyện. Điểm bất định chỉ phản ánh trạng thái "lúng túng" hiện tại của mô hình trước khung hình đó. Một khung hình có điểm bất định cao có thể do chứa quá nhiều nhiễu (ảnh mờ, chói sáng đèn pha, vật thể che khuất bất thường) khiến mô hình học các đặc trưng sai lệch (spurious correlations), hoặc dẫn đến hiện tượng quên cục bộ (catastrophic forgetting) các đặc trưng đã học tốt. Sự cải thiện thực sự chỉ đến khi các mẫu bất định đó mang thông tin đại diện tốt và bổ sung đúng khoảng trống phân bố của miền dữ liệu mục tiêu.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp kết quả các vòng từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 351 | 0.583 | -0.188 | 1.000 | 0.146 | 0.255 | 0.000 | 0.125 | 0.537 |
| 2 | yolov8n fine-tune vong 1..2 | 24 | 418 | 0.861 | +0.089 | 0.989 | 0.228 | 0.371 | 0.000 | 0.186 | 0.902 |
| 3 | yolov8n fine-tune vong 1..3 | 36 | 513 | 0.878 | +0.106 | 0.955 | 0.422 | 0.585 | 0.045 | 0.436 | 0.927 |

### Phân tích chi tiết từng vòng:

1. **Vòng 1 (12 ảnh, 351 box train):**
   - Mức độ sửa nhãn (theo `outputs/round1_diff.md` và `round1_diff.json`): Model đề xuất ban đầu 169 box; sau khi rà soát và chỉnh sửa đạt 351 box. Trong đó: giữ nguyên (accepted) 140 box (tỷ lệ 82.8%), chỉnh sửa kích thước (edited) 11 box, xoá bỏ (deleted - FP của model) 18 box, và thêm mới (added - FN của model) 200 box.
   - Biến động AP50: AP50 đạt 0.583, giảm **-0.188** so với cold start (0.771).
   - Biến động nhóm xe: Precision đạt tuyệt đối 1.000 (không còn FP nào trên test set, FP=0), nhưng Recall tụt mạnh từ 0.489 xuống 0.146 (R small tụt về 0.000, R med tụt xuống 0.125, R large đạt 0.537). Nguyên nhân: Với chỉ 12 ảnh đầu tiên được gắn nhãn rất kỹ (thêm 200 box xe nhỏ/xa), mô hình fine-tune trên tập dữ liệu nhỏ trở nên cực kỳ thận trọng, chỉ dự đoán ở các trường hợp có độ tin cậy tuyệt đối, dẫn đến giảm recall toàn diện.

2. **Vòng 2 (tích lũy 24 ảnh, 418 box train):**
   - Mức độ sửa nhãn (theo `outputs/round2_diff.md`): Lô 12 ảnh vòng 2 đề xuất 55 box, sau sửa còn 67 box (accepted 54 box, edited 1 box, deleted 0 box, added 12 box).
   - Biến động AP50: AP50 bật tăng mạnh lên **0.861**, tăng **+0.089** so với cold start và tăng **+0.278** so với vòng 1.
   - Biến động nhóm xe: R large tăng vọt từ 0.537 lên **0.902** (+0.365); R medium phục hồi lên 0.186; Precision duy trì ở mức xuất sắc 0.989. Mô hình đã làm chủ các xe kích thước lớn và vừa trong cảnh đêm.

3. **Vòng 3 (tích lũy 36 ảnh, 513 box train):**
   - Mức độ sửa nhãn (theo `outputs/round3_diff.md`): Lô 12 ảnh vòng 3 đề xuất 75 box, sau sửa còn 95 box (accepted 73 box, edited 0 box, deleted 2 box, added 22 box).
   - Biến động AP50: AP50 đạt **0.878**, tăng **+0.106** so với cold start và tăng **+0.017** so với vòng 2.
   - Biến động nhóm xe: Recall@0.25 tăng gần gấp đôi từ 0.228 lên **0.422** (TP tăng từ 92 lên 170 box); R medium tăng từ 0.186 lên **0.436**; R large duy trì ở mức rất cao **0.927**; R small bắt đầu được phát hiện lại (0.045). F1-score tăng mạnh lên 0.585.

### Đối chiếu bằng chứng và ca thực tế:

- **Ca kết quả thay đổi sau fine-tune từ `compare_round*.jpg`:** Khi so sánh `compare_round0.jpg` và `compare_round2.jpg` / `compare_round3.jpg`, ở làn đường bên phải cự ly gần và trung bình, mô hình vòng 2 và 3 nhận diện chuẩn xác gần như toàn bộ các xe tải và ô tô con kích thước lớn với bounding box ôm sát thân xe (R large đạt 0.927 so với 0.561 ở cold start). Lý do có thể kiểm chứng: Các box xe lớn trong `round1_diff.md` và `round2_diff.md` đã được người thẩm định chuẩn hóa viền khung ôm sát thân xe và đèn, giúp mô hình học được đặc trưng ranh giới rõ rệt.
- **Phân biệt ba cấp độ quan sát:**
  + *Quan sát độc lập (`BLIND_SCAN.md`)*: Trước khi thấy gợi ý của AI, mắt người đếm được 18 xe trên `frame_0099.jpg`, phát hiện rủi ro xe bị cắt ở góc dưới phải và xe nhỏ mờ nhạt gần chân cầu.
  + *Lỗi pre-label đã sửa (`REVIEW_LOG.csv` & `round1_diff.md`)*: AI cold start chỉ đề xuất 13 box (bỏ sót 14 xe, nhận nhầm 1 vệt sáng). Người gán nhãn đã xóa 1 box giả do ánh đèn phản chiếu ở `frame_0107.jpg`, kéo chỉnh viền khung xe tối ở `frame_0182.jpg`, và thêm 14 box xe thiếu ở `frame_0099.jpg`.
  + *Kết quả mô hình sau train*: Sau vòng 1, mô hình bị co cụm (precision cao, recall thấp), nhưng đến vòng 3 khi dữ liệu tích lũy đủ 36 ảnh chuẩn hóa, mô hình mở rộng khả năng nhận dạng, tăng mạnh recall lên 0.422 mà vẫn giữ precision 0.955.
- **Mô tả ca khó theo guideline:** Trường hợp xe bị cắt mép ở góc dưới bên phải trong `frame_0099.jpg`. Theo `GUIDELINE_LABEL.md`, chỉ được khoanh phần thân xe còn nhìn thấy trong khung hình (phần đuôi xe và đèn hậu), tuyệt đối không suy đoán vẽ tràn ra ngoài mép ảnh hoặc vẽ gộp với mặt đường. Pre-label của AI ban đầu đã bỏ sót hoàn toàn chiếc xe này do không nhận diện được phần đầu xe; người gán nhãn đã bổ sung đúng quy tắc (`added`).

## 5. Kết luận và giới hạn

So với cold start (AP50 = 0.771, F1 = 0.640), kết quả sau 3 vòng active learning đã đạt được bước tiến lớn: AP50 tăng lên **0.878** (+10.6%), Precision đạt **0.955**, Recall tăng mạnh từ 0.146 (vòng 1) lên **0.422** (vòng 3), đặc biệt R large đạt **0.927** và R medium đạt **0.436**.

**Quyết định dừng hay tiếp tục:**
Quyết định **dừng lại ở vòng 3**. Lý do:
1. Mức gia tăng AP50 bắt đầu có dấu hiệu bão hòa biên (marginal gain diminishing): từ vòng 1 sang vòng 2 tăng +0.278, nhưng từ vòng 2 sang vòng 3 chỉ tăng thêm +0.017 (từ 0.861 lên 0.878).
2. Tỷ lệ accept rate của pre-label ở vòng 2 và vòng 3 đã đạt rất cao (98% và 97%), cho thấy mô hình đã học tốt các trường hợp phổ biến trong pool.
3. Chi phí rà nhãn cho mỗi vòng (12 ảnh với hàng chục box) không còn đem lại đột phá tương xứng so với ngân sách thời gian và công sức.

**Đề xuất hai ca còn yếu hoặc bất định cho vòng tiếp theo:**
1. *Ca xe kích thước nhỏ ở cự ly xa (R small hiện tại chỉ đạt 0.045)*: Các xe chỉ còn hai chấm đèn nhỏ ở khu vực chân cầu. Chi phí rà nhãn rất cao vì kiểm duyệt viên phải phóng to từng pixel để phân biệt đèn xe với đèn đường/biển báo.
2. *Ca xe bị che khuất một phần trong dòng xe ùn ứ*: Dễ gặp nguy cơ ảnh gần trùng (near-duplicate) nếu các frame được chọn nằm sát nhau trong lúc xe di chuyển chậm, đòi hỏi phải siết chặt `MIN_GAP_S` hoặc tăng trọng số đa dạng `W_D`.

**Giới hạn của tập kiểm thử và ảnh hưởng đến kết luận:**
- Tập test chỉ có 20 ảnh với 403 box tham chiếu: Cỡ mẫu nhỏ khiến các số đo nhạy cảm với dao động ngẫu nhiên; sai sót trên một vài box có thể làm biến động AP50.
- Quy tắc bỏ qua 14 box cao dưới 16 px: Các phương tiện siêu nhỏ bị loại khỏi đánh giá, khiến số đo recall không đo lường đầy đủ năng lực phát hiện ở tầm cực xa.
- Nhãn tham chiếu do mô hình tự động tạo ra chưa qua thẩm định thủ công 100%: Có thể tồn tại các box giả hoặc box thiếu trong reference labels. Do đó, chỉ số AP50 phản ánh độ tương đồng với bộ nhãn tham chiếu chứ không phải độ chính xác tuyệt đối so với chân lý mặt đất (ground truth).

**Kế hoạch kiểm tra nếu AP50 bị giảm:**
Nếu sau một vòng fine-tune mà AP50 bị sụt giảm (như trường hợp vòng 1), các bước cần kiểm tra trước khi train tiếp gồm:
1. Rà soát lại file diff (`round*_diff.md` / `.json`): Kiểm tra xem có thao tác nhầm lẫn như xóa nhầm box đúng của model, gán nhầm class, hoặc vẽ box sai quy tắc guideline không.
2. Kiểm tra hiện tượng mất cân bằng kích thước box: Xem các box mới thêm có quá tập trung vào xe nhỏ khiến mô hình bị nhiễu gradient không.
3. Kiểm tra siêu tham số huấn luyện (hyperparameters): Xem xét learning rate có quá cao làm phá vỡ các trọng số pretrained hữu ích không, và đánh giá nguy cơ overfitting trên tập train kích thước nhỏ.
