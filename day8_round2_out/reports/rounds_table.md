# Bảng so sánh các vòng

Tập kiểm thử: 20 ảnh, 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px). Ngưỡng IoU 0.5; P, R, F1 tính tại conf 0.25.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 351 | 0.583 | -0.188 | 1.000 | 0.146 | 0.255 | 0.000 | 0.125 | 0.537 |
| 2 | yolov8n fine-tune vong 1..2 | 24 | 418 | 0.861 | +0.089 | 0.989 | 0.228 | 0.371 | 0.000 | 0.186 | 0.902 |
