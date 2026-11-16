# Baseline calib camera CSI (bản tạm)

Bản calib làm gốc cho tính toán, dùng tới khi có bản chụp trên bảng phẳng. Hạng C.

- Ngày 16/11/2026, từ buổi chụp `~/calib/csi0_b` (42 ảnh).
- Bảng ChArUco A3 9x12, ô 29,133 mm (thước 100 mm in ra đo được 97,11 mm).
- Camera: IMX219 ống kính 160°, cảm biến CSI 0, sensor mode 2: 1920x1080 cắt giữa cảm biến (không
  thu nhỏ), 30 fps, xoay 180. Đúng như ucpeer-detect đang chạy.

Đổi sensor mode, đổi góc xoay hoặc chỉnh nét ống kính thì baseline không còn đúng.

## Thông số

| | Giá trị | Khoảng không chắc chắn |
|---|---|---|
| fx, fy | 1603,29 / 1603,38 px | fx 1579 đến 1613 |
| cx, cy | 1006,66 / 473,30 px | cx 1002 đến 1011, cy 465 đến 476 |
| k1, k2 | −0,36303 / 0,16401 | |
| p1, p2 | 0,00036 / 0,00065 | |
| Góc nhìn | 69,4° × 38,7°, chéo 79,3° | |
| Ở tâm ảnh | 28,0 px mỗi độ (0,036° mỗi px) | |
| Méo | ở góc ảnh lệch 272 px so với camera pinhole, ở nửa đường ra góc lệch 32 px | |
| Lỗi | 0,64 px trên ảnh để riêng | |

Khoảng không chắc chắn là kết quả khi calib bằng từng phần của tờ giấy (cả tờ, từng nửa, bỏ 2 cột
ngoài mỗi bên), vì tờ giấy dán trên tường gạch không phẳng. Giá trị ± trong `calibration.json`
(fx ± 2,4 px) nhỏ hơn thực tế. Khi tính lan truyền sai số, dùng **fx ± 1,5%, cx và cy ± 8 px**
(ghi trong `calibration.json`, mục `baseline.uncertainty`).

Dùng được cho:
- Khử méo ảnh.
- Đổi điểm ảnh sang hướng tia.
- Khoảng cách thô: sai khoảng 1,5%, tức 30 cm ở 20 m.
- Phát triển và thử pipeline.

Chưa dùng được cho đo khoảng cách chính xác, SLAM, ADAS.

## File

| File | Dùng cho |
|---|---|
| `calibration.json` | camcal: thông số, sai số, ghi chú baseline |
| `opencv.yaml` | `cv::FileStorage`: `camera_matrix`, `distortion_coefficients` |
| `camera_info.yaml` | ROS |
| `kalibr.yaml` | Kalibr (camera với IMU) |
| `report.html` | Báo cáo của lần calib |
| `undistort/rectify_maps.npz` | `new_K`, `map_x`, `map_y` cho `cv2.remap`: ảnh khử méo 1920x1080, chỉ giữ vùng ống kính thấy, điểm ảnh vuông |
| `undistort/rectified.json` | Camera pinhole của ảnh đã khử méo: f 1531,70, cx 1029,23, cy 467,62, không còn méo |
| `1280x720/` | Cùng camera khi ảnh 1920x1080 được thu nhỏ về 1280x720 |

## Dùng trong code

Python: đổi điểm ảnh (u, v) sang hướng tia (x, y, 1) trong hệ camera (x sang phải, y xuống dưới,
z ra trước):

```python
import json, numpy as np, cv2

c = json.load(open("/home/machineai/calib/baseline/calibration.json"))
i, d = c["intrinsics"], c["distortion"]
K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
D = np.array([d["k1"], d["k2"], d["p1"], d["p2"]])
crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
x, y = cv2.undistortPointsIter(np.array([[[u, v]]], np.float64), K, D, None, None, crit).reshape(2)
```

Dùng `undistortPointsIter` như trên. `cv2.undistortPoints` mặc định chỉ lặp 5 lần; với ống kính này
nó lệch tới 0,117 px ở góc ảnh (đo trên board, OpenCV 4.8).

Khử méo cả ảnh:

```python
m = np.load("/home/machineai/calib/baseline/undistort/rectify_maps.npz")
flat = cv2.remap(img, m["map_x"], m["map_y"], cv2.INTER_LINEAR)  # camera m["new_K"], không méo
```

C++: đọc `opencv.yaml` bằng `cv::FileStorage`, rồi gọi `cv::undistortPoints` bản có tham số
`cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-12)`.

## Sai số khoảng cách với baseline này

```
python3 -m camcal budget ~/calib/baseline/calibration.json --height 1.2 --pitch 3
```

Thay 1,2 m và 3° bằng độ cao và góc chúi thật của camera trên xe. Với ví dụ này, đo theo mặt
đường giữ được sai số dưới 5% tới 9 m và dưới 10% tới 18 m; ở 20 m sai 2,1 m, trong đó 1,7 m là do
tâm ảnh ± 8 px. Đo theo bề rộng xe 1,8 m sai 2,0% ở 20 m.

## Kiểm tra và thay

Baseline còn đúng không: chụp 10 đến 20 ảnh bảng vào một thư mục mới, rồi

```
python3 -m camcal validate ~/calib/baseline/calibration.json <thư mục ảnh>
```

Khi có bản chụp trên bảng phẳng đạt hạng A hoặc B: đổi tên thư mục này thành `baseline_csi0_b`, rồi
đặt bản mới vào `~/calib/baseline`.
