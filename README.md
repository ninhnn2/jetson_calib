# camcal: calib camera cho Jetson

Bộ công cụ calib camera chạy ngay trên Jetson (JetPack 6: Python 3.10, OpenCV 4.8, NumPy 1.21),
không cần cài thêm gì. Trên máy khác chỉ cần Python 3.8+, OpenCV 4.7+ và NumPy; Matplotlib để vẽ
biểu đồ trong báo cáo.

Kết quả dùng cho 3D object detection, SLAM/VIO và ADAS: thông số trong (intrinsics) có sai số đi
kèm, thông số ngoài (extrinsics) của camera trong hệ toạ độ xe, bảng remap để khử méo.

```
python3 -m camcal <lệnh> -h
```

| Lệnh | Việc |
|---|---|
| `board` | Tạo bảng ChArUco (hoặc bàn cờ) thành PDF đúng kích thước thật, kèm JSON mô tả bảng |
| `capture` | Chụp bảng. Ảnh tự được lấy khi bảng đứng yên và ở vị trí mới |
| `preview` | Xem buổi chụp headless từ máy khác qua ssh |
| `monitor` | App trên Mac theo dõi buổi chụp: số ảnh đạt, hạng, việc còn thiếu, độ phẳng tờ giấy |
| `calibrate` | Tính intrinsics, tự chọn mô hình méo, xuất JSON, YAML và `report.html` |
| `validate` | Kiểm tra một calibration có còn đúng không bằng ảnh nó chưa thấy |
| `undistort` | Khử méo ảnh, xuất camera pinhole tương ứng và bảng remap |
| `extrinsics` | Vị trí và hướng camera trên xe từ một bảng đặt trên mặt đất |
| `budget` | ADAS: sai số của khoảng cách đo được ở từng tầm xa, và phần nào do đâu |
| `ground` | ADAS: đổi điểm ảnh sang điểm trên mặt đường (hệ toạ độ xe) và ngược lại |
| `bev` | ADAS: ảnh nhìn từ trên xuống (IPM) kèm bảng remap |
| `transform` | Chuyển calibration sang ảnh đã xoay, scale, resize hoặc crop |
| `synth` | Sinh ảnh bảng từ một camera biết trước (ground truth) để thử |

## Ảnh mẫu

`samples/csi0_b/`: 42 ảnh 1920x1080 chụp bảng A3 bằng camera IMX219 trên Jetson (16/11/2026), là bộ
ảnh tạo ra `results/baseline/`. Tờ giấy lúc đó dán trên tường không phẳng nên kết quả là hạng C;
dùng để chạy thử công cụ:

```
python3 -m camcal calibrate samples/csi0_b --square-mm 29.133
python3 -m camcal monitor samples/csi0_b --square-mm 29.133
```

## Quy trình

### 1. In bảng

Bảng có sẵn trong `boards/`:

| File | Giấy | Ô | Dùng cho |
|---|---|---|---|
| `board_charuco_A4_7x10_25mm.pdf` | A4 | 7x10, ô 25 mm | Camera nhìn gần (dưới khoảng 1 m) |
| `board_charuco_A3_9x12_30mm.pdf` | A3 | 9x12, ô 30 mm | Nên dùng: nhiều góc hơn, chụp được xa hơn |
| `board_ground_A1_5x7_114mm.pdf` | A1 | 5x7, ô 114 mm | Bảng mặt đất cho extrinsics |
| `board_ground_A0_5x7_164mm.pdf` | A0 | 5x7, ô 164 mm | Bảng mặt đất, camera cao hoặc nhìn xa |

Bảng mặt đất khi chỉ in được A3: ghép nhiều tờ, mỗi tờ 2x3 ô 130 mm, các tờ nối nhau đúng theo
cạnh ô. `board_ground_3x2_sheets_A3_6x6_130mm.pdf` (6 tờ, ghép thành 780x780 mm) và
`board_ground_3x1_sheets_A3_6x3_130mm.pdf` (3 tờ). Tạo cỡ khác: `python3 -m camcal board --sheets 3x2
--paper A3 -o boards`. Một tờ A3 đơn nằm trên đất thì camera trên xe không đọc được (mô phỏng với
camera cao 1,3 m: không đọc được ở mọi khoảng cách).

Tạo bảng khác: `python3 -m camcal board --paper A2 -o boards` (thêm `--ground` cho bảng mặt đất,
`--screen 2560x1440 --ppi 109` để hiện bảng trên màn hình phẳng thay vì in).

Khi in:

- In 100%, tắt "fit to page". Giấy mờ (matt), không bóng.
- Dán phẳng lên vật cứng: kính, mica, nhôm, foam board dày. Bảng cong 1 mm là sai số thấy được.
- Dưới bảng có một thước 100 mm. Đo thước đó bằng thước kẹp. Nếu ra 99,6 mm thì ô thật là
  25 x 0,996 = 24,90 mm, và truyền `--square-mm 24.9` cho `calibrate`.

Kích thước ô chỉ ảnh hưởng tới khoảng cách (tvec, extrinsics, stereo baseline). Tiêu cự, tâm ảnh và
hệ số méo không phụ thuộc vào nó.

### 2. Chụp

Calib đúng hình học mà ứng dụng dùng: cùng kích thước ảnh, cùng sensor mode, cùng góc xoay.
Với camera CSI trên board (ucpeer-detect đang dùng `rotate = 180`):

```
cd ~/camera-calib
python3 -m camcal capture csi --rotate 180 --size 1920x1080 \
    --board boards/board_charuco_A3_9x12_30mm.json -o ~/calib/csi0
```

Chỉ một chương trình được mở camera CSI. Dừng ucpeer-detect trước (`./run.sh --stop`).

Nguồn ảnh khác: `v4l2:/dev/video1` (USB, autofocus luôn bị tắt), `serve://127.0.0.1` (đúng ảnh mà
`ucpeer-detect --serve` đang xử lý, cổng 9000), số `0`, URL RTSP hoặc file video.

Board không có màn hình thì lệnh tự chạy headless. Cách tiện nhất là chạy app `monitor` trên Mac:
nó bật capture trên board, hiện camera trực tiếp và tính lại calib mỗi khi có ảnh mới.

```
cd ~/workspaces/camera-calib
.venv/bin/python -m camcal monitor machineai@192.168.1.240:calib/csi0 -i ~/.ssh/id_ed25519_machineai \
    --square-mm 29.133 --start --rotate 180 --board boards/board_charuco_A3_9x12_30mm.json
```

App hiện:

- Camera trực tiếp, kèm gợi ý vùng còn thiếu của capture.
- Số ảnh: dùng được, bị loại, không thấy bảng.
- Hạng hiện tại, từng tiêu chí so với mức cần cho hạng A, và việc cần làm tiếp. Khi đủ thì hiện
  "ĐỦ ẢNH".
- Lỗi từng ảnh, độ phủ khung hình, và bản đồ lỗi theo chỗ trên tờ giấy. Tờ giấy phẳng thì lỗi đều
  khắp tờ; chỗ nào đậm hẳn là tờ giấy không phẳng ở chỗ đó.

Ảnh được rsync qua ssh về `~/.cache/camcal/` trên Mac và mọi tính toán chạy trên Mac; board chỉ
chạy capture. Không mở cổng nào. Đóng cửa sổ thì capture trên board dừng (Ctrl-C). Bỏ `--start`
để chỉ theo dõi một buổi chụp đang chạy. Với `--start`, ảnh chỉ được lấy khi khung hình gần như
đứng yên (`--steady 5`, dưới 5 px/s) vì IMX219 là rolling shutter.

Chỉ xem ảnh camera: `python3 -m camcal preview machineai@192.168.1.240:calib/csi0 -i ~/.ssh/id_ed25519_machineai`.

Cách đưa bảng:

- 20 đến 30 ảnh tốt là đủ (capture dừng ở 30). Thứ quyết định là chất lượng từng ảnh: bảng
  phẳng, ảnh nét, phủ mép, có nghiêng. Chụp thêm chỉ giảm sai số chậm (theo 1/√số ảnh), và không
  bù được bảng cong. Dưới 15 ảnh công cụ không gợi ý calib.
- Phủ hết khung hình, cả bốn góc và bốn cạnh. Cho bảng tràn ra ngoài mép ảnh: ChArUco đọc được
  cả khi bị che một phần, và méo ống kính nằm ở mép.
- Hơn một phần ba số ảnh nghiêng trên 15 độ, theo cả hai trục. Toàn ảnh chụp thẳng thì tiêu cự
  không xác định được.
- Gần và xa. Giữ yên bảng mỗi lần: ảnh chỉ được lấy khi bảng đứng yên.
- Đủ sáng để thời gian phơi ngắn. IMX219 là rolling shutter, bảng di chuyển sẽ bị méo.
- Dòng gợi ý trên màn hình nói vùng nào còn thiếu.

Có cửa sổ: phím cách để chụp tay, `d` bỏ ảnh vừa chụp, `q` kết thúc. Chạy lại `capture` trên cùng
thư mục để chụp thêm.

### 3. Calib

```
python3 -m camcal calibrate ~/calib/csi0 --square-mm 29.9
```

Kết quả trong `~/calib/csi0/result/`:

- `calibration.json`: K, hệ số méo, sai số từng tham số, chất lượng, từng ảnh
- `opencv.yaml`, `camera_info.yaml` (ROS), `kalibr.yaml`
- `report.html`: bản đồ phủ, sai số theo vùng ảnh, từng ảnh, xếp hạng

Những việc công cụ làm ngoài một lần gọi `calibrateCamera`:

- **Chọn mô hình bằng ảnh để riêng.** Bốn mô hình (`pinhole-radtan4`, `pinhole-radtan`,
  `pinhole-rational`, `fisheye`) được fit trên 4/5 số ảnh và chấm trên 1/5 còn lại. Mô hình đơn
  giản nhất trong số tốt ngang nhau được chọn, nên không bị overfit. Ép mô hình: `--model fisheye`.
- **Loại nhiễu.** Góc lệch quá 4 lần trung vị và ảnh lệch quá 3 lần ảnh trung vị bị loại, có ghi lý do.
- **Sai số của từng tham số** (± trong kết quả là 1σ). Fisheye dùng bootstrap.
- **Bản đồ phủ** và tỉ lệ ảnh nghiêng.
- **Kiểm tra hai nửa bảng.** Camera được fit lại bốn lần, mỗi lần chỉ dùng một nửa bảng (trái, phải,
  trên, dưới). Bảng phẳng thì các nửa cho cùng một camera; bảng cong hoặc in lệch thì mỗi nửa một
  kết quả, và sai số ± của phép fit không thấy được điều này. Phần chênh vượt quá nhiễu được ghi
  vào `uncertainty` (dòng "carry"): đó là sai số nên dùng khi tính toán. Chụp thêm ảnh không giảm
  được nó.
- **Mô hình có gập trong khung hình không.** Đa thức méo có thể quay ngược ở ngoài vùng có góc;
  khi đó khử méo ở góc ảnh sai mà không có dấu hiệu gì. Nếu gập, hạng bị giới hạn ở C.
- **Thử tư thế còn lại của bảng phẳng.** Bảng phẳng nhìn gần thẳng có hai tư thế gần giống nhau;
  chọn nhầm một ảnh là phép fit kẹt ở nghiệm sai (đo được: 0,65 px thay vì 0,14 px, tâm ảnh lệch
  10 px; `cv2.calibrateCamera` cũng kẹt như vậy). Công cụ thử lại tư thế kia cho từng ảnh.
- **Truy vết.** `calibration.json` ghi số ảnh và SHA-256 của bộ ảnh (`dataset`), phiên bản công cụ
  và OpenCV, nguồn ảnh và sensor mode.
- **Xếp hạng** A, B, C, D kèm việc cần làm để lên hạng.

| Hạng | Sai số trên ảnh để riêng | Phủ ảnh | Phủ mép | Số ảnh | Tiêu cự biết tới | Ảnh nghiêng |
|---|---|---|---|---|---|---|
| A | ≤ 0,25 px | ≥ 85% | ≥ 60% | ≥ 20 | 0,1% | ≥ 30% |
| B | ≤ 0,5 px | ≥ 70% | ≥ 35% | ≥ 15 | 0,3% | ≥ 20% |
| C | ≤ 1,0 px | ≥ 40% | | ≥ 8 | 1% | |

Thêm một điều kiện: fx giữa các nửa bảng lệch không quá 0,3% (A), 1% (B), 3% (C).

A: SLAM, VIO, hình học tầm xa. B: 3D object detection, ADAS L2. C: chỉ để thử. Ngưỡng là kinh
nghiệm kỹ thuật cho camera 1 đến 2 megapixel, không phải chứng nhận.

Số ảnh chỉ là mức sàn. Đo trên camera dựng giống IMX219 của board (fx 1603, k1 −0,363, 1920x1080),
bảng A3, ảnh rải đều khung hình, 25 lần thử mỗi dòng; sai số khung hình là độ lệch giữa điểm ảnh
và tia sáng so với camera thật (trung vị):

| Số ảnh | Bảng phẳng (nhiễu góc 0,1 px) | Bảng kém (nhiễu góc 0,3 px) |
|---|---|---|
| 10 | 1,09 px | 3,25 px |
| 20 | 0,70 px | 2,12 px |
| 40 | 0,41 px | 1,24 px |
| 60 | 0,41 px | 1,22 px |

Ảnh kém thì σ tiêu cự lớn và hạng tự đòi thêm ảnh; 60 ảnh trên bảng kém vẫn tệ hơn 10 ảnh trên bảng
phẳng.

### 4. Kiểm tra lại

Chụp một buổi mới (10 đến 20 ảnh), không dùng lại ảnh đã calib:

```
python3 -m camcal validate ~/calib/csi0/result/calibration.json ~/calib/csi0_check --overlay ~/calib/check_overlay
```

Mã thoát 1 nếu calibration không còn đúng. Làm lại sau khi chạm vào ống kính, sau rung mạnh,
sau thay đổi nhiệt lớn, và định kỳ.

### 5. Dùng trong ứng dụng

```
python3 -m camcal undistort ~/calib/csi0/result/calibration.json anh.png -o out --maps
python3 -m camcal transform ~/calib/csi0/result/calibration.json scale=0.5 -o ~/calib/csi0_960
python3 -m camcal transform ~/calib/csi0/result/calibration.json rotate=180 -o ~/calib/csi0_rot0
```

`undistort --maps` ghi `rectify_maps.npz` (`new_K`, `map_x`, `map_y` cho `cv2.remap`) và
`rectified.json` (camera pinhole của ảnh đã khử méo, điểm ảnh luôn vuông để vật không bị kéo dãn). `transform` cho calibration của cùng camera
khi ảnh được xoay, scale, resize hoặc crop, không cần chụp lại. Đổi sensor mode hoặc lấy nét lại
thì phải calib lại.

### 6. Extrinsics: camera trên xe

Hệ toạ độ xe theo ISO 8855: X về trước, Y sang trái, Z lên trên, gốc trên mặt đất (thường dưới
giữa trục sau).

1. Đặt bảng mặt đất ngửa lên, trên nền phẳng, trong tầm nhìn camera.
2. Đo `--origin X,Y` (mét): góc ngoài của bảng cạnh marker 0 (góc trên bên trái của trang in).
3. Đo `--yaw` (độ): hướng trục x của bảng (dọc cạnh ngắn) so với X của xe, ngược chiều kim đồng hồ
   nhìn từ trên xuống.
4. `--board-z`: độ dày tấm lót (mét).

```
python3 -m camcal extrinsics ~/calib/csi0/result/calibration.json ground_*.png \
    --board boards/board_ground_A1_5x7_114mm.json --square-mm 113.8 --origin 2.50,0.40 --yaw 0
```

Kết quả (`extrinsics.json`): vị trí, chiều cao, pitch, yaw, roll của camera, ma trận
`T_vehicle_camera` và homography của mặt đất. Độ chính xác bằng đúng độ chính xác của thước đo: lệch 5 mm và 0,3 độ ở bảng là lệch
5 mm và 0,3 độ ở camera.

### 7. Tính toán cho ADAS

Sai số khoảng cách ở từng tầm xa (dùng `--extrinsics extrinsics.json`, hoặc cho độ cao và góc chúi
của camera):

```
python3 -m camcal budget ~/calib/baseline/calibration.json --height 1.2 --pitch 3
```

Bảng in ra hai cách đo:

- **Theo mặt đường:** khoảng cách của điểm ảnh nơi vật chạm đường. Sai số tăng theo bình phương
  khoảng cách, vì góc dưới đường chân trời nhỏ dần. Bảng tách riêng phần do tiêu cự, tâm ảnh, góc
  chúi, độ cao và detector, và thêm cột khi xe nhún 0,5° lúc chạy.
- **Theo bề rộng vật** (mặc định xe 1,8 m): sai số tương đối bằng sai số tiêu cự ở mọi tầm.

Đầu vào là `uncertainty` của calibration, sai số của extrinsics, và các tuỳ chọn `--sigma-...`.
Mỗi phần là 1σ, cộng theo căn tổng bình phương. Giả thiết: đường phẳng.

```
python3 -m camcal ground calibration.json --extrinsics extrinsics.json --pixel 960,700 --point 10,0
python3 -m camcal bev calibration.json anh.png --extrinsics extrinsics.json -o bev
```

`ground` đổi điểm ảnh sang (X, Y) trên mặt đường và ngược lại. `bev` ghi ảnh nhìn từ trên xuống,
`bev_maps.npz` cho `cv2.remap` và `bev.json` mô tả tỉ lệ.

## Số đo

Đo ngày 14/11/2026 trên Jetson Orin Nano Super (L4T R36.4.7, OpenCV 4.8.0, NumPy 1.21.5).

| Việc | Kết quả |
|---|---|
| Camera CSI IMX219, 1920x1080, xoay 180 | 177 khung trong 6,0 s (29,5 fps) |
| Tìm bảng trong một khung 1920x1080 | trung vị 76 ms |
| Độ chính xác góc ChArUco trên ảnh dựng, camera pinhole | 0,049 đến 0,071 px rms, không góc nào lệch quá 0,5 px trong khoảng 2600 góc |
| Calib 45 ảnh 1920x1080 (1892 góc), đủ 4 mô hình | 29 s |
| Cùng dữ liệu với bộ giải của OpenCV trên board | 734 s, cho cùng kết quả |
| Kết quả so với camera thật của bộ ảnh dựng | fx 1449,89 ± 0,17 (thật 1450,00); cx 975,22 ± 0,16 (thật 975,36); 0,044 px rms |
| Validate 45 ảnh | 5,1 s |

Hai điều rút ra từ OpenCV 4.8 của JetPack, đã xử lý trong công cụ:

- Bước nội suy góc ChArUco của OpenCV 4.8 trả góc lệch nửa pixel, và trên ảnh nghiêng trả sai
  173 trong 2088 góc (rms 1,2 px, tệ nhất 7,3 px). Công cụ tự định vị góc từ marker, cho cùng độ
  chính xác trên OpenCV 4.8 và 5.0.
- OpenCV của JetPack build không có LAPACK nên `calibrateCamera` mất 9 đến 18 s cho mỗi lần fit
  45 ảnh. Công cụ dùng bộ giải Levenberg-Marquardt riêng (khử pose bằng Schur complement), được
  test đối chiếu với OpenCV: cùng rms, cùng camera (lệch dưới 0,002 px), cùng σ.

## Giới hạn

- Chưa có trong bộ này: stereo, camera với IMU, camera với LiDAR, mô hình rolling shutter, calib
  lại online khi xe chạy (theo dõi góc chúi bằng điểm tụ của vạch làn), kiểm tra theo nhiệt độ,
  đồng bộ thời gian giữa các cảm biến.
- `budget`, `ground`, `bev` giả thiết đường phẳng. Đường dốc hoặc xe nhún làm sai khoảng cách theo
  mặt đường nhiều hơn mọi sai số calib ở tầm xa. `kalibr.yaml` là đầu vào để làm camera với IMU bằng Kalibr.
- IMX219 là rolling shutter. Cho SLAM tốc độ cao và L4 nên dùng cảm biến global shutter.
- Xếp hạng là hướng dẫn kỹ thuật, không phải chứng nhận. Chương trình L4 có quy trình riêng:
  phòng target, đo đạc khảo sát, kiểm tra theo nhiệt độ, giám sát online.
- Extrinsics từ bảng mặt đất là giá trị khởi đầu cho L4, đủ cho L2 nếu đo cẩn thận.
- Calibration đúng cho một lần lấy nét, một sensor mode. Khoá nét ống kính (keo hoặc ốc hãm)
  trước khi calib.
- `serve://` mới được test theo giao thức, chưa chạy với ucpeer-detect thật.

## Test

```
python3 -m unittest tests.test_camcal
```

34 test (191 s trên board) trên ảnh dựng từ camera biết trước: bảng in đúng kích thước, góc tìm đúng chỗ, bộ giải
khớp OpenCV, calib tìm lại đúng camera, validate phát hiện ống kính bị xê dịch, transform, khử
méo, extrinsics, luồng raw của ucpeer-detect, dòng lệnh, app monitor (kể cả phát hiện tờ giấy
phồng 1,5 mm), kiểm tra hai nửa bảng, tư thế còn lại của bảng phẳng, mô hình gập, và các lệnh ADAS
(đối chiếu với công thức giải tích).
