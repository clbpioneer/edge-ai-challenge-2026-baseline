# ESP32-S3 Edge AI baseline: Detection + Intent

Baseline chạy **hoàn toàn trên ESP32-S3**: phát hiện đồ vật (CNN int8) và phân loại ý định câu hỏi (TextCNN int8)
bằng TFLite Micro, kèm web server để thử trực tiếp từ trình duyệt.

```
Trình duyệt ──(ảnh 320×240 RGB + câu hỏi)──► ESP32-S3 /run
                                              ├─ Detection CNN (grid 15×20, 7 class) → box + NMS
                                              ├─ TextCNN (32 token, 5 intent)
                                              └─ Trả lời theo intent + kết quả detection
Trình duyệt ◄──────────── JSON ───────────────┘   (vẽ bounding box, danh sách đối tượng, intent, câu trả lời)
```

- **Class:** `bottle cup phone book mouse box toy_car`
- **Intent:** `COUNT EXIST LIST LOCATION OUT_OF_SCOPE`
- Repo đã kèm trọng số baseline trong `main/models/`, clone về là build và flash được ngay.
- Camera OV7670 là tùy chọn: có thì web hiện khung camera live và nút chụp ảnh. Camera **chưa** nối vào model,
  phần này để các đội tự phát triển (xem [Camera](#camera-ov7670-tùy-chọn)).

## Phần cứng

- **ESP32-S3 có PSRAM** (bắt buộc: tensor arena detection khoảng 1.5 MB). Đã cấu hình cho module **N16R8 / N8R8 (PSRAM Octal)**.
  Board PSRAM Quad: trong `sdkconfig.defaults` đổi `CONFIG_SPIRAM_MODE_OCT=y` thành `CONFIG_SPIRAM_MODE_QUAD=y`.
- Flash ≥ 4 MB.
- (Tùy chọn) Camera OV7670 không FIFO.

## Build và flash

Cần [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/) ≥ 5.1 (đã thử v5.5).
Các component `esp-tflite-micro`, `esp-nn`, `esp32-camera` được tự tải khi build.

```bash
git clone <repo-url> && cd <repo>
. $IDF_PATH/export.sh
idf.py set-target esp32s3
idf.py menuconfig          # ESP32-S3 Infer → WiFi SSID / password
idf.py -p /dev/ttyACM0 flash monitor     # Windows: -p COMx
```

- Để trống SSID (hoặc kết nối WiFi lỗi) thì ESP tự phát WiFi **`ESP32S3-INFER`**, mật khẩu `12345678`, rồi mở http://192.168.4.1/
- Kết nối WiFi nhà thì mở địa chỉ IP in ra trong log.

Log khởi động đúng sẽ có:
```
infer: detection OK | arena dung ... KB
infer: textcnn OK | arena dung ... KB
infer: vocab 261 tu
```

## Dùng web

Trang có 3 khung trên cùng một hàng:

| Khung | Chức năng |
|---|---|
| **Ảnh gốc** | Tải ảnh lên (định dạng nào cũng được, trình duyệt tự resize về 320×240), nhập câu hỏi, bấm **Run** |
| **Camera** | Hình OV7670 live (nếu có camera), nút Stream / Stop / **Chụp ảnh** (lưu file .jpg về máy) |
| **Kết quả** | Ảnh kèm bounding box, intent, câu trả lời, thời gian chạy, bảng đối tượng và score |

Câu hỏi gõ có dấu hay không dấu đều được, trình duyệt tự bỏ dấu vì vocab của TextCNN không dấu.

## API (cho script chấm hoặc test tự động)

`POST /run`: body = **230400 byte RGB888** (320×240, theo hàng, R G B) **+ câu hỏi ASCII không dấu** (tối đa 512 byte, có thể rỗng).

```json
{"ms_det": 812, "ms_intent": 3,
 "dets": [{"c": "cup", "s": 0.873, "b": [0.12, 0.30, 0.35, 0.71]}],
 "intent": "COUNT", "answer": "Có 1 cốc."}
```
`b` = `[x1, y1, x2, y2]` chuẩn hóa 0..1. Ví dụ Python:

```python
import requests, numpy as np
from PIL import Image
rgb = np.asarray(Image.open("img.png").convert("RGB").resize((320, 240)), np.uint8).tobytes()
print(requests.post("http://192.168.4.1/run", data=rgb + b"Co bao nhieu coc trong anh?").json())
```

Stream camera: `GET http://<ip>:81/stream` (MJPEG).

## Train và thay model của đội

1. Chạy `baseline.ipynb` từ trên xuống (đặt dataset ở thư mục `egde-ai-challenge-2026-lan-2-qualifiers`):
   - Phần 1 Detection: train, export int8 ra `model_int8.tflite` và `det_model.cc/.h`
   - Phần 2 TextCNN: train, export int8 ra `textcnn_int8.tflite`, `textcnn_model.cc/.h` và `vocab.cc/.h`
   - Phần 3 Submission: tạo `submission.csv`
2. Copy 6 file `.cc/.h` vào `main/models/` (hoặc đặt `cc_dir` trong ô export), rồi chạy `idf.py build flash`.

Ràng buộc để chạy được trên ESP (chi tiết trong `main/models/README.md`):
- Detection: input int8 `1×240×320×3` (RGB/255), output NHWC `1×15×20×(5+7)`.
- TextCNN: input `1×32` **int32**, output int8 `1×5`.
- Chỉ dùng op mà TFLite Micro hỗ trợ và đã đăng ký trong `REGISTER_OPS` (`main/model_config.h`).
  Thiếu op thì log báo `Didn't find op for builtin opcode ...`: thêm op vào `REGISTER_OPS`.

> ⚠️ **Đừng export TextCNN qua onnx2tf.** onnx2tf chèn `SELECT`/`CAST int64` khi gặp embedding (TFLite Micro không có),
> và nối sai trục khi độ dài câu = số chiều embedding (= 32). TFLite trên PC vẫn chạy đúng nên rất khó phát hiện.
> Ô export trong notebook dựng graph thẳng bằng TensorFlow và có `assert` kiểm tra danh sách op.

## Cấu trúc

```
baseline.ipynb          train + export int8 + .cc/.h + submission
main/
  main.c                WiFi, HTTP server (/ và /run)
  infer.cc/.h           TFLite Micro: nạp 2 model, chạy detect / intent
  logic.c/.h            hậu xử lý thuần C: decode grid, NMS, tokenize, câu trả lời
  model_config.h        tên mảng model, kích thước arena, ngưỡng conf/NMS, REGISTER_OPS
  cam.c/.h              camera OV7670 + MJPEG stream cổng 81
  index.html            giao diện web (được nhúng vào firmware)
  models/               trọng số dạng mảng C (.cc/.h)
test/test_logic.c       test hậu xử lý trên máy tính
```

Test logic C trên máy tính (không cần board):
```bash
gcc -I main test/test_logic.c main/logic.c -lm -o /tmp/t && /tmp/t
```

## Camera OV7670 (tùy chọn)

| OV7670 | ESP32-S3 | | OV7670 | ESP32-S3 |
|---|---|---|---|---|
| SDA | GPIO16 | | D0..D7 | GPIO4..11 |
| SCL | GPIO17 | | RST | 3.3V |
| XCLK | GPIO12 | | PWDN | GND |
| PCLK | GPIO13 | | 3.3V / GND | 3V3 / GND |
| HREF | GPIO15 | | VSYNC | GPIO14 |

- Đổi chân: sửa `CAM_PIN_*` trong `main/cam.c`. XCLK, CLKRC, chất lượng JPEG: chỉnh trong `menuconfig`.
- Không có camera thì firmware vẫn chạy bình thường, web báo "Không có camera".
- Lúc khởi động có nhiều dòng `I2C transaction unexpected nack`: đó là driver dò các loại cảm biến khác, **không phải lỗi**.
- Camera chạy QVGA 320×240 RGB565, **đúng bằng input của model detection**. Để ghép camera vào model:
  `esp_camera_fb_get()` → đổi RGB565 (big-endian) sang RGB888 → `infer_detect()` / `infer_intent()` / `make_answer()` như trong `run_post` (`main/main.c`).

## Lỗi thường gặp

| Log | Cách xử lý |
|---|---|
| `AllocateTensors loi` + `Didn't find op ...` | Thêm op vào `REGISTER_OPS` (`main/model_config.h`) |
| `khong cap phat duoc ... KB PSRAM` | Board không có PSRAM hoặc PSRAM Quad (xem [Phần cứng](#phần-cứng)) |
| `detection: can input int8 1x240x320x3 ...` | Model detection sai shape hoặc sai số class |
| `Thieu main/models/... ` khi build | Chưa copy đủ 6 file `.cc/.h` vào `main/models/` |

## License

[MIT](LICENSE)
