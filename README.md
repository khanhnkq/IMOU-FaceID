# Imou FaceID đa cam - nhìn vào cam là biết TÔI / NGƯỜI LẠ

Không cần train. Model InsightFace đã train sẵn, bạn chỉ cần enroll mặt.

Cam đang chạy: **Hành lang** (Cue 2) + **Trong nhà** (Ranger 2). Thêm cam mới thì
điền `cams.env` (CAM2_*, CAM3_*) theo mẫu — code tự nhận, không cần sửa.

## 1. Cài đặt

```bash
cd ~/Codes/imou-faceid
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## 2. Cấu hình camera

Điền vào file `.env` (tự nạp bởi `config.py`):

- `IMOU_USER`: mặc định `admin`
- `IMOU_PASS`: Safety Code 8 ký tự dưới đít cam
- `IMOU_IP`: IP cam (quét được `192.168.1.2`, lấy từ Dahua ConfigTool / Fing nếu đổi mạng)
- Hoặc set thẳng `IMOU_RTSP="rtsp://admin:PASS@IP:554/cam/realmonitor?channel=1&subtype=1"`

Test bằng VLC trước: Media > Open Network Stream > dán link RTSP.

## 3. Đăng ký mặt (nhiều góc, kiểu Face ID)

Mở GUI → nút `Đăng ký mặt mới`, xoay đầu theo hướng dẫn, máy tự chụp.
File lưu tại `data/faces/<tên>.npy`. Thêm người khác thì đăng ký thêm tên khác.

## 4. Chạy FaceID

Bản GUI native theo màu Caelestia (đăng ký + xem live + chỉnh ngưỡng):

```bash
.venv/bin/python app_gui.py
```

Bản service canh 24/7 (không cửa sổ, chỉ báo Telegram):

```bash
.venv/bin/python watch.py
```

Bật/tắt nhanh service bằng phím `SUPER + F9` (hoặc chạy `./toggle-faceid.sh`).
Log service nằm ở `data/faceid.log`.

Lệnh Telegram trên bot báo trộm (chỉ chủ nhà): `/cam` xem mọi cam,
`/cam1`, `/cam2` hoặc `/cam <tên>` xem 1 cam.

Đăng ký mặt cộng dồn: enroll cùng tên sẽ **thêm góc mới** vào profile có sẵn
(tối đa 60 góc) thay vì ghi đè — đứng cam nào enroll thì mở hộp chọn cam đó.

- Hiện `TÔI: <tên> - Chào bạn!` = là bạn
- Hiện `NGƯỜI LẠ` = không khớp ai → gửi ảnh về Telegram

## 5. Báo về Telegram khi đi xa (không cần public port)

1. Telegram chat với `@BotFather` → `/newbot` → đặt tên → nhận token
2. Chat với `@userinfobot` → lấy chat id của bạn
3. Điền vào `.env`:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABC...
   TELEGRAM_CHAT_ID=123456789
   ALERT_COOLDOWN=120
   ```
4. Chạy app như bình thường. Thấy `NGƯỜI LẠ` là ảnh chụp từ cam gửi thẳng về điện thoại (tối đa 1 tin / 2 phút chống spam). Đi đâu cũng nhận được vì chỉ cần mạng ra.

## Cấu trúc

```
imou-faceid/
  config.py        # cấu hình + nạp .env
  face_engine.py   # nhận diện + vẽ chữ tiếng Việt dùng chung
  headpose.py      # enroll kiểu Face ID (đo hướng đầu, tự chụp)
  notify.py        # cảnh báo Telegram
  app_gui.py       # GUI Qt theo màu Caelestia (live + enroll + ngưỡng)
  watch.py         # service canh 24/7, chỉ báo Telegram
  toggle-faceid.sh # bật/tắt service (SUPER + F9)
  requirements.txt
  .env             # IP + Safety Code + token (không commit)
  data/faces/*.npy
```
