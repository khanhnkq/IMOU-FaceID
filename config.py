import os

from dotenv import load_dotenv

# Đọc biến môi trường từ file .env (cùng thư mục với file này)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# === ĐỔI CHO PHÙ HỢP VỚI CAM CỦA BẠN ===
# User mặc định của Imou Cue 2 là admin
# Mật khẩu = Safety Code 8 ký tự dưới đít cam, vd L2AC937D
# IP lấy từ Dahua ConfigTool / app Fing / router
RTSP_USER = os.getenv("IMOU_USER", "admin")
RTSP_PASS = os.getenv("IMOU_PASS", "SAFETY_CODE_DOI_O_DAY")
RTSP_IP = os.getenv("IMOU_IP", "192.168.1.2")

# subtype=1 nhẹ hơn (640x360), subtype=0 nét hơn (1080p) nhưng lag hơn
RTSP_URL = os.getenv(
    "IMOU_RTSP",
    f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_IP}:554/cam/realmonitor?channel=1&subtype=1",
)

# Ngưỡng nhận diện (cosine similarity). 0.40 dễ, 0.50 chặt.
# Nếu hay báo nhầm người lạ thành bạn -> tăng lên. Ngược lại -> giảm xuống.
THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.45"))

# Chế độ CHỈ QUÉT KHI Ở GẦN (không cần cảm biến):
# Bỏ qua mặt nhỏ hơn tỉ lệ này so với chiều rộng khung hình (mặt càng to = càng gần).
# 0.12 ~ đứng trong 1-2m với Cue 2. Tăng lên nếu muốn phải dí sát mới quét.
NEAR_MODE = os.getenv("NEAR_MODE", "1") == "1"
NEAR_MIN_RATIO = float(os.getenv("NEAR_MIN_RATIO", "0.12"))

# Cảnh báo Telegram khi thấy NGƯỜI LẠ (để trống = tắt, app vẫn chạy).
# Lấy token: chat với @BotFather -> /newbot. Lấy chat id: chat với @userinfobot.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
# Số giây tối thiểu giữa 2 cảnh báo (chống spam khi người lạ đứng lâu)
ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "120"))

# Thư mục lưu khuôn mặt đã đăng ký
FACES_DIR = os.path.join(os.path.dirname(__file__), "data", "faces")

# Model InsightFace: buffalo_s nhẹ, chạy CPU tốt. Muốn chuẩn hơn đổi sang buffalo_l
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")
# Kích thước ảnh đưa vào model detect (320 nhanh gấp đôi 640, vẫn chuẩn với 1-2 mặt gần)
DET_SIZE = int(os.getenv("DET_SIZE", "320"))
# Quét 1 frame sau mỗi N frame cam (25 = ~1 lần/giây với cam 25fps)
RECOG_EVERY_N_FRAMES = int(os.getenv("FRAME_SKIP", "25"))
# Motion-gate: cảnh tĩnh thì bỏ qua nhận diện (tiết kiệm CPU khi không có gì).
# Ngưỡng chênh lệch sáng trung bình (0-255). Tăng nếu cam nhiễu/hay báo động giả.
MOTION_GATE = os.getenv("MOTION_GATE", "1") == "1"
MOTION_THRESH = float(os.getenv("MOTION_THRESH", "2.5"))
