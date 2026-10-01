"""Gửi cảnh báo NGƯỜI LẠ về Telegram (kèm ảnh chụp từ cam).

Chưa điền token trong .env thì tự tắt, app vẫn chạy bình thường.
"""
import time

import cv2

import config

try:
    import requests
except ImportError:  # chưa cài requests -> tắt cảnh báo
    requests = None

_last_sent = {}  # cooldown riêng từng cam: {tên_cam: timestamp}


def enabled():
    return (requests is not None
            and bool(getattr(config, "TELEGRAM_BOT_TOKEN", ""))
            and bool(getattr(config, "TELEGRAM_CHAT_ID", "")))


def send_stranger_alert(frame, sim, is_new=False, cam_name="Cam"):
    """Gửi 1 ảnh NGƯỜI LẠ về Telegram. Trả True nếu đã gửi.

    is_new=True (người lạ vừa xuất hiện) thì chỉ cần cách tin trước 20s;
    đứng lì một chỗ thì mỗi ALERT_COOLDOWN giây mới nhắc lại (chống spam).
    Cooldown tính riêng từng cam.
    """
    if not enabled():
        return False
    now = time.time()
    # lần xuất hiện mới báo nhanh, nhưng không dày hơn cooldown (chống chập chờn)
    min_gap = min(20, config.ALERT_COOLDOWN) if is_new else config.ALERT_COOLDOWN
    if now - _last_sent.get(cam_name, 0.0) < min_gap:
        return False
    _last_sent[cam_name] = now
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return False
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("nguoi-la.jpg", buf.tobytes(), "image/jpeg")}
        data = {"chat_id": config.TELEGRAM_CHAT_ID,
                "caption": f"[{cam_name}] NGƯỜI LẠ (độ giống {sim:.2f}) - "
                           f"{time.strftime('%H:%M:%S %d/%m/%Y')}"}
        r = requests.post(url, files=files, data=data, timeout=15)
        return r.ok
    except Exception:
        return False
