"""Lệnh Telegram 2 chiều cho bot báo trộm (chỉ chủ nhà mới được đáp).

Lệnh: /start, /help, /cam (mọi cam), /cam1, /cam2, /cam <tên/số>.
Chạy nền trong watch.py / app_gui.py, không chặn luồng chính.
"""
import threading
import time

import cv2
import requests

import config
import face_engine

_api = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/"
_stop = threading.Event()
_thread = None


def _post(method, **kwargs):
    r = requests.post(_api + method, data=kwargs, timeout=30)
    return r.json()


def _send_text(chat_id, text):
    try:
        _post("sendMessage", chat_id=chat_id, text=text[:4000])
    except Exception:
        pass


def snapshot(rtsp, wait_s=8):
    """Chụp 1 frame luồng phụ, trả bytes jpg hoặc None."""
    cap = face_engine.open_camera(rtsp)
    if not cap.isOpened():
        return None
    img, t0 = None, time.time()
    while time.time() - t0 < wait_s:
        ret, frame, _ = cap.read_new(-1)
        if ret and frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                img = buf.tobytes()
                break
        time.sleep(0.1)
    cap.release()
    return img


def _send_cam(chat_id, cam):
    img = snapshot(cam["rtsp"])
    if img is None:
        _send_text(chat_id, f"[{cam['name']}] Không lấy được hình. Cam offline?")
        return
    try:
        requests.post(_api + "sendPhoto",
                      files={"photo": ("cam.jpg", img, "image/jpeg")},
                      data={"chat_id": chat_id,
                            "caption": time.strftime(f"[{cam['name']}] %H:%M:%S %d/%m")},
                      timeout=30)
    except Exception:
        _send_text(chat_id, f"[{cam['name']}] Gửi ảnh lỗi.")


def _pick_cam(cams, arg):
    arg = (arg or "").strip().lower()
    if arg.isdigit():
        i = int(arg) - 1
        return cams[i] if 0 <= i < len(cams) else None
    for c in cams:
        if arg and arg in c["name"].lower():
            return c
    return None


def _handle(msg, cams):
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if not chat_id or chat_id != str(config.TELEGRAM_CHAT_ID):
        return  # không phải chủ nhà -> im lặng
    text = (msg.get("text") or "").strip()
    names = ", ".join(f"{i+1}.{c['name']}" for i, c in enumerate(cams))

    if text.startswith("/start") or text.startswith("/help"):
        _send_text(chat_id, "Bot canh cam.\n/cam - xem mọi cam\n"
                            f"/cam1, /cam2... hoặc /cam <tên> ({names})")
    elif text.startswith("/cam"):
        arg = text[len("/cam"):].strip()
        if not arg:
            for c in cams:
                _send_cam(chat_id, c)
        else:
            cam = _pick_cam(cams, arg)
            if cam is None:
                _send_text(chat_id, f"Không có cam đó. Chọn: {names}")
            else:
                _send_cam(chat_id, cam)


def _loop(cams):
    offset = 0
    while not _stop.is_set():
        try:
            data = requests.post(_api + "getUpdates",
                                 data={"offset": offset, "timeout": 20},
                                 timeout=40).json()
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                if "message" in upd:
                    _handle(upd["message"], cams)
        except Exception:
            time.sleep(3)


def start(cams):
    """Bật luồng nghe lệnh. Không có token/chat id thì thôi."""
    global _thread
    if _thread is not None or not config.TELEGRAM_BOT_TOKEN:
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(cams,), daemon=True)
    _thread.start()


def stop():
    global _thread
    _stop.set()
    _thread = None
