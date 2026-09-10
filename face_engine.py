"""Nhận diện khuôn mặt dùng chung cho watch.py (service) và app_gui.py (Qt).

Không cần train: so embedding InsightFace với từng góc đã enroll.
Chữ tiếng Việt trên video được vẽ bằng Pillow (OpenCV không vẽ được dấu).
"""
import glob
import os
import threading
import time

# Giảm delay RTSP: TCP cho ổn định + không đệm + low-delay.
# Phải set trước khi VideoCapture đầu tiên được tạo.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0",
)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
import headpose

NGUOI_LA = "NGƯỜI LẠ"
KHONG_THAY_MAT = "KHÔNG THẤY MẶT"

_app = None
_fonts = {}


def get_app():
    """Singleton FaceAnalysis (model nặng, chỉ load 1 lần)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(config.DET_SIZE, config.DET_SIZE))
    return _app


def load_faces():
    """{tên: (N,512)} - file cũ 1 vector vẫn đọc được."""
    faces = {}
    for f in glob.glob(os.path.join(config.FACES_DIR, "*.npy")):
        name = os.path.splitext(os.path.basename(f))[0]
        arr = np.load(f)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        faces[name] = arr
    return faces


def has_motion(prev_gray, frame, thresh=6.0):
    """Cảnh có chuyển động không? So frame thu nhỏ với lần quét trước.

    Trả (motion, gray_hiện_tại). Rẻ hơn nhận diện ~1000 lần, dùng để
    bỏ qua inference khi cảnh tĩnh.
    """
    small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(small, (160, 120))
    small = cv2.GaussianBlur(small, (9, 9), 0)
    if prev_gray is None:
        return True, small
    return float(cv2.absdiff(prev_gray, small).mean()) > thresh, small


class ThreadedCamera:
    """Đọc cam trên luồng riêng, luôn giữ frame MỚI NHẤT, bỏ frame cũ.

    Dùng thay cv2.VideoCapture để hết delay tích lũy: read() trả về
    frame mới nhất thay vì frame cũ trong buffer.
    """

    def __init__(self, rtsp):
        self.rtsp = rtsp
        self.cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._opened = self.cap.isOpened()
        self._lock = threading.Lock()
        self._frame = None
        self._counter = 0
        self._run = self._opened
        self._thread = None
        if self._opened:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _reconnect(self):
        """Tạo lại kết nối RTSP (cam vừa rớt mạng / reboot)."""
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(2)
        self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _loop(self):
        fails = 0
        while self._run:
            ret, frame = self.cap.read()
            with self._lock:
                if ret:
                    self._frame = frame
                    self._counter += 1
                else:
                    self._frame = None
            if ret:
                fails = 0
            else:
                fails += 1
                if fails > 100:
                    # rớt lâu -> nối lại từ đầu, cam reboot cũng tự bắt lại
                    if self._reconnect():
                        fails = 0
                    else:
                        time.sleep(5)
                elif fails > 30:
                    time.sleep(0.5)

    def isOpened(self):
        return self._opened

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read_new(self, last_counter):
        """Trả (ret, frame, counter). counter không đổi = chưa có hình mới."""
        with self._lock:
            if self._frame is None:
                return False, None, last_counter
            if self._counter == last_counter:
                return True, None, last_counter
            return True, self._frame.copy(), self._counter

    def release(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.cap.release()


def open_camera(rtsp):
    """Mở camera chống delay (thay cv2.VideoCapture)."""
    return ThreadedCamera(rtsp)


def _get_font(size=22):
    if size not in _fonts:
        for p in ("/usr/share/fonts/noto/NotoSans-Regular.ttf",
                  "/usr/share/fonts/TTF/DejaVuSans.ttf"):
            if os.path.exists(p):
                _fonts[size] = ImageFont.truetype(p, size)
                break
        else:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def draw_text_vi(frame, text, pos, color=(0, 255, 0), size=22):
    """Vẽ chữ có dấu lên frame OpenCV (màu BGR, vẽ đè trực tiếp)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    d.text(pos, text, font=_get_font(size),
           fill=(color[2], color[1], color[0]),
           stroke_width=2, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def match_frame(frame, known, thresh, min_ratio=0.0):
    """Trả về (frame_đã_vẽ, [(label, sim, bbox)], n_mặt_xa, n_mặt_mờ).

    Mặt nhỏ hơn min_ratio (đứng xa) hoặc mờ/nhòe (đang đi nhanh) bị bỏ qua
    để tiết kiệm CPU và tránh báo nhầm.
    """
    h, w = frame.shape[:2]
    app = get_app()
    out = []
    n_far = 0
    n_poor = 0
    try:
        faces = app.get(frame)
    except Exception:
        return frame, out, n_far, n_poor
    for fc in faces:
        x1, y1, x2, y2 = map(int, fc.bbox)
        if min_ratio > 0 and (x2 - x1) / max(1, w) < min_ratio:
            n_far += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            continue
        ok_q, _ = headpose.face_quality(frame, fc.bbox)
        if not ok_q:
            n_poor += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)
            continue
        sims = {n: float(np.max(known[n] @ fc.normed_embedding)) for n in known}
        label = max(sims, key=sims.get)
        sim = sims[label]
        if sim <= thresh:
            label = NGUOI_LA
        color = (0, 255, 0) if label != NGUOI_LA else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_text_vi(frame, f"{label} {sim:.2f}", (x1, max(0, y1 - 28)), color)
        out.append((label, sim, (x1, y1, x2, y2)))
    return frame, out, n_far, n_poor
