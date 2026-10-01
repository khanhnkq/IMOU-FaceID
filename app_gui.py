"""App FaceID đa cam - GUI native Qt theo màu Caelestia.

Chạy:
    python app_gui.py

- Mỗi cam 1 khung video + badge TÔI (xanh) / NGƯỜI LẠ (đỏ)
- Danh sách người đã enroll, nút Enroll (chọn cam) có hướng dẫn từng góc
- Thanh trượt ngưỡng nhận diện (áp dụng mọi cam), nhật ký nhận diện
"""
import os
import re
import sys
import threading

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import config
import face_engine
import headpose
import notify
import telecmd

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------- Theme Caelestia ----------
def load_caelestia_colors():
    colors = {}
    path = os.path.expanduser("~/.config/hypr/scheme/current.lua")
    try:
        with open(path, encoding="utf-8") as f:
            colors = dict(re.findall(r'(\w+)\s*=\s*"([0-9a-fA-F]{6})"', f.read()))
    except OSError:
        pass

    def c(*keys, default="ffffff"):
        for k in keys:
            if k in colors:
                return "#" + colors[k]
        return "#" + default

    return {
        "bg": c("base", "background", default="0c0e12"),
        "surface": c("surfaceContainer", "surface0", default="161a1f"),
        "surface2": c("surfaceContainerHigh", "surface1", default="1c2026"),
        "text": c("text", "onBackground", default="e1e5ef"),
        "muted": c("subtext1", "onSurfaceVariant", default="a7abb4"),
        "primary": c("primary", default="afc9eb"),
        "on_primary": c("onPrimary", default="28425e"),
        "error": c("error", default="fa746f"),
        "success": c("success", default="b5ccba"),
    }


def build_qss(t):
    return f"""
    QWidget {{ background: {t['bg']}; color: {t['text']}; font-size: 14px; }}
    QFrame#card {{ background: {t['surface']}; border-radius: 12px; }}
    QLabel#badge-ok {{ background: {t['success']}; color: #10210f; font-size: 22px;
        font-weight: bold; border-radius: 12px; padding: 12px; }}
    QLabel#badge-stranger {{ background: {t['error']}; color: #2b0505; font-size: 22px;
        font-weight: bold; border-radius: 12px; padding: 12px; }}
    QLabel#badge-idle {{ background: {t['surface2']}; color: {t['muted']}; font-size: 22px;
        font-weight: bold; border-radius: 12px; padding: 12px; }}
    QLabel#badge-near {{ background: {t['primary']}; color: {t['on_primary']}; font-size: 22px;
        font-weight: bold; border-radius: 12px; padding: 12px; }}
    QLabel#video {{ background: #000; border-radius: 12px; }}
    QPushButton {{ background: {t['surface2']}; border-radius: 10px; padding: 10px 16px; }}
    QPushButton:hover {{ background: {t['primary']}; color: {t['on_primary']}; }}
    QPushButton#primary {{ background: {t['primary']}; color: {t['on_primary']}; font-weight: bold; }}
    QListWidget {{ background: {t['surface']}; border-radius: 12px; padding: 6px; }}
    QTextEdit {{ background: {t['surface']}; border-radius: 12px; padding: 6px; }}
    QLineEdit {{ background: {t['surface2']}; border-radius: 8px; padding: 8px; }}
    QComboBox {{ background: {t['surface2']}; border-radius: 8px; padding: 8px; }}
    QSlider::groove:horizontal {{ background: {t['surface2']}; height: 6px; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {t['primary']}; width: 18px; margin: -6px 0; border-radius: 9px; }}
    """


# ---------- Worker đọc camera (1 worker / 1 cam) ----------
class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.Signal(str, np.ndarray)  # (tên cam, frame)
    status_ready = QtCore.Signal(str, list, int, int)  # (tên, [(label, sim)], xa, mờ)

    def __init__(self, cam, thresh_getter, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.thresh_getter = thresh_getter
        self._run = True
        self.known = face_engine.load_faces()

    def reload_faces(self):
        self.known = face_engine.load_faces()

    def stop(self):
        self._run = False
        self.wait(3000)

    def run(self):
        name = self.cam["name"]
        cap = face_engine.open_camera(self.cam["rtsp"])
        if not cap.isOpened():
            self.status_ready.emit(name, [("LỖI CAM", 0.0)], 0, 0)
            return
        n = 0
        last_counter = -1
        prev_gray = None
        while self._run:
            ret, frame, last_counter = cap.read_new(last_counter)
            if not ret:
                self.msleep(200)
                continue
            if frame is None:
                # chưa có hình mới, nghỉ nhẹ tránh quay tít CPU
                self.msleep(15)
                continue
            n += 1
            if n % config.RECOG_EVERY_N_FRAMES == 0 and self.known:
                motion, prev_gray = face_engine.has_motion(
                    prev_gray, frame, config.MOTION_THRESH)
                if config.MOTION_GATE and not motion:
                    self.frame_ready.emit(name, frame)
                    continue
                frame, res, n_far, n_poor = face_engine.match_frame(
                    frame, self.known, self.thresh_getter(), self.cam["near"])
                self.status_ready.emit(name, [(label, sim) for label, sim, _ in res],
                                       n_far, n_poor)
            self.frame_ready.emit(name, frame)
        cap.release()


# ---------- Dialog Enroll có hướng dẫn góc ----------
class EnrollDialog(QtWidgets.QDialog):
    enrolled = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cams = {c["name"]: c["rtsp"] for c in config.CAMERAS}
        self.rtsp = next(iter(self.cams.values()), "")
        self.embs = []
        self.guide = headpose.PoseGuide(shots_per_target=2)
        self._hint = "đang tìm mặt..."
        self._last_det = 0
        self.setWindowTitle("Đăng ký khuôn mặt kiểu Face ID")
        self.resize(700, 660)

        lay = QtWidgets.QVBoxLayout(self)
        camrow = QtWidgets.QHBoxLayout()
        camrow.addWidget(QtWidgets.QLabel("Camera:"))
        self.cam_combo = QtWidgets.QComboBox()
        self.cam_combo.addItems(list(self.cams.keys()))
        self.cam_combo.currentTextChanged.connect(self.set_cam)
        camrow.addWidget(self.cam_combo, 1)
        lay.addLayout(camrow)
        self.video = QtWidgets.QLabel("Đang mở camera...")
        self.video.setObjectName("video")
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        lay.addWidget(self.video)

        self.lbl_angle = QtWidgets.QLabel("")
        f = self.lbl_angle.font()
        f.setPointSize(16)
        f.setBold(True)
        self.lbl_angle.setFont(f)
        self.lbl_angle.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.lbl_angle)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)

        row = QtWidgets.QHBoxLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("Tên (vd: toi)")
        self.btn_shoot = QtWidgets.QPushButton("Chụp (C)")
        self.btn_shoot.setObjectName("primary")
        self.btn_done = QtWidgets.QPushButton("Xong và lưu")
        row.addWidget(self.name_edit)
        row.addWidget(self.btn_shoot)
        row.addWidget(self.btn_done)
        lay.addLayout(row)

        self.btn_shoot.clicked.connect(self.shoot)
        self.btn_done.clicked.connect(self.finish)

        self.cap = face_engine.open_camera(self.rtsp)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(60)
        self.cur = None
        self.update_label()

    def set_cam(self, name):
        """Đổi cam enroll: mở lại stream + làm mới hướng dẫn (tránh trộn góc 2 view)."""
        if self.cap.isOpened():
            self.cap.release()
        self.rtsp = self.cams.get(name, self.rtsp)
        self.cap = face_engine.open_camera(self.rtsp)
        self.embs = []
        self.guide = headpose.PoseGuide(shots_per_target=2)
        self._hint = "đang tìm mặt..."
        self.update_label()

    def update_label(self):
        label, _, _ = self.guide.current
        self.lbl_angle.setText(f"{self.guide.idx}/{self.guide.total} - {label}: {self._hint}")
        self.progress.setValue(int(100 * self.guide.idx / max(1, self.guide.total)))

    def tick(self):
        ret, frame = self.cap.read() if self.cap.isOpened() else (False, None)
        if not ret:
            return
        self.cur = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        # tự detect + chụp theo góc (kiểu Face ID), mỗi 400ms một lần
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now - self._last_det >= 400 and not self.guide.done:
            self._last_det = now
            try:
                faces = face_engine.get_app().get(frame)
            except Exception:
                faces = []
            if faces:
                fc = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                ok, reason = headpose.face_quality(frame, fc.bbox)
                yaw, pitch = headpose.estimate_pose(fc)
                captured, _, hint = self.guide.update(yaw, pitch, ok)
                self._hint = reason if not ok else hint
                if captured:
                    self.embs.append(fc.normed_embedding.copy())
                    QtWidgets.QApplication.beep()
            else:
                self._hint = "không thấy mặt"
            self.update_label()
            if self.guide.done:
                self.finish()

    def shoot(self):
        if self.cur is None:
            return
        try:
            faces = face_engine.get_app().get(self.cur)
        except Exception:
            faces = []
        if not faces:
            QtWidgets.QMessageBox.warning(self, "Không thấy mặt", "Lại gần, đủ sáng, nhìn thẳng vào cam.")
            return
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        self.embs.append(faces[0].normed_embedding.copy())
        self.update_label()

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_C:
            self.shoot()
        super().keyPressEvent(ev)

    def finish(self):
        name = self.name_edit.text().strip() or "toi"
        if len(self.embs) < 8:
            QtWidgets.QMessageBox.warning(self, "Chưa đủ", "Chụp ít nhất 8 ảnh nhiều góc.")
            return
        os.makedirs(config.FACES_DIR, exist_ok=True)
        path = os.path.join(config.FACES_DIR, f"{name}.npy")
        new = np.stack(self.embs)
        if os.path.exists(path):
            try:
                old = np.load(path)
                if old.ndim == 1:
                    old = old.reshape(1, -1)
                # cộng dồn góc mới vào profile có sẵn, giữ tối đa 60 góc mới nhất
                new = np.vstack([old, new])[-60:]
                msg = f"Đã thêm {len(self.embs)} góc vào '{name}' (tổng {len(new)} góc)."
            except Exception:
                msg = f"File cũ lỗi, đã ghi đè '{name}' ({len(new)} góc)."
        else:
            msg = f"Đã tạo '{name}' ({len(new)} góc)."
        np.save(path, new)
        QtWidgets.QMessageBox.information(self, "Xong", msg)
        self.enrolled.emit(name)
        self.accept()

    def closeEvent(self, ev):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(ev)


# ---------- Cửa sổ chính ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Imou FaceID - 2 Cam")
        self.resize(1280, 760)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)

        # Cột trái: 1 panel video + badge cho mỗi cam
        left = QtWidgets.QVBoxLayout()
        self.cam_video = {}
        self.cam_badge = {}
        if not config.CAMERAS:
            left.addWidget(QtWidgets.QLabel("Chưa cấu hình cam nào (.env / cams.env)"))
        for cam in config.CAMERAS:
            title = QtWidgets.QLabel(cam["name"])
            f = title.font()
            f.setBold(True)
            f.setPointSize(15)
            title.setFont(f)
            left.addWidget(title)
            v = QtWidgets.QLabel("Nhấn Bắt đầu để mở camera")
            v.setObjectName("video")
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setMinimumSize(480, 270)
            left.addWidget(v, 1)
            b = QtWidgets.QLabel("ĐANG DỪNG")
            b.setObjectName("badge-idle")
            b.setAlignment(QtCore.Qt.AlignCenter)
            left.addWidget(b)
            self.cam_video[cam["name"]] = v
            self.cam_badge[cam["name"]] = b
        main.addLayout(left, 2)

        # Cột phải: điều khiển
        right = QtWidgets.QVBoxLayout()
        card = QtWidgets.QFrame()
        card.setObjectName("card")
        card_lay = QtWidgets.QVBoxLayout(card)

        self.btn_toggle = QtWidgets.QPushButton("Bắt đầu")
        self.btn_toggle.setObjectName("primary")
        self.btn_enroll = QtWidgets.QPushButton("Đăng ký mặt mới")
        card_lay.addWidget(QtWidgets.QLabel("Điều khiển"))
        card_lay.addWidget(self.btn_toggle)
        card_lay.addWidget(self.btn_enroll)

        card_lay.addWidget(QtWidgets.QLabel("Ngưỡng nhận diện (mọi cam)"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(30, 60)
        self.slider.setValue(int(config.THRESHOLD * 100))
        self.lbl_thresh = QtWidgets.QLabel(f"{config.THRESHOLD:.2f}")
        card_lay.addWidget(self.slider)
        card_lay.addWidget(self.lbl_thresh)
        right.addWidget(card)

        self.faces_list = QtWidgets.QListWidget()
        right.addWidget(QtWidgets.QLabel("Người đã đăng ký"))
        right.addWidget(self.faces_list, 1)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        right.addWidget(QtWidgets.QLabel("Nhật ký"))
        right.addWidget(self.log)
        main.addLayout(right, 1)

        self.btn_toggle.clicked.connect(self.toggle)
        self.btn_enroll.clicked.connect(self.open_enroll)
        self.slider.valueChanged.connect(self.on_thresh)

        self.workers = {}
        self._last_frame = {}
        self._prev_stranger = {}
        self.cam_thresh = {c["name"]: c["thresh"] for c in config.CAMERAS}
        self.refresh_faces()
        self.log_msg("Sẵn sàng. Nhấn Bắt đầu.")

    def log_msg(self, msg):
        self.log.append(f"[{QtCore.QTime.currentTime().toString()}] {msg}")

    def refresh_faces(self):
        self.faces_list.clear()
        known = face_engine.load_faces()
        for n, arr in known.items():
            self.faces_list.addItem(f"{n} ({arr.shape[0]} góc)")

    def on_thresh(self, v):
        for name in self.cam_thresh:
            self.cam_thresh[name] = v / 100
        self.lbl_thresh.setText(f"{v / 100:.2f}")

    def toggle(self):
        if not self.workers:
            known = face_engine.load_faces()
            if not known:
                QtWidgets.QMessageBox.information(self, "Chưa có dữ liệu",
                                                  "Nhấn 'Đăng ký mặt mới' trước.")
                return
            if not config.CAMERAS:
                QtWidgets.QMessageBox.information(self, "Chưa có cam",
                                                  "Điền .env (cam 1) / cams.env (cam 2).")
                return
            for cam in config.CAMERAS:
                name = cam["name"]
                w = VideoWorker(cam, lambda n=name: self.cam_thresh[n], self)
                w.frame_ready.connect(self.show_frame)
                w.status_ready.connect(self.show_status)
                w.finished.connect(lambda n=name: self.on_stopped(n))
                w.start()
                self.workers[name] = w
            telecmd.start(config.CAMERAS)
            self.btn_toggle.setText("Dừng lại")
            self.log_msg(f"Đã mở {len(self.workers)} camera + lệnh Telegram /cam.")
        else:
            for w in self.workers.values():
                w.stop()
            self.workers = {}
            telecmd.stop()
            self.btn_toggle.setText("Bắt đầu")

    def on_stopped(self, name):
        badge = self.cam_badge.get(name)
        if badge is None:
            return
        badge.setText("ĐANG DỪNG")
        badge.setObjectName("badge-idle")
        badge.setStyleSheet("")

    def show_frame(self, name, frame):
        self._last_frame[name] = frame.copy()
        video = self.cam_video.get(name)
        if video is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        video.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def show_status(self, name, res, n_far=0, n_poor=0):
        badge = self.cam_badge.get(name)
        if badge is None:
            return
        if not res:
            self._prev_stranger[name] = False
            if n_far > 0:
                badge.setText("ĐỨNG GẦN CAMERA HƠN")
                badge.setObjectName("badge-near")
            elif n_poor > 0:
                badge.setText("MẶT MỜ - ĐI CHẬM LẠI")
                badge.setObjectName("badge-near")
            else:
                badge.setText(face_engine.KHONG_THAY_MAT)
                badge.setObjectName("badge-idle")
        else:
            label, sim = max(res, key=lambda x: x[1])
            if label == face_engine.NGUOI_LA:
                badge.setText(f"{face_engine.NGUOI_LA} ({sim:.2f})")
                badge.setObjectName("badge-stranger")
                self.log_msg(f"[{name}] Cảnh báo: {face_engine.NGUOI_LA} ({sim:.2f})")
                is_new = not self._prev_stranger.get(name, False)
                self._prev_stranger[name] = True
                if self._last_frame.get(name) is not None:
                    img = self._last_frame[name].copy()
                    threading.Thread(target=notify.send_stranger_alert,
                                     args=(img, sim),
                                     kwargs={"is_new": is_new, "cam_name": name},
                                     daemon=True).start()
            elif label == "LỖI CAM":
                self._prev_stranger[name] = False
                badge.setText("LỖI CAM - kiểm tra RTSP")
                badge.setObjectName("badge-stranger")
            else:
                self._prev_stranger[name] = False
                badge.setText(f"TÔI: {label} ({sim:.2f}) - Chào bạn!")
                badge.setObjectName("badge-ok")
        # refresh style theo objectName mới
        badge.style().unpolish(badge)
        badge.style().polish(badge)

    def open_enroll(self):
        dlg = EnrollDialog(self)
        dlg.enrolled.connect(lambda n: (self.refresh_faces(), self.log_msg(f"Đã đăng ký: {n}")))
        dlg.exec()

    def closeEvent(self, ev):
        for w in self.workers.values():
            w.stop()
        self.workers = {}
        telecmd.stop()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Imou FaceID")
    t = load_caelestia_colors()
    app.setStyleSheet(build_qss(t))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
