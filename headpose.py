"""Enroll kiểu Apple Face ID: dẫn dắt xoay đầu, tự chụp khi đúng góc.

- estimate_pose: đo hướng đầu (yaw/pitch tương đối) từ 5 điểm mốc InsightFace.
  yaw > 0 = quay sang TRÁI (của chủ thể), pitch > 0 = cúi xuống.
  (Nếu thấy hướng dẫn ngược trên máy bạn thì báo để đảo dấu.)
- face_quality: loại ảnh mờ / tối / chói trước khi chụp.
- PoseGuide: máy trạng thái - hiệu chuẩn frame đầu, rồi đi qua từng góc,
  đủ ổn định mới "tách" chụp 1 tấm.
"""
import cv2
import numpy as np

# (nhãn, yaw_mục_tiêu, pitch_mục_tiêu) - yaw/pitch tương đối so với mặt thẳng
TARGETS = [
    ("Nhìn thẳng", 0.00, 0.00),
    ("Quay trái", 0.30, 0.00),
    ("Quay phải", -0.30, 0.00),
    ("Ngước lên", 0.00, -0.18),
    ("Cúi xuống", 0.00, 0.22),
    ("Trái, cười nhẹ", 0.22, 0.00),
    ("Phải, cười nhẹ", -0.22, 0.00),
    ("Nghiêng trái mạnh", 0.42, 0.00),
    ("Nghiêng phải mạnh", -0.42, 0.00),
]

TOL = 0.10          # dung sai góc
STABLE_HITS = 3     # số lần liên tiếp đúng góc mới chụp
BLUR_MIN = 60.0     # Laplacian variance tối thiểu (chống rung/mờ)


def estimate_pose(face):
    """Trả (yaw, pitch) tuyệt đối từ face InsightFace (có .kps 5 điểm)."""
    kps = np.asarray(face.kps, dtype=np.float32)
    le, re, no, ml, mr = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_c = (le + re) / 2.0
    eye_d = float(np.linalg.norm(le - re)) + 1e-6
    mouth_c = (ml + mr) / 2.0
    vdist = float(abs(mouth_c[1] - eye_c[1])) + 1e-6
    yaw = float((no[0] - eye_c[0]) / eye_d)
    pitch = float((no[1] - eye_c[1]) / vdist)
    return yaw, pitch


def face_quality(frame, bbox):
    """Trả (ok, lý_do). Loại ảnh mờ/tối/chói."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return False, "đứng gần hơn"
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_MIN:
        return False, "giữ yên, đang mờ"
    bright = float(gray.mean())
    if bright < 50:
        return False, "tối quá, thêm đèn"
    if bright > 210:
        return False, "chói quá"
    return True, "ok"


def direction_hint(dyaw, dpitch, tyaw, tpitch, tol=TOL):
    """Câu hướng dẫn để đi từ pose hiện tại tới pose mục tiêu."""
    hints = []
    if tyaw - dyaw > tol:
        hints.append("quay trái thêm")
    elif dyaw - tyaw > tol:
        hints.append("quay phải thêm")
    if tpitch - dpitch > tol:
        hints.append("cúi xuống thêm")
    elif dpitch - tpitch > tol:
        hints.append("ngước lên thêm")
    return ", ".join(hints)


class PoseGuide:
    """Dẫn dắt qua targets (lặp lại shots_per_target vòng)."""

    def __init__(self, targets=TARGETS, shots_per_target=2, tol=TOL, stable_hits=STABLE_HITS):
        self.targets = [(l, y, p) for l, y, p in targets for _ in range(shots_per_target)]
        self.tol = tol
        self.stable_hits = stable_hits
        self.idx = 0
        self.hits = 0
        self.base = None  # (yaw, pitch) lúc nhìn thẳng

    @property
    def total(self):
        return len(self.targets)

    @property
    def done(self):
        return self.idx >= len(self.targets)

    @property
    def current(self):
        if self.done:
            return ("Xong!", 0.0, 0.0)
        return self.targets[self.idx]

    def update(self, yaw, pitch, quality_ok):
        """Trả (da_chup, nhan, goi_y). Gọi mỗi lần detect được mặt."""
        if self.done:
            return False, "Xong!", ""
        if self.base is None:
            self.base = (yaw, pitch)
        label, tyaw, tpitch = self.current
        dyaw, dpitch = yaw - self.base[0], pitch - self.base[1]
        if not quality_ok:
            self.hits = 0
            return False, label, "giữ yên"
        hint = direction_hint(dyaw, dpitch, tyaw, tpitch, self.tol)
        if hint:
            self.hits = 0
            return False, label, hint
        self.hits += 1
        if self.hits >= self.stable_hits:
            self.idx += 1
            self.hits = 0
            return True, label, "đẹp!"
        return False, label, "giữ nguyên..."
