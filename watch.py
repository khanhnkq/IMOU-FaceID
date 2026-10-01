"""Service canh cam 24/7 (đa cam): quét mặt, thấy NGƯỜI LẠ thì báo Telegram.

Không cửa sổ, không GUI. Bật/tắt nhanh bằng toggle-faceid.sh (SUPER + F9).

Chạy tay:
    .venv/bin/python watch.py
"""
import argparse
import threading
import time

import config
import face_engine
import notify
import telecmd


def watch_cam(cam, known, stop_event):
    name, thresh = cam["name"], cam["thresh"]
    print(f"[{name}] Đang mở camera (ngưỡng {thresh})...", flush=True)
    cap = face_engine.open_camera(cam["rtsp"])
    if not cap.isOpened():
        print(f"[{name}] KHÔNG MỞ ĐƯỢC CAM. Kiểm tra IP + Safety Code.", flush=True)
        return

    frame_id = 0
    last_counter = -1
    prev_stranger = False
    prev_gray = None
    text = "..."

    while not stop_event.is_set():
        ret, frame, last_counter = cap.read_new(last_counter)
        if not ret:
            time.sleep(0.2)
            continue
        if frame is None:  # chưa có hình mới
            time.sleep(0.015)
            continue
        frame_id += 1

        # chỉ quét 1/N frame cho nhẹ CPU (N trong .env)
        if frame_id % config.RECOG_EVERY_N_FRAMES == 0:
            motion, prev_gray = face_engine.has_motion(
                prev_gray, frame, config.MOTION_THRESH)
            if config.MOTION_GATE and not motion:
                text = "CANH TĨNH - ĐANG NGHỈ"
            else:
                frame, res, n_far, n_poor = face_engine.match_frame(
                    frame, known, thresh, cam["near"])
                if not res:
                    if n_far > 0:
                        text = "ĐỨNG GẦN CAMERA HƠN"
                    elif n_poor > 0:
                        text = "MẶT MỜ - ĐI CHẬM LẠI"
                    else:
                        text = face_engine.KHONG_THAY_MAT
                    prev_stranger = False
                else:
                    best = max(res, key=lambda r: (r[2][2] - r[2][0]) * (r[2][3] - r[2][1]))
                    label, sim = best[0], best[1]
                    if label != face_engine.NGUOI_LA:
                        text = f"TÔI: {label} ({sim:.2f})"
                        prev_stranger = False
                    else:
                        text = f"{face_engine.NGUOI_LA} ({sim:.2f})"
                        is_new = not prev_stranger
                        prev_stranger = True
                        if notify.send_stranger_alert(frame, sim, is_new=is_new,
                                                      cam_name=name):
                            print(f"[{name}] Đã gửi cảnh báo Telegram.", flush=True)

        if frame_id % 25 == 0:
            print(f"[{name}] {text}", flush=True)

    cap.release()
    print(f"[{name}] Đã dừng.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thresh", type=float, default=None,
                   help="ghi đè ngưỡng mọi cam")
    args = p.parse_args()

    known = face_engine.load_faces()
    if not known:
        print("Chưa có ai đăng ký! Mở app_gui.py -> Đăng ký mặt mới.")
        return
    if not config.CAMERAS:
        print("Chưa có cam nào cấu hình! Điền .env (cam 1) / cams.env (cam 2).")
        return
    if args.thresh is not None:
        for c in config.CAMERAS:
            c["thresh"] = args.thresh

    print(f"Đã load mặt: {list(known.keys())}")
    print(f"Canh {len(config.CAMERAS)} cam: "
          + ", ".join(f"{c['name']} (ngưỡng {c['thresh']})" for c in config.CAMERAS))
    print("Ctrl+C để dừng.")

    stop_event = threading.Event()
    threads = [threading.Thread(target=watch_cam, args=(c, known, stop_event),
                                daemon=True, name=f"watch-{c['name']}")
               for c in config.CAMERAS]
    for t in threads:
        t.start()
    telecmd.start(config.CAMERAS)  # nghe lệnh /cam trên Telegram
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nĐang dừng...")
        stop_event.set()
        telecmd.stop()
        for t in threads:
            t.join(timeout=5)
        print("Đã dừng.")


if __name__ == "__main__":
    main()
