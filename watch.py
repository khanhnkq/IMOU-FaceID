"""Service canh cam 24/7: quét mặt, thấy NGƯỜI LẠ thì báo Telegram.

Không cửa sổ, không GUI. Bật/tắt nhanh bằng toggle-faceid.sh (SUPER + F9).

Chạy tay:
    .venv/bin/python watch.py
"""
import argparse
import time

import config
import face_engine
import notify


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rtsp", default=config.RTSP_URL)
    p.add_argument("--thresh", type=float, default=config.THRESHOLD)
    args = p.parse_args()

    known = face_engine.load_faces()
    if not known:
        print("Chưa có ai đăng ký! Mở app_gui.py -> Đăng ký mặt mới.")
        return

    print(f"Đã load: {list(known.keys())} | ngưỡng={args.thresh}")
    print("Đang canh. Ctrl+C để dừng.")

    cap = face_engine.open_camera(args.rtsp)
    if not cap.isOpened():
        print("KHÔNG MỞ ĐƯỢC CAM. Kiểm tra IP + Safety Code trong .env.")
        return

    frame_id = 0
    last_counter = -1
    prev_stranger = False
    prev_gray = None
    text = "..."

    try:
        while True:
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
                near = config.NEAR_MIN_RATIO if config.NEAR_MODE else 0.0
                motion, prev_gray = face_engine.has_motion(
                    prev_gray, frame, config.MOTION_THRESH)
                if config.MOTION_GATE and not motion:
                    text = "CANH TĨNH - ĐANG NGHỈ"
                else:
                    frame, res, n_far, n_poor = face_engine.match_frame(
                        frame, known, args.thresh, near)
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
                            if notify.send_stranger_alert(frame, sim, is_new=is_new):
                                print("Đã gửi cảnh báo Telegram.", flush=True)

            if frame_id % 25 == 0:
                print(text, flush=True)

    except KeyboardInterrupt:
        print("\nĐã dừng.")

    cap.release()


if __name__ == "__main__":
    main()
