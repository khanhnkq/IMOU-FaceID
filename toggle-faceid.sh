#!/usr/bin/env bash
# Bật/tắt nhanh service canh cam headless (báo qua Telegram, không cửa sổ).
# Gán vào phím tắt Hyprland, vd: SUPER + F9.
# PAT dùng mẹo [f] để pkill/pgrep không tự khớp chính nó.
DIR="$HOME/Codes/imou-faceid"
PAT="[w]atch.py"

note() {
    timeout 3 notify-send "FaceID" "$1" -a FaceID 2>/dev/null &
}

if pgrep -f "$PAT" >/dev/null; then
    pkill -f "$PAT"
    note "Đã tắt canh cam"
else
    cd "$DIR" || exit 1
    nohup .venv/bin/python watch.py >>data/faceid.log 2>&1 &
    note "Đã bật canh cam (báo qua Telegram)"
fi
