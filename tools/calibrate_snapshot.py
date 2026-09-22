"""Снимки с камеры с координатной сеткой — для настройки зоны подсчёта.

Запуск внутри работающего контейнера (пересборка образа не нужна):

    docker exec rtsp-backend python /app/uploads/calibrate_snapshot.py "rtsp://логин:пароль@адрес:554/поток"

Результат появится в папке data/uploads/calib рядом с docker-compose.yml.
Каждый кадр размечен сеткой в долях кадра и текущей зоной подсчёта.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, "/app")

try:
    from app.analytics import ZONE_ABOVE_HOPPER, ZONE_CONTROL, _zone_bounds
except Exception:
    ZONE_ABOVE_HOPPER = (0.490, 0.000, 0.680, 0.212)
    ZONE_CONTROL = (0.050, 0.020, 0.300, 0.250)
    _zone_bounds = None

OUT_DIR = Path("/app/uploads/calib")
SHOTS = 12
INTERVAL_SECONDS = 10.0
WORK_WIDTH = 1280


def zone_px(width: int, height: int, zone) -> tuple[int, int, int, int]:
    if _zone_bounds is not None:
        return _zone_bounds(width, height, zone)
    x0, y0, x1, y1 = zone
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


def annotate(frame):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = zone_px(width, height, ZONE_ABOVE_HOPPER)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 128, 255), 4)
    cv2.putText(frame, "ZONA SCHETA", (x1 + 6, max(y1 + 34, 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (32, 128, 255), 3, cv2.LINE_AA)

    cx1, cy1, cx2, cy2 = zone_px(width, height, ZONE_CONTROL)
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (120, 200, 120), 3)

    # сетка в долях кадра: по ней измеряются новые координаты зоны
    for step in range(1, 10):
        fraction = step / 10.0
        gx = int(width * fraction)
        gy = int(height * fraction)
        colour = (255, 255, 255) if step != 5 else (0, 255, 255)
        thickness = 1 if step != 5 else 2
        cv2.line(frame, (gx, 0), (gx, height), colour, thickness)
        cv2.line(frame, (0, gy), (width, gy), colour, thickness)
        cv2.putText(frame, f"{fraction:.1f}", (gx + 4, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{fraction:.1f}", (gx + 4, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fraction:.1f}", (6, gy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"{fraction:.1f}", (6, gy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    if width > WORK_WIDTH:
        scale = WORK_WIDTH / float(width)
        frame = cv2.resize(frame, (WORK_WIDTH, int(round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame


def main() -> int:
    if len(sys.argv) < 2:
        print("Укажите RTSP-ссылку камеры первым аргументом.")
        return 1

    source = sys.argv[1]
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else SHOTS
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else INTERVAL_SECONDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("Не удалось подключиться к камере. Проверьте ссылку и доступность адреса.")
        return 2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Камера подключена, кадр {width}x{height}")
    print(f"Снимаю {shots} кадров с интервалом {interval:.0f} с — это займёт "
          f"{shots * interval / 60:.0f} мин.")

    saved = 0
    for index in range(shots):
        deadline = time.monotonic() + interval
        frame = None
        while time.monotonic() < deadline:
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
            else:
                time.sleep(0.2)
            if time.monotonic() >= deadline - 0.1:
                break

        if frame is None:
            print(f"  кадр {index + 1}: не получен")
            continue

        target = OUT_DIR / f"calib-{index + 1:02d}.jpg"
        if cv2.imwrite(str(target), annotate(frame), [cv2.IMWRITE_JPEG_QUALITY, 88]):
            saved += 1
            print(f"  кадр {index + 1}: сохранён -> data/uploads/calib/{target.name}")

    capture.release()
    print(f"\nГотово: {saved} кадров в папке data/uploads/calib")
    print("Пришлите те из них, где мешок опускается в бункер, и один-два пустых.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
