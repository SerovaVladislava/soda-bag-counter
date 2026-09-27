"""Ловит моменты, когда в верхней части кадра что-то появляется (мешок на
кране), и сохраняет эти кадры с координатной сеткой и текущими зонами.
Сидеть у экрана не нужно: скрипт сам отбирает кадры, где есть что смотреть.

Запуск внутри работающего контейнера (образ пересобирать не нужно):

    docker compose exec backend python /app/uploads/watch_snapshots.py "rtsp://логин:пароль@адрес:554/поток" 8

Второй аргумент - сколько часов наблюдать (по умолчанию 8). Кадры появляются в
data/uploads/calib рядом с docker-compose.yml:

    watch-2509-150412-a0.31.jpg   что-то в верхней части кадра, a - доля
                                  изменившихся пикселей (чем больше, тем крупнее)
    ref-2509-150000.jpg           опорный кадр раз в 30 минут, для пустой сцены

Пришлите те кадры, где мешок висит над бункером, и один-два опорных.
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")

try:
    from app.analytics import ZONE_ABOVE_HOPPER, ZONE_CONTROL
except Exception:
    ZONE_ABOVE_HOPPER = (0.140, 0.000, 0.560, 0.280)
    ZONE_CONTROL = (0.720, 0.020, 0.970, 0.240)

OUT_DIR = Path(os.environ.get("WATCH_OUT_DIR", "/app/uploads/calib"))
LOCAL = timezone(timedelta(hours=5))  # Екатеринбург, как в main.py

SAMPLE_SECONDS = 1.0        # как часто сравнивать кадр с фоном
BACKGROUND_STEP = 5.0       # шаг проб, из которых строится фон
BACKGROUND_WINDOW = 600.0   # фон = медиана за последние 10 минут:
                            # висящий минуту мешок в него не попадает
TOP_FRACTION = 0.45         # верхняя часть кадра, где ходит кран
DIFF_LEVEL = 35             # перепад яркости, который считается изменением
ACTIVITY_MIN = 0.10         # доля изменившихся пикселей, с которой кадр сохраняется
SAVE_GAP_SECONDS = 15.0     # не чаще одного кадра в 15 с
REF_SECONDS = 1800.0        # опорный кадр раз в 30 минут
MAX_FILES = 400
WORK_WIDTH = 1280


def annotate(frame, activity: float, stamp: datetime):
    frame = frame.copy()
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = (int(ZONE_ABOVE_HOPPER[0] * width), int(ZONE_ABOVE_HOPPER[1] * height),
                      int(ZONE_ABOVE_HOPPER[2] * width), int(ZONE_ABOVE_HOPPER[3] * height))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 128, 255), 4)
    cv2.putText(frame, "ZONA SCHETA", (x1 + 6, max(y1 + 34, 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (32, 128, 255), 3, cv2.LINE_AA)
    cx1, cy1, cx2, cy2 = (int(ZONE_CONTROL[0] * width), int(ZONE_CONTROL[1] * height),
                          int(ZONE_CONTROL[2] * width), int(ZONE_CONTROL[3] * height))
    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (120, 200, 120), 3)

    for step in range(1, 10):
        fraction = step / 10.0
        gx, gy = int(width * fraction), int(height * fraction)
        colour = (255, 255, 255) if step != 5 else (0, 255, 255)
        thickness = 1 if step != 5 else 2
        cv2.line(frame, (gx, 0), (gx, height), colour, thickness)
        cv2.line(frame, (0, gy), (width, gy), colour, thickness)
        for text, org in ((f"{fraction:.1f}", (gx + 4, 26)), (f"{fraction:.1f}", (6, gy - 6))):
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    label = f"{stamp:%d.%m %H:%M:%S}  activity {activity:.2f}"
    cv2.putText(frame, label, (6, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (6, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    if width > WORK_WIDTH:
        scale = WORK_WIDTH / float(width)
        frame = cv2.resize(frame, (WORK_WIDTH, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return frame


def top_band(frame) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    return small[: int(120 * TOP_FRACTION)].astype(np.int16)


def main() -> int:
    if len(sys.argv) < 2:
        print("Укажите RTSP-ссылку камеры первым аргументом.")
        return 1
    source = sys.argv[1]
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    is_file = not source.lower().startswith(("rtsp://", "rtsps://"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print("Не удалось подключиться к источнику. Проверьте ссылку и доступность адреса.")
        return 2

    file_fps = capture.get(cv2.CAP_PROP_FPS) if is_file else 0.0
    if is_file and not (1.0 <= file_fps <= 120.0):
        file_fps = 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Источник открыт, кадр {width}x{height}. Наблюдаю {hours:g} ч, кадры -> {OUT_DIR}")
    sys.stdout.flush()

    started = time.monotonic()
    frame_index = 0
    background_samples: deque[tuple[float, np.ndarray]] = deque()
    background = None
    last_sample_t = -1e9
    last_background_t = -1e9
    last_save_t = -1e9
    last_ref_t = -1e9
    saved = 0
    failures = 0

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            failures += 1
            if is_file or failures > 300:
                break
            time.sleep(0.2)
            continue
        failures = 0
        frame_index += 1
        now = frame_index / file_fps if is_file else time.monotonic() - started
        if now >= hours * 3600.0:
            break
        if now - last_sample_t < SAMPLE_SECONDS:
            continue
        last_sample_t = now

        band = top_band(frame)
        if now - last_background_t >= BACKGROUND_STEP:
            background_samples.append((now, band))
            while background_samples and now - background_samples[0][0] > BACKGROUND_WINDOW:
                background_samples.popleft()
            last_background_t = now
            if len(background_samples) >= 6:
                background = np.median(np.stack([b for _, b in background_samples]), axis=0)

        stamp = datetime.now(LOCAL) if not is_file else datetime(2000, 1, 1, tzinfo=LOCAL) + timedelta(seconds=now)
        if now - last_ref_t >= REF_SECONDS and saved < MAX_FILES:
            target = OUT_DIR / f"ref-{stamp:%d%m-%H%M%S}.jpg"
            cv2.imwrite(str(target), annotate(frame, 0.0, stamp), [cv2.IMWRITE_JPEG_QUALITY, 85])
            last_ref_t = now
            saved += 1
            print(f"  опорный кадр -> {target.name}")
            sys.stdout.flush()

        if background is None:
            continue
        activity = float((np.abs(band - background) > DIFF_LEVEL).mean())
        if activity >= ACTIVITY_MIN and now - last_save_t >= SAVE_GAP_SECONDS and saved < MAX_FILES:
            target = OUT_DIR / f"watch-{stamp:%d%m-%H%M%S}-a{activity:.2f}.jpg"
            cv2.imwrite(str(target), annotate(frame, activity, stamp), [cv2.IMWRITE_JPEG_QUALITY, 85])
            last_save_t = now
            saved += 1
            print(f"  {stamp:%d.%m %H:%M:%S}  движение в верхней части кадра {activity:.2f} -> {target.name}")
            sys.stdout.flush()

    capture.release()
    print(f"\nГотово: {saved} кадров в папке {OUT_DIR}")
    if saved >= MAX_FILES:
        print("Достигнут предел файлов - наблюдение остановлено раньше срока.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
