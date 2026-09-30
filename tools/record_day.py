"""Запись потока камеры кусками по 10 минут - чтобы настраивать счёт по
реальному видео с площадки, а не по журналу после.

Запуск внутри работающего контейнера (ffmpeg там уже есть):

    docker compose exec -d backend python /app/uploads/record_day.py "rtsp://логин:пароль@адрес:554/поток" 24

Второй аргумент - сколько часов писать (по умолчанию 24). Файлы появляются в
data/uploads/recordings рядом с docker-compose.yml:

    rec-2026-10-01_14-00-00.mp4   начало куска по местному времени, 10 минут,
                                  1024x768, 2 кадра/с - 10-20 МБ на кусок

За сутки набирается 1-3 ГБ; перед каждым куском проверяется, что на диске
есть хотя бы 3 ГБ, иначе запись останавливается. Мониторинг продолжает
работать: это отдельное подключение к камере. Ход записи - в
data/uploads/recordings/record.log.

Остановить раньше срока:

    docker compose exec backend pkill -f record_day.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT_DIR = Path(os.environ.get("RECORD_OUT_DIR", "/app/uploads/recordings"))
FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")
LOCAL = timezone(timedelta(hours=5))  # Екатеринбург, как в main.py
SEGMENT_SECONDS = 600
MIN_FREE_BYTES = 3 * 1024 ** 3
RETRY_SECONDS = 5.0
# Счёт берёт пробу раз в 2 с, поэтому 2 кадра/с хватает и для проверки счёта,
# и чтобы глазами увидеть, что было. Размер подобран по записям с площадки.
RECORD_FPS = os.environ.get("RECORD_FPS", "2")
RECORD_WIDTH = os.environ.get("RECORD_WIDTH", "1024")
RECORD_CRF = os.environ.get("RECORD_CRF", "28")


def log(handle, message: str) -> None:
    line = f"{datetime.now(LOCAL):%d.%m %H:%M:%S}  {message}"
    print(line, flush=True)
    try:
        handle.write(line + "\n")
        handle.flush()
    except Exception:
        pass


def ffmpeg_command(source: str, seconds: float, target: Path) -> list[str]:
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    if source.lower().startswith(("rtsp://", "rtsps://")):
        cmd += ["-rtsp_transport", "tcp"]
    cmd += [
        "-i", source,
        "-t", f"{seconds:.0f}",
        "-vf", f"fps={RECORD_FPS},scale={RECORD_WIDTH}:-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", RECORD_CRF, "-pix_fmt", "yuv420p",
        "-an",
        # фрагментированный mp4 читается целиком, даже если запись оборвали на середине
        "-movflags", "+frag_keyframe+empty_moov",
        "-y", str(target),
    ]
    return cmd


def main() -> int:
    if len(sys.argv) < 2:
        print("Укажите RTSP-ссылку камеры первым аргументом.")
        return 1
    source = sys.argv[1]
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0
    segment = float(sys.argv[3]) if len(sys.argv) > 3 else SEGMENT_SECONDS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handle = (OUT_DIR / "record.log").open("a", encoding="utf-8")
    deadline = time.monotonic() + hours * 3600.0
    log(handle, f"запись начата: {hours:g} ч кусками по {segment:.0f} с -> {OUT_DIR}")

    written = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            break
        free = shutil.disk_usage(OUT_DIR).free
        if free < MIN_FREE_BYTES:
            log(handle, f"на диске осталось {free / 1024 ** 3:.1f} ГБ - запись остановлена, чтобы не забить диск")
            break

        stamp = datetime.now(LOCAL)
        target = OUT_DIR / f"rec-{stamp:%Y-%m-%d_%H-%M-%S}.mp4"
        seconds = min(segment, remaining)
        started = time.monotonic()
        try:
            result = subprocess.run(ffmpeg_command(source, seconds, target), capture_output=True, text=True)
            code = result.returncode
            err = (result.stderr or "").strip().splitlines()
        except FileNotFoundError:
            log(handle, f"ffmpeg не найден ({FFMPEG})")
            return 2
        elapsed = time.monotonic() - started
        size = target.stat().st_size if target.exists() else 0
        if code == 0 and size > 0:
            written += 1
            log(handle, f"{target.name}: {size / 1024 ** 2:.0f} МБ за {elapsed:.0f} с")
        else:
            tail = err[-1] if err else "без сообщения"
            log(handle, f"{target.name}: ffmpeg завершился с кодом {code} через {elapsed:.0f} с ({tail})")
            if size == 0 and target.exists():
                target.unlink()
            if elapsed < RETRY_SECONDS:
                time.sleep(RETRY_SECONDS)

    log(handle, f"запись закончена: кусков {written}, папка {OUT_DIR}")
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
