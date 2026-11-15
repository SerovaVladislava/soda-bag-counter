"""Исправить историю мешков по сменам за один день - без интерфейса.

Запуск внутри контейнера (скрипт положить в data/uploads):

    docker compose exec backend python /app/uploads/set_shift_history.py 2026-11-13 4 2

Аргументы: дата, мешков за дневную смену, мешков за ночную смену. Если
потоков несколько, четвёртым аргументом - часть имени потока. Скрипт
обращается к работающему серверу, поэтому перезапуск не нужен: строка сразу
меняется и в файле истории, и на странице.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("BAG_API", "http://127.0.0.1:8000")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def call(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Сервер ответил {exc.code}: {body}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Сервер не отвечает ({exc.reason}). Запущен ли backend?")


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    day, day_count, night_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    name_part = sys.argv[4].lower() if len(sys.argv) > 4 else ""
    if day_count < 0 or night_count < 0:
        raise SystemExit("Количество мешков не может быть отрицательным.")

    query = urllib.parse.urlencode({"date_from": day, "date_to": day, "limit": 5})
    history = call("GET", f"/api/rtsp-monitor/history?{query}")
    streams = [s for s in history.get("streams") or [] if s.get("id")]
    row = next((item for item in history.get("items") or [] if item.get("date") == day), None)
    entries = (row or {}).get("entries") or {}

    if name_part:
        streams = [s for s in streams if name_part in str(s.get("name") or "").lower()]
    elif len(streams) > 1:
        with_counts = [s for s in streams if (entries.get(s["id"]) or {}).get("total_count")]
        streams = with_counts or streams
    if len(streams) != 1:
        names = ", ".join(str(s.get("name")) for s in history.get("streams") or [])
        raise SystemExit(f"Не понял, какой поток править (потоки: {names}). Добавьте часть имени четвёртым аргументом.")

    stream = streams[0]
    before = entries.get(stream["id"]) or {}
    print(f"{day}, {stream.get('name')}: было день {before.get('day_count', 0)}, ночь {before.get('night_count', 0)}")
    result = call("PATCH", f"/api/rtsp-monitor/history/{day}", {
        "entries": [{"stream_id": stream["id"], "day_count": day_count, "night_count": night_count}],
    })
    after = next((r for r in result.get("items_updated") or [] if r.get("stream_id") == stream["id"]), {})
    print(f"{day}, {stream.get('name')}: стало день {after.get('day_count')}, ночь {after.get('night_count')}")
    print(result.get("message", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
