"""Чистка истории мешков по сменам от выдуманных строк.

До версии от 08.10.2026 сервер при первом запуске сам дописывал в историю
30 дней «демонстрационных» значений для потоков из реестра и для
несуществующего потока «Цех по загрузке реагентов». Скрипт убирает строки
потоков, которых нет в реестре, и строки старше указанной даты (всё, что
было до запуска настоящего подсчёта).

Запуск внутри контейнера:

    docker compose exec backend python /app/uploads/reset_shift_history.py 2026-09-27

Дата - первая, которую нужно оставить. Без даты удаляются только строки
несуществующих потоков. Исходный файл сохраняется рядом как
bag_shift_history.json.bak.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

UPLOADS = Path("/app/uploads")
HISTORY = UPLOADS / "bag_shift_history.json"
REGISTRY = UPLOADS / "rtsp_stream_history.json"


def main() -> int:
    keep_from = sys.argv[1] if len(sys.argv) > 1 else None
    if not HISTORY.exists():
        print("история пуста, чистить нечего")
        return 0
    rows = json.loads(HISTORY.read_text(encoding="utf-8") or "[]")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8") or "[]") if REGISTRY.exists() else []
    known = {str(s.get("id") or "") for s in registry}

    kept, dropped = [], []
    for row in rows:
        stream_id = str(row.get("stream_id") or "")
        date = str(row.get("date") or "")
        if stream_id not in known or (keep_from and date < keep_from):
            dropped.append(row)
        else:
            kept.append(row)

    shutil.copy(HISTORY, HISTORY.with_suffix(".json.bak"))
    HISTORY.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"оставлено строк: {len(kept)}, удалено: {len(dropped)}")
    for row in dropped[:10]:
        print(f"  - {row.get('date')} {row.get('stream_name')}: день {row.get('day_count')}, ночь {row.get('night_count')}")
    if len(dropped) > 10:
        print(f"  ... и ещё {len(dropped) - 10}")
    print("перезапустите сервер: docker compose restart backend")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
