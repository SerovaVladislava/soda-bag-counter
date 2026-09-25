"""Сводка по журналу проб zone_samples.csv: когда мешок был в зоне, сколько
держался, засчитан ли, и что именно мешало, если нет.

Запуск внутри контейнера (журнал лежит на томе с данными):

    docker compose exec backend python /app/uploads/summarize_samples.py

Скрипт нужно один раз положить в data/uploads рядом с журналом.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/uploads/zone_samples.csv")
LOCAL = timezone(timedelta(hours=5))  # Екатеринбург, как в main.py
FILL_ENTER = 0.21        # ZONE_FILL_ENTER
FILL_MAX = 0.85          # ZONE_FILL_MAX
BASELINE_MARGIN = 0.06   # запас над базовой линией пустой зоны
MIN_PRESENT = 10.0       # ZONE_MIN_PRESENT_SECONDS
GAP_SECONDS = 20.0       # разрыв, который ещё не делит одно появление на два

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(LOCAL)


def main() -> int:
    if not LOG.exists():
        print(f"Журнал не найден: {LOG}")
        return 1

    rows = []
    with LOG.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "t": parse_time(raw["time_utc"]),
                        "fill": float(raw["fill"]),
                        "control": float(raw["control"]),
                        "baseline": float(raw["baseline"]),
                        "present": raw["present"] == "1",
                        "committed": raw["committed"] == "1",
                        "state": raw["state"],
                        "suppressed": raw["suppressed"] == "1",
                        "stalled": raw["stalled"] == "1",
                        "count": int(raw["count"]),
                    }
                )
            except (KeyError, ValueError):
                continue
            r = rows[-1]
            r["in_zone"] = r["present"] or r["fill"] >= max(FILL_ENTER, r["baseline"] + BASELINE_MARGIN)

    if not rows:
        print("Журнал пуст.")
        return 1

    first, last = rows[0]["t"], rows[-1]["t"]
    hours = max((last - first).total_seconds() / 3600.0, 1e-6)
    idle = sorted(r["fill"] for r in rows if not r["in_zone"])
    ctrl = sorted(r["control"] for r in rows)

    def pct(values, q):
        return values[min(int(len(values) * q), len(values) - 1)] if values else 0.0

    print(f"Журнал: {LOG}")
    print(f"Период: {first:%d.%m %H:%M} - {last:%d.%m %H:%M} ({hours:.1f} ч), проб {len(rows)}")
    print(f"Засчитано мешков за период: {sum(1 for r in rows if r['committed'])}, счётчик на конце журнала: {rows[-1]['count']}")
    print()
    print("Пустая зона (пробы без мешка):")
    print(f"  медиана {pct(idle, 0.5):.3f}   p95 {pct(idle, 0.95):.3f}   максимум {idle[-1] if idle else 0:.3f}")
    print(f"  контрольная зона: медиана {pct(ctrl, 0.5):.3f}, p95 {pct(ctrl, 0.95):.3f}")
    threshold = max(FILL_ENTER, pct(sorted(r["baseline"] for r in rows if r["baseline"] > 0), 0.5) + BASELINE_MARGIN)
    print(f"  действующий порог входа {threshold:.3f}; запас над p95 пустой зоны: {threshold - pct(idle, 0.95):+.3f}")

    stalled = sum(1 for r in rows if r["stalled"])
    suppressed = sum(1 for r in rows if r["suppressed"])
    if stalled:
        print(f"  ВНИМАНИЕ: {stalled} проб с признаком замершего потока ({stalled * 100 // len(rows)}% времени)")
    if suppressed:
        print(f"  ВНИМАНИЕ: {suppressed} проб под замком после застрявшего эпизода")

    # кластеры: подряд идущие пробы, где в зоне что-то есть; разрывы до GAP_SECONDS не рвут
    events = []
    current = None
    for r in rows:
        if r["in_zone"]:
            if current and (r["t"] - current["last"]).total_seconds() <= GAP_SECONDS:
                current["last"] = r["t"]
                current["n"] += 1
                current["peak"] = max(current["peak"], r["fill"])
                current["present"] += int(r["present"])
                current["committed"] |= r["committed"]
            else:
                if current:
                    events.append(current)
                current = {"start": r["t"], "last": r["t"], "n": 1, "peak": r["fill"],
                           "present": int(r["present"]), "committed": r["committed"]}
        elif current and r["committed"]:
            current["committed"] = True
    if current:
        events.append(current)

    print()
    if not events:
        print("Мешок в зоне не появлялся ни разу: заполнение ни на одной пробе не превысило порог.")
        print("Если выгрузки за это время были - порог для этой сцены завышен, пришлите журнал.")
        return 0

    print(f"Появления в зоне: {len(events)}")
    print(f"{'начало':>14} {'длит.':>7} {'проб':>5} {'пик':>6}  решение")
    for e in events:
        dur = (e["last"] - e["start"]).total_seconds()
        if e["committed"]:
            verdict = "ЗАСЧИТАН"
        elif e["present"] == 0 and e["peak"] > FILL_MAX:
            verdict = f"не засчитан: заполнение выше {FILL_MAX:.2f} - объект вплотную к камере"
        elif dur < MIN_PRESENT:
            verdict = f"не засчитан: в зоне {dur:.0f} с, меньше {MIN_PRESENT:.0f} - перенос мимо камеры"
        elif e["present"] < e["n"] // 2:
            verdict = "не засчитан: пробы с мешком - меньше половины (отсечено контрольной зоной или потолком)"
        else:
            verdict = "НЕ ЗАСЧИТАН - разобрать по журналу"
        print(f"{e['start']:%d.%m %H:%M:%S} {dur:>6.0f}с {e['n']:>5} {e['peak']:>6.3f}  {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
