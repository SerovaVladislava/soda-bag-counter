from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from datetime import date as date_value, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape
from zipfile import ZIP_DEFLATED, ZipFile
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.analytics import BagAnalyticsManager

BASE_DIR = Path("/app")
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_INDEX = BASE_DIR / "frontend" / "index.html"
FRONTEND_RTSP_DUAL_SCRIPT = BASE_DIR / "frontend" / "rtsp-dual.js"
FRONTEND_DETECTIONS_SCRIPT = BASE_DIR / "frontend" / "detections.js"
FFMPEG_LOG_PATH = UPLOAD_DIR / "ffmpeg.log"
RTSP_HISTORY_PATH = UPLOAD_DIR / "rtsp_stream_history.json"
SHIFT_HISTORY_PATH = UPLOAD_DIR / "bag_shift_history.json"
DETECTION_INDEX_PATH = UPLOAD_DIR / "bag_detection_frames.json"
DETECTION_FRAME_DIR = UPLOAD_DIR / "detections"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
STREAM_NAME = "teststream"
PUBLIC_RTSP_URL = f"rtsp://localhost:8554/{STREAM_NAME}"
INTERNAL_RTSP_URL = f"rtsp://mediamtx:8554/{STREAM_NAME}"
MAX_RTSP_HISTORY_ITEMS = 100
LOCAL_TIMEZONE_NAME = "Asia/Yekaterinburg"
LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_NAME)
DAY_SHIFT_START_HOUR = 7
NIGHT_SHIFT_START_HOUR = 19
MONITOR_POLL_SECONDS = 1.0
DEFAULT_SHIFT_HISTORY_LIMIT = 90
MAX_SHIFT_HISTORY_LIMIT = 365
MAX_SHIFT_HISTORY_EXPORT_ROWS = 5000
DEMO_SHIFT_HISTORY_DAYS = 30
MIN_DEMO_SHIFT_HISTORY_ROWS = 60
RTSP_MONITOR_LIMIT = 2
# Видео в каталоге загрузок определяется по расширению, а не по списку
# исключений: денylist уже трижды принимал за видео служебные файлы
# (индекс кадров, .gitkeep), и публикация падала.
# Контейнер может врать о частоте кадров: записи с этой камеры заявляют
# 60 fps при реальных 15. С -re ffmpeg разгоняется по заявленной частоте,
# перекодирует впятеро больше кадров, чем нужно, забивает процессор и в
# итоге падает с Conversion failed. Поэтому частота измеряется по таймлайну
# контейнера и передаётся ffmpeg явно.
FPS_PROBE_FRAMES = 300
FPS_PROBE_MIN = 1.0
FPS_PROBE_MAX = 120.0
FPS_PROBE_MISMATCH = 1.25

VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".m4v",
    ".webm",
    ".mpg",
    ".mpeg",
    ".ts",
    ".flv",
    ".wmv",
}
MAX_DETECTION_FRAMES = 3000
DETECTION_RETENTION_DAYS = 30
DEFAULT_DETECTION_LIMIT = 300
MAX_DETECTION_LIMIT = 2000
# Frontend assets change with every deploy; without this browsers keep serving a
# stale bundle from heuristic cache and the new UI silently never arrives.
NO_CACHE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


def ensure_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return ensure_datetime(datetime.fromisoformat(value))
    except ValueError:
        return None


def resolve_shift_bucket(observed_at: datetime | None = None) -> tuple[str, str]:
    aware_observed_at = ensure_datetime(observed_at or datetime.now(timezone.utc))
    if aware_observed_at is None:
        aware_observed_at = datetime.now(timezone.utc)

    local_dt = aware_observed_at.astimezone(LOCAL_TIMEZONE)
    if DAY_SHIFT_START_HOUR <= local_dt.hour < NIGHT_SHIFT_START_HOUR:
        return local_dt.date().isoformat(), "day"
    if local_dt.hour >= NIGHT_SHIFT_START_HOUR:
        return local_dt.date().isoformat(), "night"
    return (local_dt.date() - timedelta(days=1)).isoformat(), "night"


def build_shift_schedule_payload() -> dict[str, Any]:
    return {
        "timezone": LOCAL_TIMEZONE_NAME,
        "day_shift": {
            "key": "day",
            "label": "Дневная смена",
            "hours": "07:00-19:00",
        },
        "night_shift": {
            "key": "night",
            "label": "Ночная смена",
            "hours": "19:00-07:00",
        },
    }


def probe_real_fps(path: Path) -> float | None:
    """Реальная частота кадров по таймлайну контейнера.

    Возвращает None, если измерить не удалось или результат неправдоподобен —
    тогда публикация идёт как раньше, по заявленной частоте.
    """
    try:
        import cv2
    except Exception:
        return None

    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        if not capture.grab():
            return None
        started_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        grabbed = 0
        for _ in range(FPS_PROBE_FRAMES):
            if not capture.grab():
                break
            grabbed += 1
        if grabbed < 30:
            return None
        elapsed = (capture.get(cv2.CAP_PROP_POS_MSEC) - started_ms) / 1000.0
        if elapsed <= 0:
            return None
        fps = grabbed / elapsed
        if not (FPS_PROBE_MIN <= fps <= FPS_PROBE_MAX):
            return None
        return round(fps, 3)
    except Exception:
        return None
    finally:
        if capture is not None:
            capture.release()


def is_valid_rtsp_url(value: str) -> bool:
    parsed_url = urlparse((value or "").strip())
    return parsed_url.scheme in {"rtsp", "rtsps"} and bool(parsed_url.netloc)


def default_monitor_url(public_url: str) -> str:
    parsed_url = urlparse((public_url or "").strip())
    if parsed_url.hostname in {"localhost", "127.0.0.1"}:
        netloc = parsed_url.netloc.replace(parsed_url.hostname or "", "host.docker.internal", 1)
        return urlunparse(parsed_url._replace(netloc=netloc))
    return public_url


def validate_date_range(date_from: date_value | None, date_to: date_value | None) -> None:
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError("Дата начала периода не может быть позже даты окончания.")


def normalize_stream_ids(
    values: list[str] | tuple[str, ...] | None,
    *,
    limit: int | None = None,
) -> list[str]:
    normalized_ids: list[str] = []
    for value in values or []:
        for part in str(value or "").split(","):
            normalized_value = part.strip()
            if not normalized_value or normalized_value in normalized_ids:
                continue
            normalized_ids.append(normalized_value)
            if limit is not None and len(normalized_ids) >= max(int(limit), 1):
                return normalized_ids
    return normalized_ids


def build_history_export_filename(
    date_from: date_value | None,
    date_to: date_value | None,
) -> str:
    date_from_label = date_from.isoformat() if date_from else "all"
    date_to_label = date_to.isoformat() if date_to else "all"
    return f"bag_shift_history_{date_from_label}_{date_to_label}.xlsx"


def _excel_column_name(column_number: int) -> str:
    result: list[str] = []
    current = max(int(column_number), 1)
    while current:
        current, remainder = divmod(current - 1, 26)
        result.append(chr(65 + remainder))
    return "".join(reversed(result))


def _excel_inline_cell(
    row_number: int,
    column_number: int,
    value: Any,
    *,
    style_id: int | None = None,
) -> str:
    cell_ref = f"{_excel_column_name(column_number)}{row_number}"
    style_attr = f' s="{style_id}"' if style_id is not None else ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"{style_attr}><v>{value}</v></c>'

    text = "".join(character for character in str(value or "") if character in "\t\n\r" or ord(character) >= 32)
    escaped_text = xml_escape(text)
    return (
        f'<c r="{cell_ref}" t="inlineStr"{style_attr}>'
        f"<is><t xml:space=\"preserve\">{escaped_text}</t></is></c>"
    )


def build_shift_history_workbook(
    rows: list[dict[str, Any]],
    *,
    streams: list[dict[str, str]] | None = None,
    generated_at: datetime | None = None,
) -> bytes:
    workbook_generated_at = ensure_datetime(generated_at or datetime.now(timezone.utc)) or datetime.now(timezone.utc)
    workbook_generated_at = workbook_generated_at.astimezone(timezone.utc).replace(microsecond=0)
    workbook_timestamp = workbook_generated_at.isoformat().replace("+00:00", "Z")

    normalized_streams = [dict(stream) for stream in (streams or []) if str(stream.get("id") or "").strip()]

    table_rows: list[list[Any]]
    merge_ranges: list[str] = []
    header_row_count = 1
    if normalized_streams:
        header_row: list[Any] = ["Дата"]
        for stream in normalized_streams:
            stream_name = str(stream.get("name") or "RTSP-поток").strip() or "RTSP-поток"
            header_row.extend(
                [
                    f"{stream_name} / дневная",
                    f"{stream_name} / ночная",
                    f"{stream_name} / всего",
                ]
            )

        table_rows = [header_row]
        for item in rows:
            row_values: list[Any] = [item.get("date") or ""]
            row_entries = item.get("entries") if isinstance(item.get("entries"), dict) else {}
            for stream in normalized_streams:
                entry = row_entries.get(stream["id"]) if isinstance(row_entries, dict) else {}
                row_values.extend(
                    [
                        int((entry or {}).get("day_count") or 0),
                        int((entry or {}).get("night_count") or 0),
                        int((entry or {}).get("total_count") or 0),
                    ]
                )
            table_rows.append(row_values)

        grouped_header_row: list[Any] = ["\u0414\u0430\u0442\u0430"]
        sub_header_row: list[Any] = [""]
        merge_ranges.append("A1:A2")
        current_column = 2
        for stream_index, stream in enumerate(normalized_streams, start=1):
            stream_name = str(stream.get("name") or "RTSP-\u043f\u043e\u0442\u043e\u043a").strip() or "RTSP-\u043f\u043e\u0442\u043e\u043a"
            stream_label = f"RTSP \u043f\u043e\u0442\u043e\u043a {stream_index}"
            if stream_name:
                stream_label = f"{stream_label}: {stream_name}"
            grouped_header_row.extend([stream_label, "", ""])
            sub_header_row.extend(["\u0414\u043d\u0435\u0432\u043d\u0430\u044f", "\u041d\u043e\u0447\u043d\u0430\u044f", "\u0412\u0441\u0435\u0433\u043e"])
            merge_ranges.append(
                f"{_excel_column_name(current_column)}1:{_excel_column_name(current_column + 2)}1"
            )
            current_column += 3

        table_rows = [grouped_header_row, sub_header_row, *table_rows[1:]]
        header_row_count = 2
    else:
        table_rows = [
            ["Дата", "Дневная смена", "Ночная смена", "Всего"],
        ]
        for item in rows:
            table_rows.append(
                [
                    item.get("date") or "",
                    int(item.get("day_count") or 0),
                    int(item.get("night_count") or 0),
                    int(item.get("total_count") or 0),
                ]
            )

    worksheet_rows: list[str] = []
    for row_index, values in enumerate(table_rows, start=1):
        cells = []
        for column_index, value in enumerate(values, start=1):
            style_id = 1 if row_index <= header_row_count else None
            cells.append(_excel_inline_cell(row_index, column_index, value, style_id=style_id))
        worksheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    last_row_number = max(len(table_rows), 1)
    last_column_number = max(len(table_rows[0]) if table_rows else 1, 1)
    sheet_dimension = f"A1:{_excel_column_name(last_column_number)}{last_row_number}"
    sheet_rows_xml = "".join(worksheet_rows)
    merge_cells_xml = (
        f'<mergeCells count="{len(merge_ranges)}">'
        + "".join(f'<mergeCell ref="{cell_range}"/>' for cell_range in merge_ranges)
        + "</mergeCells>"
        if merge_ranges
        else ""
    )

    first_data_column = min(last_column_number, 2)
    last_data_column = max(first_data_column, last_column_number)
    freeze_top_left = f"A{header_row_count + 1}"
    auto_filter_ref = (
        f"A{header_row_count}:{_excel_column_name(last_column_number)}{last_row_number}"
        if header_row_count > 1
        else sheet_dimension
    )

    worksheet_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<dimension ref=\"{sheet_dimension}\"/>"
        "<sheetViews><sheetView workbookViewId=\"0\">"
        f'<pane ySplit="{header_row_count}" topLeftCell="{freeze_top_left}" activePane="bottomLeft" state="frozen"/>'
        "</sheetView></sheetViews>"
        "<cols>"
        "<col min=\"1\" max=\"1\" width=\"14\" customWidth=\"1\"/>"
        f"<col min=\"{first_data_column}\" max=\"{last_data_column}\" width=\"18\" customWidth=\"1\"/>"
        "</cols>"
        f"<sheetData>{sheet_rows_xml}</sheetData>"
        f"<autoFilter ref=\"{auto_filter_ref}\"/>"
        f"{merge_cells_xml}"
        "</worksheet>"
    )

    workbook_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<sheets><sheet name=\"История смен\" sheetId=\"1\" r:id=\"rId1\"/></sheets>"
        "</workbook>"
    )

    styles_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<styleSheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        "<fonts count=\"2\">"
        "<font><sz val=\"11\"/><name val=\"Calibri\"/><family val=\"2\"/></font>"
        "<font><b/><sz val=\"11\"/><name val=\"Calibri\"/><family val=\"2\"/></font>"
        "</fonts>"
        "<fills count=\"2\">"
        "<fill><patternFill patternType=\"none\"/></fill>"
        "<fill><patternFill patternType=\"gray125\"/></fill>"
        "</fills>"
        "<borders count=\"1\">"
        "<border><left/><right/><top/><bottom/><diagonal/></border>"
        "</borders>"
        "<cellStyleXfs count=\"1\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/>"
        "</cellStyleXfs>"
        "<cellXfs count=\"2\">"
        "<xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\" xfId=\"0\"/>"
        "<xf numFmtId=\"0\" fontId=\"1\" fillId=\"0\" borderId=\"0\" xfId=\"0\" applyFont=\"1\"/>"
        "</cellXfs>"
        "<cellStyles count=\"1\">"
        "<cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/>"
        "</cellStyles>"
        "</styleSheet>"
    )

    content_types_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/docProps/app.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>"
        "<Override PartName=\"/docProps/core.xml\" "
        "ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
        "<Override PartName=\"/xl/workbook.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
        "<Override PartName=\"/xl/worksheets/sheet1.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
        "<Override PartName=\"/xl/styles.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml\"/>"
        "</Types>"
    )

    root_relationships_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"xl/workbook.xml\"/>"
        "<Relationship Id=\"rId2\" "
        "Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" "
        "Target=\"docProps/core.xml\"/>"
        "<Relationship Id=\"rId3\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" "
        "Target=\"docProps/app.xml\"/>"
        "</Relationships>"
    )

    workbook_relationships_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
        "Target=\"worksheets/sheet1.xml\"/>"
        "<Relationship Id=\"rId2\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" "
        "Target=\"styles.xml\"/>"
        "</Relationships>"
    )

    app_properties_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" "
        "xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">"
        "<Application>RTSP Shift History Export</Application>"
        "</Properties>"
    )

    core_properties_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<cp:coreProperties "
        "xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
        "xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
        "xmlns:dcterms=\"http://purl.org/dc/terms/\" "
        "xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" "
        "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">"
        "<dc:title>История мешков по сменам</dc:title>"
        "<dc:creator>RTSP File Stream Tester</dc:creator>"
        "<cp:lastModifiedBy>RTSP File Stream Tester</cp:lastModifiedBy>"
        f"<dcterms:created xsi:type=\"dcterms:W3CDTF\">{workbook_timestamp}</dcterms:created>"
        f"<dcterms:modified xsi:type=\"dcterms:W3CDTF\">{workbook_timestamp}</dcterms:modified>"
        "</cp:coreProperties>"
    )

    workbook_buffer = io.BytesIO()
    with ZipFile(workbook_buffer, "w", compression=ZIP_DEFLATED) as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types_xml)
        workbook_zip.writestr("_rels/.rels", root_relationships_xml)
        workbook_zip.writestr("docProps/app.xml", app_properties_xml)
        workbook_zip.writestr("docProps/core.xml", core_properties_xml)
        workbook_zip.writestr("xl/workbook.xml", workbook_xml)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_relationships_xml)
        workbook_zip.writestr("xl/styles.xml", styles_xml)
        workbook_zip.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    workbook_buffer.seek(0)
    return workbook_buffer.getvalue()


class StreamManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[Any] | None = None
        self.log_handle: Any | None = None
        self.current_file: Path | None = None
        self.started_at: datetime | None = None
        self.last_error: str | None = None

    def ensure_dirs(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        if FFMPEG_LOG_PATH.exists():
            FFMPEG_LOG_PATH.write_text("", encoding="utf-8")

        existing_uploads = sorted(
            [
                candidate
                for candidate in UPLOAD_DIR.iterdir()
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES
            ],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if existing_uploads:
            with self.lock:
                self.current_file = existing_uploads[0]

    def prepare_upload(self, original_filename: str) -> Path:
        self.stop(ignore_missing=True)

        suffix = Path(original_filename or "input.mp4").suffix.lower() or ".mp4"
        target = UPLOAD_DIR / f"input{suffix}"

        with self.lock:
            self._sync_process_state_locked()
            for candidate in UPLOAD_DIR.iterdir():
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
                    candidate.unlink(missing_ok=True)
            self.current_file = None
            self.last_error = None
            FFMPEG_LOG_PATH.write_text("", encoding="utf-8")

        return target

    def set_uploaded_file(self, path: Path) -> None:
        with self.lock:
            self.current_file = path
            self.last_error = None

    def source_path(self) -> Path | None:
        with self.lock:
            if self.current_file is None or not self.current_file.exists():
                return None
            return self.current_file

    def start(self) -> dict[str, Any]:
        with self.lock:
            self._sync_process_state_locked()

            if self.process is not None:
                raise RuntimeError("RTSP-поток уже запущен.")

            if self.current_file is None or not self.current_file.exists():
                raise FileNotFoundError("Перед запуском RTSP-потока загрузите видеофайл.")

            self.last_error = None
            FFMPEG_LOG_PATH.write_text("", encoding="utf-8")
            self.log_handle = FFMPEG_LOG_PATH.open("a", encoding="utf-8")

            input_options: list[str] = []
            declared_fps = 0.0
            real_fps = probe_real_fps(self.current_file)
            if real_fps is not None:
                try:
                    import cv2

                    probe = cv2.VideoCapture(str(self.current_file))
                    declared_fps = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
                    probe.release()
                except Exception:
                    declared_fps = 0.0
                # Переопределяем только при заметном расхождении: если контейнер
                # не врёт, вмешиваться незачем.
                if declared_fps <= 0 or declared_fps / real_fps >= FPS_PROBE_MISMATCH:
                    input_options = ["-r", str(real_fps)]

            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "info",
                "-nostdin",
                "-re",
                "-stream_loop",
                "-1",
                *input_options,
                "-i",
                str(self.current_file),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-ar",
                "48000",
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                INTERNAL_RTSP_URL,
            ]

            try:
                self.process = subprocess.Popen(
                    command,
                    stdout=self.log_handle,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                self._close_log_handle_locked()
                raise

            self.started_at = datetime.now(timezone.utc)

        time.sleep(1)

        with self.lock:
            self._sync_process_state_locked()
            if self.process is None:
                raise RuntimeError(self.last_error or "FFmpeg не смог запустить RTSP-поток.")
            return self._snapshot_locked()

    def stop(self, ignore_missing: bool = False) -> tuple[bool, str]:
        with self.lock:
            self._sync_process_state_locked()
            process = self.process

        if process is None:
            if ignore_missing:
                return False, "RTSP-поток не запущен."
            return False, "RTSP-поток уже остановлен."

        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

        with self.lock:
            self.process = None
            self.started_at = None
            self._close_log_handle_locked()

        return True, "RTSP-поток остановлен."

    def status(self) -> dict[str, Any]:
        with self.lock:
            self._sync_process_state_locked()
            return self._snapshot_locked()

    def read_log_tail(self, max_lines: int = 30) -> str | None:
        if not FFMPEG_LOG_PATH.exists():
            return None

        lines = FFMPEG_LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return None

        return "\n".join(lines[-max_lines:])

    def _sync_process_state_locked(self) -> None:
        if self.process is None:
            return

        return_code = self.process.poll()
        if return_code is None:
            return

        self.process = None
        self.started_at = None
        self._close_log_handle_locked()

        if return_code != 0:
            self.last_error = self.read_log_tail() or f"FFmpeg exited with code {return_code}."

    def _close_log_handle_locked(self) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def _snapshot_locked(self) -> dict[str, Any]:
        uploaded = self.current_file is not None and self.current_file.exists()
        running = self.process is not None and self.process.poll() is None
        size_bytes = None
        if uploaded and self.current_file is not None:
            size_bytes = self.current_file.stat().st_size

        return {
            "uploaded": uploaded,
            "uploaded_file": self.current_file.name if uploaded and self.current_file else None,
            "uploaded_size_bytes": size_bytes,
            "running": running,
            "ffmpeg_pid": self.process.pid if running and self.process else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stream_name": STREAM_NAME,
            "rtsp_url": PUBLIC_RTSP_URL,
            "last_error": self.last_error,
            "ffmpeg_log_tail": self.read_log_tail(),
        }


class RtspStreamCreate(BaseModel):
    name: str
    url: str
    monitor_url: str | None = None


class RtspMonitorStartRequest(BaseModel):
    stream_id: str


class PipelineRtspSaveRequest(BaseModel):
    name: str


class ShiftHistoryUpdateRequest(BaseModel):
    day_count: int
    night_count: int


class ShiftHistoryStreamEntryRequest(BaseModel):
    stream_id: str
    day_count: int
    night_count: int


class ShiftHistoryRowEntriesUpdateRequest(BaseModel):
    entries: list[ShiftHistoryStreamEntryRequest]


class RtspHistoryManager:
    def __init__(self, path: Path, max_items: int = MAX_RTSP_HISTORY_ITEMS) -> None:
        self.path = path
        self.max_items = max(int(max_items), 1)
        self.lock = threading.Lock()
        self.entries: list[dict[str, str]] = []

    def ensure_storage(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load_locked()

    def list(self) -> list[dict[str, str]]:
        with self.lock:
            return [dict(entry) for entry in self.entries]

    def get(self, item_id: str) -> dict[str, str] | None:
        normalized_item_id = (item_id or "").strip()
        if not normalized_item_id:
            return None

        with self.lock:
            for entry in self.entries:
                if entry.get("id") == normalized_item_id:
                    return dict(entry)
        return None

    def add(self, name: str, url: str, monitor_url: str | None = None) -> dict[str, str]:
        normalized_name = " ".join((name or "").split())
        normalized_url = (url or "").strip()
        normalized_monitor_url = (monitor_url or "").strip() or default_monitor_url(normalized_url)

        if not normalized_name:
            raise ValueError("Укажите название RTSP-потока.")
        if len(normalized_name) > 120:
            raise ValueError("Название RTSP-потока должно быть не длиннее 120 символов.")

        if not is_valid_rtsp_url(normalized_url):
            raise ValueError("Укажите корректную RTSP-ссылку, например rtsp://camera.example.com/live.")
        if not is_valid_rtsp_url(normalized_monitor_url):
            raise ValueError("Укажите корректную RTSP-ссылку для мониторинга.")

        entry = {
            "id": uuid4().hex,
            "name": normalized_name,
            "url": normalized_url,
            "monitor_url": normalized_monitor_url,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with self.lock:
            self.entries.insert(0, entry)
            self.entries = self.entries[: self.max_items]
            self._save_locked()

        return dict(entry)

    def delete(self, item_id: str) -> dict[str, str] | None:
        normalized_item_id = (item_id or "").strip()
        if not normalized_item_id:
            return None

        with self.lock:
            for index, entry in enumerate(self.entries):
                if entry.get("id") == normalized_item_id:
                    removed = self.entries.pop(index)
                    self._save_locked()
                    return dict(removed)
        return None

    def _load_locked(self) -> None:
        if not self.path.exists():
            self.entries = []
            self._save_locked()
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.entries = []
            return

        if not isinstance(payload, list):
            payload = []

        normalized_entries: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            entry_name = " ".join(str(item.get("name") or "").split())
            entry_url = str(item.get("url") or "").strip()
            monitor_url = str(item.get("monitor_url") or "").strip() or default_monitor_url(entry_url)
            created_at = str(item.get("created_at") or "").strip()
            if not entry_name or not entry_url or not monitor_url or not created_at:
                continue

            normalized_entries.append(
                {
                    "id": str(item.get("id") or uuid4().hex),
                    "name": entry_name,
                    "url": entry_url,
                    "monitor_url": monitor_url,
                    "created_at": created_at,
                }
            )

        self.entries = normalized_entries[: self.max_items]
        self._save_locked()

    def _save_locked(self) -> None:
        self.path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class BagShiftHistoryManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []

    def ensure_storage(self) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load_locked()

    def record_increment(
        self,
        stream: dict[str, str],
        observed_at: datetime | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        increment = max(int(count), 0)
        if increment <= 0:
            return {}

        timestamp = ensure_datetime(observed_at or datetime.now(timezone.utc))
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        bucket_date, shift_key = resolve_shift_bucket(timestamp)

        with self.lock:
            row = self._find_row_locked(stream_id=stream["id"], bucket_date=bucket_date)
            if row is None:
                row = {
                    "stream_id": stream["id"],
                    "stream_name": stream["name"],
                    "stream_url": stream["url"],
                    "date": bucket_date,
                    "day_count": 0,
                    "night_count": 0,
                    "total_count": 0,
                    "updated_at": timestamp.isoformat(),
                }
                self.rows.append(row)

            row["stream_name"] = stream["name"]
            row["stream_url"] = stream["url"]
            row[f"{shift_key}_count"] = int(row.get(f"{shift_key}_count", 0)) + increment
            row["total_count"] = int(row.get("day_count", 0)) + int(row.get("night_count", 0))
            row["updated_at"] = timestamp.isoformat()
            self._sort_rows_locked()
            self._save_locked()
            return dict(row)

    def list(
        self,
        stream_id: str | None = None,
        limit: int = DEFAULT_SHIFT_HISTORY_LIMIT,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
    ) -> list[dict[str, Any]]:
        normalized_stream_id = (stream_id or "").strip()
        date_from_key = date_from.isoformat() if date_from else None
        date_to_key = date_to.isoformat() if date_to else None

        with self.lock:
            rows: list[dict[str, Any]] = []
            for row in self.rows:
                if normalized_stream_id and row.get("stream_id") != normalized_stream_id:
                    continue

                bucket_date = str(row.get("date") or "").strip()
                if date_from_key and bucket_date < date_from_key:
                    continue
                if date_to_key and bucket_date > date_to_key:
                    continue

                rows.append(dict(row))

        rows.sort(
            key=lambda row: (
                str(row.get("date") or ""),
                str(row.get("updated_at") or ""),
            ),
            reverse=True,
        )
        return rows[: max(int(limit), 1)]

    def ensure_demo_rows(
        self,
        streams: list[dict[str, str]] | None = None,
        *,
        days: int = DEMO_SHIFT_HISTORY_DAYS,
        min_rows: int = MIN_DEMO_SHIFT_HISTORY_ROWS,
    ) -> int:
        normalized_streams: list[dict[str, str]] = []
        for stream in streams or []:
            stream_id = str(stream.get("id") or "").strip()
            stream_name = " ".join(str(stream.get("name") or "").split())
            stream_url = str(stream.get("url") or "").strip()
            if not stream_id or not stream_name or not stream_url:
                continue
            normalized_streams.append(
                {
                    "id": stream_id,
                    "name": stream_name,
                    "url": stream_url,
                }
            )

        if not normalized_streams:
            normalized_streams = [
                {
                    "id": "demo-reagents",
                    "name": "Цех по обработке реагентов",
                    "url": PUBLIC_RTSP_URL,
                },
                {
                    "id": "demo-loading",
                    "name": "Цех по загрузке реагентов",
                    "url": PUBLIC_RTSP_URL,
                },
            ]
        elif len(normalized_streams) == 1:
            normalized_streams.append(
                {
                    "id": "demo-loading",
                    "name": "Цех по загрузке реагентов",
                    "url": PUBLIC_RTSP_URL,
                }
            )

        with self.lock:
            if len(self.rows) >= max(int(min_rows), 1):
                return 0

            added_count = 0
            stream_samples = normalized_streams[:2]
            current_local_date = datetime.now(LOCAL_TIMEZONE).date()

            for day_offset in range(max(int(days), 1) - 1, -1, -1):
                target_date = current_local_date - timedelta(days=day_offset)
                bucket_date = target_date.isoformat()

                for stream_index, stream in enumerate(stream_samples):
                    if self._find_row_locked(stream["id"], bucket_date) is not None:
                        continue

                    day_count = 2 + ((day_offset + stream_index * 2) % 6)
                    night_count = (day_offset * 2 + stream_index) % 5
                    if (day_offset + stream_index) % 4 == 0:
                        night_count += 1

                    updated_local = datetime(
                        target_date.year,
                        target_date.month,
                        target_date.day,
                        19 if night_count == 0 else 23,
                        20 + stream_index * 10,
                        tzinfo=LOCAL_TIMEZONE,
                    )
                    self.rows.append(
                        {
                            "stream_id": stream["id"],
                            "stream_name": stream["name"],
                            "stream_url": stream["url"],
                            "date": bucket_date,
                            "day_count": day_count,
                            "night_count": night_count,
                            "total_count": day_count + night_count,
                            "updated_at": updated_local.astimezone(timezone.utc).isoformat(),
                        }
                    )
                    added_count += 1

            if added_count:
                self._sort_rows_locked()
                self._save_locked()

            return added_count

    def today_summary(self, stream_id: str | None = None) -> dict[str, Any]:
        target_date, _ = resolve_shift_bucket()
        summary = {
            "date": target_date,
            "day_count": 0,
            "night_count": 0,
            "total_count": 0,
        }

        for row in self.list(stream_id=stream_id, limit=500):
            if row.get("date") != target_date:
                continue
            summary["day_count"] += int(row.get("day_count") or 0)
            summary["night_count"] += int(row.get("night_count") or 0)

        summary["total_count"] = summary["day_count"] + summary["night_count"]
        return summary

    def delete_for_stream(self, stream_id: str) -> int:
        normalized_stream_id = (stream_id or "").strip()
        if not normalized_stream_id:
            return 0

        with self.lock:
            before_count = len(self.rows)
            self.rows = [row for row in self.rows if row.get("stream_id") != normalized_stream_id]
            deleted_count = before_count - len(self.rows)
            if deleted_count:
                self._save_locked()
            return deleted_count

    def stream_catalog(self) -> list[dict[str, str]]:
        with self.lock:
            rows = sorted(
                self.rows,
                key=lambda row: (
                    str(row.get("updated_at") or ""),
                    str(row.get("date") or ""),
                ),
                reverse=True,
            )
            stream_map: dict[str, dict[str, str]] = {}
            for row in rows:
                stream_id = str(row.get("stream_id") or "").strip()
                stream_name = " ".join(str(row.get("stream_name") or "").split())
                stream_url = str(row.get("stream_url") or "").strip()
                if not stream_id or not stream_name or not stream_url or stream_id in stream_map:
                    continue
                stream_map[stream_id] = {
                    "id": stream_id,
                    "name": stream_name,
                    "url": stream_url,
                }
            return list(stream_map.values())

    def pivot(
        self,
        streams: list[dict[str, str]],
        *,
        limit: int = DEFAULT_SHIFT_HISTORY_LIMIT,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
    ) -> list[dict[str, Any]]:
        normalized_streams: list[dict[str, str]] = []
        for stream in streams:
            stream_id = str(stream.get("id") or "").strip()
            if not stream_id:
                continue
            normalized_streams.append(
                {
                    "id": stream_id,
                    "name": " ".join(str(stream.get("name") or "").split()) or "RTSP-поток",
                    "url": str(stream.get("url") or "").strip(),
                }
            )

        selected_stream_ids = {stream["id"] for stream in normalized_streams}
        date_from_key = date_from.isoformat() if date_from else None
        date_to_key = date_to.isoformat() if date_to else None

        grouped_rows: dict[str, dict[str, Any]] = {}
        with self.lock:
            for row in self.rows:
                stream_id = str(row.get("stream_id") or "").strip()
                if selected_stream_ids and stream_id not in selected_stream_ids:
                    continue

                bucket_date = str(row.get("date") or "").strip()
                if not bucket_date:
                    continue
                if date_from_key and bucket_date < date_from_key:
                    continue
                if date_to_key and bucket_date > date_to_key:
                    continue

                grouped_row = grouped_rows.setdefault(
                    bucket_date,
                    {
                        "date": bucket_date,
                        "entries": {},
                        "updated_at": str(row.get("updated_at") or ""),
                    },
                )
                grouped_row["entries"][stream_id] = {
                    "stream_id": stream_id,
                    "stream_name": str(row.get("stream_name") or ""),
                    "day_count": int(row.get("day_count") or 0),
                    "night_count": int(row.get("night_count") or 0),
                    "total_count": int(row.get("total_count") or 0),
                }
                updated_at = str(row.get("updated_at") or "")
                if updated_at > str(grouped_row.get("updated_at") or ""):
                    grouped_row["updated_at"] = updated_at

        pivot_rows = list(grouped_rows.values())
        pivot_rows.sort(
            key=lambda row: (
                str(row.get("date") or ""),
                str(row.get("updated_at") or ""),
            ),
            reverse=True,
        )

        for row in pivot_rows:
            entries = row.setdefault("entries", {})
            for stream in normalized_streams:
                entries.setdefault(
                    stream["id"],
                    {
                        "stream_id": stream["id"],
                        "stream_name": stream["name"],
                        "day_count": 0,
                        "night_count": 0,
                        "total_count": 0,
                    },
                )

        return pivot_rows[: max(int(limit), 1)]

    def upsert_entries(
        self,
        bucket_date: str,
        stream_map: dict[str, dict[str, str]],
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_bucket_date = self._normalize_bucket_date(bucket_date)
        if normalized_bucket_date is None:
            raise ValueError("Некорректная дата строки истории.")
        if not entries:
            raise ValueError("Передайте хотя бы один поток для обновления.")

        updated_at = datetime.now(timezone.utc).isoformat()
        updated_rows: list[dict[str, Any]] = []
        with self.lock:
            for entry in entries:
                stream_id = str(entry.get("stream_id") or "").strip()
                stream = stream_map.get(stream_id)
                if stream is None:
                    raise LookupError(f"RTSP-поток {stream_id or 'без id'} не найден.")

                row = self._find_row_locked(stream_id=stream_id, bucket_date=normalized_bucket_date)
                if row is None:
                    row = {
                        "stream_id": stream_id,
                        "stream_name": str(stream.get("name") or "").strip() or "RTSP-поток",
                        "stream_url": str(stream.get("url") or "").strip(),
                        "date": normalized_bucket_date,
                        "day_count": 0,
                        "night_count": 0,
                        "total_count": 0,
                        "updated_at": updated_at,
                    }
                    self.rows.append(row)

                row["stream_name"] = str(stream.get("name") or "").strip() or "RTSP-поток"
                row["stream_url"] = str(stream.get("url") or "").strip()
                row["day_count"] = max(int(entry.get("day_count") or 0), 0)
                row["night_count"] = max(int(entry.get("night_count") or 0), 0)
                row["total_count"] = int(row["day_count"]) + int(row["night_count"])
                row["updated_at"] = updated_at
                updated_rows.append(dict(row))

            self._sort_rows_locked()
            self._save_locked()

        return updated_rows

    def delete_rows(
        self,
        bucket_date: str,
        *,
        stream_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_bucket_date = self._normalize_bucket_date(bucket_date)
        if normalized_bucket_date is None:
            return []

        selected_stream_ids = set(normalize_stream_ids(stream_ids))
        removed_rows: list[dict[str, Any]] = []
        with self.lock:
            remaining_rows: list[dict[str, Any]] = []
            for row in self.rows:
                row_date = str(row.get("date") or "").strip()
                row_stream_id = str(row.get("stream_id") or "").strip()
                should_remove = row_date == normalized_bucket_date and (
                    not selected_stream_ids or row_stream_id in selected_stream_ids
                )
                if should_remove:
                    removed_rows.append(dict(row))
                else:
                    remaining_rows.append(row)

            if removed_rows:
                self.rows = remaining_rows
                self._save_locked()

        return removed_rows

    def _find_row_locked(self, stream_id: str, bucket_date: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row.get("stream_id") == stream_id and row.get("date") == bucket_date:
                return row
        return None

    def _normalize_bucket_date(self, value: Any) -> str | None:
        normalized_value = str(value or "").strip()
        if not normalized_value:
            return None
        try:
            return date_value.fromisoformat(normalized_value).isoformat()
        except ValueError:
            return None

    def _sort_rows_locked(self) -> None:
        self.rows.sort(
            key=lambda row: (
                str(row.get("date") or ""),
                str(row.get("updated_at") or ""),
            ),
            reverse=True,
        )

    def _load_locked(self) -> None:
        if not self.path.exists():
            self.rows = []
            self._save_locked()
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.rows = []
            return

        if not isinstance(payload, list):
            payload = []

        normalized_rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            stream_id = str(item.get("stream_id") or "").strip()
            stream_name = " ".join(str(item.get("stream_name") or "").split())
            stream_url = str(item.get("stream_url") or "").strip()
            bucket_date = str(item.get("date") or "").strip()
            updated_at = str(item.get("updated_at") or "").strip()
            if not stream_id or not stream_name or not stream_url or not bucket_date:
                continue

            day_count = max(int(item.get("day_count") or 0), 0)
            night_count = max(int(item.get("night_count") or 0), 0)
            normalized_rows.append(
                {
                    "stream_id": stream_id,
                    "stream_name": stream_name,
                    "stream_url": stream_url,
                    "date": bucket_date,
                    "day_count": day_count,
                    "night_count": night_count,
                    "total_count": day_count + night_count,
                    "updated_at": updated_at or datetime.now(timezone.utc).isoformat(),
                }
            )

        self.rows = normalized_rows
        self._save_locked()

    def _save_locked(self) -> None:
        self.path.write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class AggregatedBagShiftHistoryManager(BagShiftHistoryManager):
    def record_increment(
        self,
        stream: dict[str, str],
        observed_at: datetime | None = None,
        count: int = 1,
    ) -> dict[str, Any]:
        del stream
        increment = max(int(count), 0)
        if increment <= 0:
            return {}

        timestamp = ensure_datetime(observed_at or datetime.now(timezone.utc))
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        bucket_date, shift_key = resolve_shift_bucket(timestamp)

        with self.lock:
            row = self._find_row_locked(bucket_date)
            if row is None:
                row = {
                    "date": bucket_date,
                    "day_count": 0,
                    "night_count": 0,
                    "total_count": 0,
                    "updated_at": timestamp.isoformat(),
                }
                self.rows.append(row)

            row[f"{shift_key}_count"] = int(row.get(f"{shift_key}_count", 0)) + increment
            row["total_count"] = int(row.get("day_count") or 0) + int(row.get("night_count") or 0)
            row["updated_at"] = timestamp.isoformat()
            self._sort_rows_locked()
            self._save_locked()
            return dict(row)

    def list(
        self,
        stream_id: str | None = None,
        limit: int = DEFAULT_SHIFT_HISTORY_LIMIT,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
    ) -> list[dict[str, Any]]:
        del stream_id
        date_from_key = date_from.isoformat() if date_from else None
        date_to_key = date_to.isoformat() if date_to else None

        with self.lock:
            rows = []
            for row in self.rows:
                bucket_date = str(row.get("date") or "").strip()
                if date_from_key and bucket_date < date_from_key:
                    continue
                if date_to_key and bucket_date > date_to_key:
                    continue
                rows.append(dict(row))

        rows.sort(
            key=lambda row: (
                str(row.get("date") or ""),
                str(row.get("updated_at") or ""),
            ),
            reverse=True,
        )
        return rows[: max(int(limit), 1)]

    def ensure_demo_rows(
        self,
        streams: list[dict[str, str]] | None = None,
        *,
        days: int = DEMO_SHIFT_HISTORY_DAYS,
        min_rows: int = MIN_DEMO_SHIFT_HISTORY_ROWS,
    ) -> int:
        del streams
        with self.lock:
            if len(self.rows) >= max(int(min_rows), 1):
                return 0

            added_count = 0
            current_local_date = datetime.now(LOCAL_TIMEZONE).date()
            for day_offset in range(max(int(days), 1) - 1, -1, -1):
                target_date = current_local_date - timedelta(days=day_offset)
                bucket_date = target_date.isoformat()
                if self._find_row_locked(bucket_date) is not None:
                    continue

                day_count = 2 + (day_offset % 6)
                night_count = (day_offset * 2) % 5
                if day_offset % 4 == 0:
                    night_count += 1

                updated_local = datetime(
                    target_date.year,
                    target_date.month,
                    target_date.day,
                    19 if night_count == 0 else 23,
                    20,
                    tzinfo=LOCAL_TIMEZONE,
                )
                self.rows.append(
                    {
                        "date": bucket_date,
                        "day_count": day_count,
                        "night_count": night_count,
                        "total_count": day_count + night_count,
                        "updated_at": updated_local.astimezone(timezone.utc).isoformat(),
                    }
                )
                added_count += 1

            if added_count:
                self._sort_rows_locked()
                self._save_locked()

            return added_count

    def today_summary(self, stream_id: str | None = None) -> dict[str, Any]:
        del stream_id
        target_date, _ = resolve_shift_bucket()
        summary = {
            "date": target_date,
            "day_count": 0,
            "night_count": 0,
            "total_count": 0,
        }

        row = next((item for item in self.list(limit=500) if item.get("date") == target_date), None)
        if row is not None:
            summary["day_count"] = int(row.get("day_count") or 0)
            summary["night_count"] = int(row.get("night_count") or 0)

        summary["total_count"] = summary["day_count"] + summary["night_count"]
        return summary

    def delete_for_stream(self, stream_id: str) -> int:
        del stream_id
        return 0

    def update_row(self, bucket_date: str, day_count: int, night_count: int) -> dict[str, Any]:
        normalized_bucket_date = self._normalize_bucket_date(bucket_date)
        if normalized_bucket_date is None:
            raise ValueError("Некорректная дата строки истории.")

        with self.lock:
            row = self._find_row_locked(normalized_bucket_date)
            if row is None:
                raise LookupError("Строка истории не найдена.")

            row["day_count"] = max(int(day_count), 0)
            row["night_count"] = max(int(night_count), 0)
            row["total_count"] = int(row["day_count"]) + int(row["night_count"])
            row["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._sort_rows_locked()
            self._save_locked()
            return dict(row)

    def delete_row(self, bucket_date: str) -> dict[str, Any] | None:
        normalized_bucket_date = self._normalize_bucket_date(bucket_date)
        if normalized_bucket_date is None:
            return None

        with self.lock:
            for index, row in enumerate(self.rows):
                if row.get("date") == normalized_bucket_date:
                    removed_row = self.rows.pop(index)
                    self._save_locked()
                    return dict(removed_row)
        return None

    def _find_row_locked(self, bucket_date: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row.get("date") == bucket_date:
                return row
        return None

    def _normalize_bucket_date(self, value: Any) -> str | None:
        normalized_value = str(value or "").strip()
        if not normalized_value:
            return None
        try:
            return date_value.fromisoformat(normalized_value).isoformat()
        except ValueError:
            return None

    def _sort_rows_locked(self) -> None:
        self.rows.sort(
            key=lambda row: (
                str(row.get("date") or ""),
                str(row.get("updated_at") or ""),
            ),
            reverse=True,
        )

    def _load_locked(self) -> None:
        if not self.path.exists():
            self.rows = []
            self._save_locked()
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.rows = []
            return

        if not isinstance(payload, list):
            payload = []

        aggregated_rows: dict[str, dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue

            bucket_date = self._normalize_bucket_date(item.get("date"))
            if bucket_date is None:
                continue

            updated_at = str(item.get("updated_at") or "").strip() or datetime.now(timezone.utc).isoformat()
            day_count = max(int(item.get("day_count") or 0), 0)
            night_count = max(int(item.get("night_count") or 0), 0)

            row = aggregated_rows.setdefault(
                bucket_date,
                {
                    "date": bucket_date,
                    "day_count": 0,
                    "night_count": 0,
                    "total_count": 0,
                    "updated_at": updated_at,
                },
            )
            row["day_count"] += day_count
            row["night_count"] += night_count
            if updated_at > str(row.get("updated_at") or ""):
                row["updated_at"] = updated_at

        self.rows = list(aggregated_rows.values())
        for row in self.rows:
            row["total_count"] = int(row.get("day_count") or 0) + int(row.get("night_count") or 0)
        self._sort_rows_locked()
        self._save_locked()


class DetectionFrameArchive:
    def __init__(
        self,
        index_path: Path,
        frame_dir: Path,
        max_frames: int = MAX_DETECTION_FRAMES,
        retention_days: int = DETECTION_RETENTION_DAYS,
    ) -> None:
        self.index_path = index_path
        self.frame_dir = frame_dir
        self.max_frames = max(int(max_frames), 1)
        self.retention_days = max(int(retention_days), 1)
        self.lock = threading.Lock()
        self.items: list[dict[str, Any]] = []
        self.loaded = False

    def ensure_storage(self) -> None:
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self._load_locked()

    def record(self, stream: dict[str, str], capture: Any) -> dict[str, Any] | None:
        captured_at = ensure_datetime(getattr(capture, "captured_at", None)) or datetime.now(timezone.utc)
        local_dt = captured_at.astimezone(LOCAL_TIMEZONE)
        local_date = local_dt.date().isoformat()
        shift_date, shift = resolve_shift_bucket(captured_at)

        item_id = uuid4().hex
        relative_name = f"{local_date}/{item_id}.jpg"
        target = self.frame_dir / local_date / f"{item_id}.jpg"

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(getattr(capture, "image_jpeg", b"")))
        except Exception:
            return None

        entry = {
            "id": item_id,
            "stream_id": str(stream.get("id") or "").strip(),
            "stream_name": str(stream.get("name") or "").strip() or "RTSP-поток",
            "captured_at": captured_at.isoformat(),
            "local_date": local_date,
            "local_time": local_dt.strftime("%H:%M:%S"),
            "shift_date": shift_date,
            "shift": shift,
            "bag_index": int(getattr(capture, "bag_index", 0) or 0),
            "fill": round(float(getattr(capture, "fill", 0.0) or 0.0), 4),
            "frame_position": int(getattr(capture, "frame_position", 0) or 0),
            "filename": relative_name,
            "width": int(getattr(capture, "image_width", 0) or 0),
            "height": int(getattr(capture, "image_height", 0) or 0),
        }

        with self.lock:
            self._load_locked()
            self.items.append(entry)
            self._sort_locked()
            self._prune_locked()
            self._save_locked()

        return dict(entry)

    def list(
        self,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
        stream_ids: list[str] | None = None,
        limit: int = DEFAULT_DETECTION_LIMIT,
    ) -> list[dict[str, Any]]:
        normalized_ids = {value for value in (stream_ids or []) if value}
        bounded_limit = min(max(int(limit), 1), MAX_DETECTION_LIMIT)

        with self.lock:
            self._load_locked()
            selected: list[dict[str, Any]] = []
            for item in self.items:
                local_date = item.get("local_date")
                if not isinstance(local_date, str):
                    continue
                try:
                    parsed = date_value.fromisoformat(local_date)
                except ValueError:
                    continue
                if date_from is not None and parsed < date_from:
                    continue
                if date_to is not None and parsed > date_to:
                    continue
                if normalized_ids and item.get("stream_id") not in normalized_ids:
                    continue
                selected.append(dict(item))

        return selected[:bounded_limit]

    def grouped(
        self,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
        stream_ids: list[str] | None = None,
        limit: int = DEFAULT_DETECTION_LIMIT,
    ) -> dict[str, Any]:
        items = self.list(
            date_from=date_from,
            date_to=date_to,
            stream_ids=stream_ids,
            limit=limit,
        )

        groups: dict[str, dict[str, Any]] = {}
        for item in items:
            local_date = str(item.get("local_date") or "")
            group = groups.get(local_date)
            if group is None:
                group = {"date": local_date, "total": 0, "day_count": 0, "night_count": 0, "items": []}
                groups[local_date] = group
            group["items"].append(item)
            group["total"] += 1
            if item.get("shift") == "night":
                group["night_count"] += 1
            else:
                group["day_count"] += 1

        ordered = sorted(groups.values(), key=lambda group: group["date"], reverse=True)
        return {
            "groups": ordered,
            "total": len(items),
            "streams": self.stream_catalog(),
            "retention_days": self.retention_days,
            "max_frames": self.max_frames,
        }

    def stream_catalog(self) -> list[dict[str, str]]:
        with self.lock:
            self._load_locked()
            seen: dict[str, str] = {}
            for item in self.items:
                stream_id = str(item.get("stream_id") or "")
                if not stream_id or stream_id in seen:
                    continue
                seen[stream_id] = str(item.get("stream_name") or stream_id)

        return [{"id": key, "name": value} for key, value in seen.items()]

    def clear(self, stream_id: str | None = None) -> int:
        normalized = (stream_id or "").strip()
        with self.lock:
            self._load_locked()
            kept: list[dict[str, Any]] = []
            removed = 0
            for item in self.items:
                if normalized and item.get("stream_id") != normalized:
                    kept.append(item)
                    continue
                self._delete_file_locked(item)
                removed += 1
            self.items = kept
            self._save_locked()

        return removed

    def _delete_file_locked(self, item: dict[str, Any]) -> None:
        filename = str(item.get("filename") or "")
        if not filename:
            return
        candidate = self.frame_dir / filename
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            return

    def _prune_locked(self) -> None:
        cutoff = (datetime.now(timezone.utc).astimezone(LOCAL_TIMEZONE).date()
                  - timedelta(days=self.retention_days))
        survivors: list[dict[str, Any]] = []
        for item in self.items:
            local_date = item.get("local_date")
            try:
                parsed = date_value.fromisoformat(str(local_date))
            except ValueError:
                parsed = None
            if parsed is not None and parsed < cutoff:
                self._delete_file_locked(item)
                continue
            survivors.append(item)

        if len(survivors) > self.max_frames:
            for item in survivors[self.max_frames:]:
                self._delete_file_locked(item)
            survivors = survivors[: self.max_frames]

        self.items = survivors

    def _sort_locked(self) -> None:
        self.items.sort(key=lambda item: str(item.get("captured_at") or ""), reverse=True)

    def _load_locked(self) -> None:
        if self.loaded:
            return
        self.loaded = True

        if not self.index_path.exists():
            self.items = []
            return

        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.items = []
            return

        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            self.items = []
            return

        self.items = [item for item in items if isinstance(item, dict) and item.get("filename")]
        self._sort_locked()

    def _save_locked(self) -> None:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(
                json.dumps({"items": self.items}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return



class RtspMonitorManager:
    def __init__(
        self,
        analytics: BagAnalyticsManager,
        registry: RtspHistoryManager,
        history: BagShiftHistoryManager,
    ) -> None:
        self.analytics = analytics
        self.registry = registry
        self.history = history
        self.lock = threading.Lock()
        self.active_stream: dict[str, str] | None = None
        self.started_at: datetime | None = None
        self.last_recorded_count = 0
        self.last_completed_count = 0
        self.collector_thread: threading.Thread | None = None
        self.collector_stop_event: threading.Event | None = None

    def start(self, stream_id: str) -> dict[str, Any]:
        stream = self.registry.get(stream_id)
        if stream is None:
            raise ValueError("Выберите RTSP-поток из реестра.")

        self.stop(ignore_missing=True)
        self.analytics.reset(wait=True, clear_result=True)
        self.analytics.start(stream.get("monitor_url") or stream["url"])

        stop_event = threading.Event()
        collector_thread = threading.Thread(
            target=self._collector_loop,
            args=(dict(stream), stop_event),
            daemon=True,
        )

        with self.lock:
            self.active_stream = dict(stream)
            self.started_at = datetime.now(timezone.utc)
            self.last_recorded_count = 0
            self.collector_thread = collector_thread
            self.collector_stop_event = stop_event

        collector_thread.start()
        return self.status()

    def stop(self, ignore_missing: bool = False) -> tuple[bool, str]:
        with self.lock:
            active_stream = dict(self.active_stream) if self.active_stream else None
            collector_thread = self.collector_thread
            stop_event = self.collector_stop_event

        if active_stream is None:
            if ignore_missing:
                return False, "Мониторинг RTSP-потока не запущен."
            return False, "Мониторинг RTSP-потока уже остановлен."

        if stop_event is not None:
            stop_event.set()
        if collector_thread is not None and collector_thread.is_alive():
            collector_thread.join(timeout=3)

        snapshot = self.analytics.status()
        self._flush_snapshot_to_history(stream=active_stream, snapshot=snapshot)

        bag_count = snapshot.get("bag_count")
        if isinstance(bag_count, int):
            with self.lock:
                self.last_completed_count = bag_count

        self.analytics.reset(wait=True, clear_result=True)

        with self.lock:
            self.active_stream = None
            self.started_at = None
            self.last_recorded_count = 0
            self.collector_thread = None
            self.collector_stop_event = None

        return True, "Мониторинг RTSP-потока остановлен."

    def shutdown(self) -> None:
        self.stop(ignore_missing=True)
        self.analytics.shutdown()

    def status(self) -> dict[str, Any]:
        analytics_snapshot = self.analytics.status()

        with self.lock:
            active_stream = dict(self.active_stream) if self.active_stream else None
            started_at = self.started_at.isoformat() if self.started_at else None
            last_completed_count = self.last_completed_count

        today_summary = self.history.today_summary(stream_id=active_stream["id"] if active_stream else None)
        return {
            "active": active_stream is not None,
            "stream": active_stream,
            "started_at": started_at,
            "last_completed_bag_count": last_completed_count,
            "today_summary": today_summary,
            "shift_schedule": build_shift_schedule_payload(),
            "analytics": analytics_snapshot,
        }

    def history_payload(
        self,
        stream_id: str | None = None,
        limit: int = DEFAULT_SHIFT_HISTORY_LIMIT,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
    ) -> dict[str, Any]:
        normalized_stream_id = (stream_id or "").strip() or None
        return {
            "items": self.history.list(
                stream_id=normalized_stream_id,
                limit=limit,
                date_from=date_from,
                date_to=date_to,
            ),
            "stream": self.registry.get(normalized_stream_id) if normalized_stream_id else None,
            "shift_schedule": build_shift_schedule_payload(),
        }

    def _collector_loop(self, stream: dict[str, str], stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            snapshot = self.analytics.status()
            self._flush_snapshot_to_history(stream=stream, snapshot=snapshot)
            stop_event.wait(MONITOR_POLL_SECONDS)

    def _flush_snapshot_to_history(self, stream: dict[str, str], snapshot: dict[str, Any]) -> None:
        bag_count = snapshot.get("bag_count")
        if not isinstance(bag_count, int):
            return

        with self.lock:
            if self.active_stream is not None and self.active_stream.get("id") != stream.get("id"):
                return
            last_recorded_count = self.last_recorded_count

        if bag_count <= last_recorded_count:
            return

        delta = bag_count - last_recorded_count
        observed_at = parse_iso_datetime(snapshot.get("last_result_at")) or datetime.now(timezone.utc)
        self.history.record_increment(stream=stream, observed_at=observed_at, count=delta)

        with self.lock:
            if self.active_stream is None or self.active_stream.get("id") != stream.get("id"):
                return
            self.last_recorded_count = bag_count


class DualRtspMonitorManager:
    def __init__(
        self,
        registry: RtspHistoryManager,
        history: BagShiftHistoryManager,
        model_path: Path,
        monitor_limit: int = RTSP_MONITOR_LIMIT,
        detections: DetectionFrameArchive | None = None,
    ) -> None:
        self.registry = registry
        self.history = history
        self.model_path = model_path
        self.detections = detections
        self.monitor_limit = max(int(monitor_limit), 1)
        self.lock = threading.Lock()
        self.sessions: dict[str, dict[str, Any]] = {}

    def lookup_stream(self, stream_id: str) -> dict[str, str] | None:
        normalized_stream_id = (stream_id or "").strip()
        if not normalized_stream_id:
            return None

        with self.lock:
            session = self.sessions.get(normalized_stream_id)
            if session is not None:
                return dict(session["stream"])

        stream = self.registry.get(normalized_stream_id)
        if stream is not None:
            return stream

        for history_stream in self.history.stream_catalog():
            if history_stream.get("id") == normalized_stream_id:
                return dict(history_stream)
        return None

    def resolve_stream_map(self, stream_ids: list[str] | tuple[str, ...]) -> dict[str, dict[str, str]]:
        stream_map: dict[str, dict[str, str]] = {}
        for stream_id in normalize_stream_ids(list(stream_ids)):
            stream = self.lookup_stream(stream_id)
            if stream is None:
                raise LookupError(f"RTSP-поток {stream_id or 'без id'} не найден.")
            stream_map[stream_id] = {
                "id": str(stream.get("id") or stream_id).strip(),
                "name": " ".join(str(stream.get("name") or "").split()) or "RTSP-поток",
                "url": str(stream.get("url") or "").strip(),
                "monitor_url": str(stream.get("monitor_url") or "").strip(),
            }
        return stream_map

    def resolve_history_streams(
        self,
        requested_stream_ids: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, str]]:
        selected_streams: list[dict[str, str]] = []
        seen_stream_ids: set[str] = set()
        normalized_requested_ids = normalize_stream_ids(requested_stream_ids, limit=self.monitor_limit)

        if normalized_requested_ids:
            candidate_streams = [self.lookup_stream(stream_id) for stream_id in normalized_requested_ids]
        else:
            with self.lock:
                active_streams = [
                    dict(session["stream"])
                    for session in sorted(
                        self.sessions.values(),
                        key=lambda session: str(session.get("started_at") or ""),
                    )
                ]
            candidate_streams = active_streams + self.registry.list() + self.history.stream_catalog()

        for stream in candidate_streams:
            if not isinstance(stream, dict):
                continue
            stream_id = str(stream.get("id") or "").strip()
            stream_name = " ".join(str(stream.get("name") or "").split())
            stream_url = str(stream.get("url") or stream.get("stream_url") or "").strip()
            if not stream_id or not stream_name or not stream_url or stream_id in seen_stream_ids:
                continue

            selected_streams.append(
                {
                    "id": stream_id,
                    "name": stream_name,
                    "url": stream_url,
                }
            )
            seen_stream_ids.add(stream_id)
            if len(selected_streams) >= self.monitor_limit:
                break

        return selected_streams

    def is_active(self, stream_id: str) -> bool:
        normalized_stream_id = (stream_id or "").strip()
        if not normalized_stream_id:
            return False
        with self.lock:
            return normalized_stream_id in self.sessions

    def start(self, stream_id: str) -> dict[str, Any]:
        normalized_stream_id = (stream_id or "").strip()
        stream = self.registry.get(normalized_stream_id)
        if stream is None:
            raise ValueError("Выберите RTSP-поток из реестра.")

        with self.lock:
            if normalized_stream_id in self.sessions:
                return self.status()
            if len(self.sessions) >= self.monitor_limit:
                raise ValueError(
                    f"Можно одновременно анализировать не более {self.monitor_limit} RTSP-потоков."
                )

        analytics_instance = BagAnalyticsManager(model_path=self.model_path)
        if self.detections is not None:
            archive = self.detections
            captured_stream = dict(stream)
            analytics_instance.set_detection_sink(
                lambda capture: archive.record(stream=captured_stream, capture=capture)
            )

        try:
            analytics_instance.start(stream.get("monitor_url") or stream["url"])
        except Exception:
            analytics_instance.shutdown()
            raise

        stop_event = threading.Event()
        session = {
            "stream": dict(stream),
            "analytics": analytics_instance,
            "started_at": datetime.now(timezone.utc),
            "last_recorded_count": 0,
            "last_completed_count": 0,
            "collector_stop_event": stop_event,
            "collector_thread": None,
        }
        collector_thread = threading.Thread(
            target=self._collector_loop,
            args=(normalized_stream_id, stop_event),
            daemon=True,
        )
        session["collector_thread"] = collector_thread

        with self.lock:
            self.sessions[normalized_stream_id] = session

        collector_thread.start()
        return self.status()

    def stop(
        self,
        stream_id: str | None = None,
        *,
        ignore_missing: bool = False,
    ) -> tuple[bool, str]:
        normalized_stream_id = (stream_id or "").strip() or None

        if normalized_stream_id is not None:
            with self.lock:
                session = self.sessions.pop(normalized_stream_id, None)
            if session is None:
                if ignore_missing:
                    return False, "Мониторинг RTSP-потока не запущен."
                return False, "Мониторинг RTSP-потока уже остановлен."

            self._stop_session(session)
            stream_name = str(session["stream"].get("name") or "без названия").strip() or "без названия"
            return True, f"Мониторинг RTSP-потока «{stream_name}» остановлен."

        with self.lock:
            sessions = list(self.sessions.values())
            self.sessions = {}

        if not sessions:
            if ignore_missing:
                return False, "Мониторинг RTSP-потоков не запущен."
            return False, "Мониторинг RTSP-потоков уже остановлен."

        for session in sessions:
            self._stop_session(session)
        return True, "Мониторинг всех RTSP-потоков остановлен."

    def shutdown(self) -> None:
        self.stop(ignore_missing=True)

    def status(self) -> dict[str, Any]:
        with self.lock:
            sessions = sorted(
                self.sessions.values(),
                key=lambda session: str(session.get("started_at") or ""),
            )

        monitors = [self._build_session_payload(session) for session in sessions]
        primary_monitor = monitors[0] if monitors else None
        empty_summary = {
            "date": resolve_shift_bucket()[0],
            "day_count": 0,
            "night_count": 0,
            "total_count": 0,
        }

        return {
            "active": bool(monitors),
            "active_count": len(monitors),
            "monitor_limit": self.monitor_limit,
            "available_slots": max(self.monitor_limit - len(monitors), 0),
            "monitors": monitors,
            "stream": primary_monitor.get("stream") if primary_monitor else None,
            "started_at": primary_monitor.get("started_at") if primary_monitor else None,
            "last_completed_bag_count": (
                primary_monitor.get("last_completed_bag_count") if primary_monitor else 0
            ),
            "today_summary": primary_monitor.get("today_summary") if primary_monitor else empty_summary,
            "shift_schedule": build_shift_schedule_payload(),
            "analytics": primary_monitor.get("analytics") if primary_monitor else None,
        }

    def history_payload(
        self,
        *,
        stream_ids: list[str] | tuple[str, ...] | None = None,
        limit: int = DEFAULT_SHIFT_HISTORY_LIMIT,
        date_from: date_value | None = None,
        date_to: date_value | None = None,
    ) -> dict[str, Any]:
        selected_streams = self.resolve_history_streams(stream_ids)
        return {
            "items": self.history.pivot(
                selected_streams,
                limit=limit,
                date_from=date_from,
                date_to=date_to,
            ),
            "streams": selected_streams,
            "shift_schedule": build_shift_schedule_payload(),
        }

    def _build_session_payload(self, session: dict[str, Any]) -> dict[str, Any]:
        stream = dict(session["stream"])
        analytics_snapshot = session["analytics"].status()
        return {
            "stream": stream,
            "started_at": (
                session["started_at"].isoformat()
                if isinstance(session.get("started_at"), datetime)
                else None
            ),
            "last_completed_bag_count": int(session.get("last_completed_count") or 0),
            "today_summary": self.history.today_summary(stream_id=stream.get("id")),
            "analytics": analytics_snapshot,
        }

    def _collector_loop(self, stream_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self.lock:
                session = self.sessions.get(stream_id)
            if session is None:
                return

            snapshot = session["analytics"].status()
            self._flush_snapshot_to_history(session=session, snapshot=snapshot)
            stop_event.wait(MONITOR_POLL_SECONDS)

    def _stop_session(self, session: dict[str, Any]) -> None:
        stop_event = session.get("collector_stop_event")
        collector_thread = session.get("collector_thread")
        analytics_instance = session.get("analytics")

        if isinstance(stop_event, threading.Event):
            stop_event.set()
        if isinstance(collector_thread, threading.Thread) and collector_thread.is_alive():
            collector_thread.join(timeout=3)

        if not isinstance(analytics_instance, BagAnalyticsManager):
            return

        snapshot = analytics_instance.status()
        self._flush_snapshot_to_history(session=session, snapshot=snapshot, allow_inactive=True)
        bag_count = snapshot.get("bag_count")
        if isinstance(bag_count, int):
            session["last_completed_count"] = bag_count
        analytics_instance.shutdown()

    def _flush_snapshot_to_history(
        self,
        *,
        session: dict[str, Any],
        snapshot: dict[str, Any],
        allow_inactive: bool = False,
    ) -> None:
        bag_count = snapshot.get("bag_count")
        if not isinstance(bag_count, int):
            return

        stream = dict(session.get("stream") or {})
        stream_id = str(stream.get("id") or "").strip()
        if not stream_id:
            return

        with self.lock:
            current_session = self.sessions.get(stream_id)
            if current_session is None:
                if not allow_inactive:
                    return
                current_session = session
            elif current_session is not session:
                return
            last_recorded_count = int(current_session.get("last_recorded_count") or 0)

        if bag_count <= last_recorded_count:
            return

        delta = bag_count - last_recorded_count
        observed_at = parse_iso_datetime(snapshot.get("last_result_at")) or datetime.now(timezone.utc)
        self.history.record_increment(stream=stream, observed_at=observed_at, count=delta)

        with self.lock:
            current_session = self.sessions.get(stream_id)
            if current_session is None:
                if allow_inactive:
                    session["last_recorded_count"] = bag_count
                return
            if current_session is not session:
                return
            current_session["last_recorded_count"] = bag_count


manager = StreamManager()
analytics = BagAnalyticsManager(model_path=MODEL_PATH)
rtsp_history = RtspHistoryManager(path=RTSP_HISTORY_PATH)
bag_shift_history = BagShiftHistoryManager(path=SHIFT_HISTORY_PATH)
detection_archive = DetectionFrameArchive(
    index_path=DETECTION_INDEX_PATH,
    frame_dir=DETECTION_FRAME_DIR,
)
rtsp_monitor = DualRtspMonitorManager(
    registry=rtsp_history,
    history=bag_shift_history,
    model_path=MODEL_PATH,
    detections=detection_archive,
)


def build_response(message: str | None = None) -> dict[str, Any]:
    payload = {
        **manager.status(),
        "analytics": analytics.status(),
    }
    if message is not None:
        payload["message"] = message
    return payload


def build_rtsp_monitor_response(message: str | None = None) -> dict[str, Any]:
    payload = rtsp_monitor.status()
    if message is not None:
        payload["message"] = message
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.ensure_dirs()
    rtsp_history.ensure_storage()
    bag_shift_history.ensure_storage()
    bag_shift_history.ensure_demo_rows(rtsp_history.list())
    detection_archive.ensure_storage()
    try:
        yield
    finally:
        analytics.shutdown()
        rtsp_monitor.shutdown()
        manager.stop(ignore_missing=True)


app = FastAPI(
    title="RTSP File Stream Tester",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_INDEX, headers=NO_CACHE_HEADERS)


@app.get("/rtsp-dual.js", include_in_schema=False)
async def rtsp_dual_script() -> FileResponse:
    return FileResponse(
        FRONTEND_RTSP_DUAL_SCRIPT,
        media_type="application/javascript",
        headers=NO_CACHE_HEADERS,
    )


@app.get("/status")
@app.get("/api/status")
async def status() -> dict[str, Any]:
    return build_response()


@app.get("/rtsp-streams")
@app.get("/api/rtsp-streams")
async def list_rtsp_streams() -> dict[str, Any]:
    return {"items": rtsp_history.list()}


@app.post("/rtsp-streams")
@app.post("/api/rtsp-streams")
async def create_rtsp_stream(payload: RtspStreamCreate) -> dict[str, Any]:
    try:
        item = rtsp_history.add(payload.name, payload.url, monitor_url=payload.monitor_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": f"RTSP-поток «{item['name']}» добавлен в историю.",
        "item": item,
        "items": rtsp_history.list(),
    }


@app.post("/rtsp-streams/from-pipeline")
@app.post("/api/rtsp-streams/from-pipeline")
async def save_pipeline_rtsp_stream(payload: PipelineRtspSaveRequest) -> dict[str, Any]:
    stream_status = manager.status()
    if not stream_status["running"]:
        raise HTTPException(status_code=400, detail="Сначала запустите тестовый RTSP-поток из видео.")

    try:
        item = rtsp_history.add(
            payload.name,
            PUBLIC_RTSP_URL,
            monitor_url=INTERNAL_RTSP_URL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": f"Тестовый RTSP-поток «{item['name']}» сохранён в историю.",
        "item": item,
        "items": rtsp_history.list(),
    }


@app.delete("/rtsp-streams/{stream_id}")
@app.delete("/api/rtsp-streams/{stream_id}")
async def delete_rtsp_stream(stream_id: str) -> dict[str, Any]:
    if rtsp_monitor.is_active(stream_id):
        rtsp_monitor.stop(stream_id=stream_id, ignore_missing=True)

    removed_item = rtsp_history.delete(stream_id)
    if removed_item is None:
        raise HTTPException(status_code=404, detail="RTSP-поток не найден в истории.")

    deleted_history_rows = bag_shift_history.delete_for_stream(stream_id)
    return {
        "message": f"RTSP-поток «{removed_item['name']}» удалён из истории.",
        "item": removed_item,
        "deleted_shift_rows": deleted_history_rows,
        "items": rtsp_history.list(),
        "shift_history": rtsp_monitor.history_payload(limit=90),
    }


@app.get("/rtsp-monitor/status")
@app.get("/api/rtsp-monitor/status")
async def rtsp_monitor_status() -> dict[str, Any]:
    return build_rtsp_monitor_response()


@app.post("/rtsp-monitor/start")
@app.post("/api/rtsp-monitor/start")
async def start_rtsp_monitor(payload: RtspMonitorStartRequest) -> dict[str, Any]:
    try:
        snapshot = rtsp_monitor.start(payload.stream_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось запустить мониторинг RTSP-потока: {exc}") from exc

    stream_name = None
    for item in snapshot.get("monitors") or []:
        stream = item.get("stream") if isinstance(item, dict) else None
        if isinstance(stream, dict) and stream.get("id") == payload.stream_id:
            stream_name = stream.get("name")
            break
    if not stream_name and isinstance(snapshot.get("stream"), dict):
        stream_name = snapshot.get("stream", {}).get("name")
    return build_rtsp_monitor_response(
        message=f"Мониторинг RTSP-потока «{stream_name or 'без названия'}» запущен."
    )


@app.post("/rtsp-monitor/stop")
@app.post("/api/rtsp-monitor/stop")
async def stop_rtsp_monitor(stream_id: str | None = Query(default=None)) -> dict[str, Any]:
    stopped, message = rtsp_monitor.stop(stream_id=stream_id, ignore_missing=False)
    if not stopped:
        raise HTTPException(status_code=409, detail=message)
    return build_rtsp_monitor_response(message=message)


@app.get("/rtsp-monitor/history")
@app.get("/api/rtsp-monitor/history")
async def rtsp_monitor_history(
    stream_id: str | None = Query(default=None),
    stream_ids: list[str] | None = Query(default=None),
    limit: int = Query(default=DEFAULT_SHIFT_HISTORY_LIMIT, ge=1, le=MAX_SHIFT_HISTORY_LIMIT),
    date_from: date_value | None = Query(default=None),
    date_to: date_value | None = Query(default=None),
) -> dict[str, Any]:
    try:
        validate_date_range(date_from, date_to)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return rtsp_monitor.history_payload(
        stream_ids=normalize_stream_ids([stream_id, *(stream_ids or [])], limit=RTSP_MONITOR_LIMIT),
        limit=limit,
        date_from=date_from,
        date_to=date_to,
    )


@app.patch("/rtsp-monitor/history/{bucket_date}")
@app.patch("/api/rtsp-monitor/history/{bucket_date}")
async def update_rtsp_monitor_history_row(
    bucket_date: str,
    payload: ShiftHistoryRowEntriesUpdateRequest,
) -> dict[str, Any]:
    selected_stream_ids = [entry.stream_id for entry in payload.entries]
    try:
        updated_rows = bag_shift_history.upsert_entries(
            bucket_date=bucket_date,
            stream_map=rtsp_monitor.resolve_stream_map(selected_stream_ids),
            entries=[
                entry.model_dump() if hasattr(entry, "model_dump") else entry.dict()
                for entry in payload.entries
            ],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "message": f"Строка истории за {bucket_date} обновлена.",
        "items_updated": updated_rows,
        **rtsp_monitor.history_payload(
            stream_ids=selected_stream_ids,
            limit=DEFAULT_SHIFT_HISTORY_LIMIT,
        ),
    }


@app.delete("/rtsp-monitor/history/{bucket_date}")
@app.delete("/api/rtsp-monitor/history/{bucket_date}")
async def delete_rtsp_monitor_history_row(
    bucket_date: str,
    stream_id: str | None = Query(default=None),
    stream_ids: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    selected_stream_ids = normalize_stream_ids([stream_id, *(stream_ids or [])], limit=RTSP_MONITOR_LIMIT)
    removed_rows = bag_shift_history.delete_rows(bucket_date=bucket_date, stream_ids=selected_stream_ids)
    if not removed_rows:
        raise HTTPException(status_code=404, detail="Строка истории не найдена.")

    return {
        "message": f"Строка истории за {bucket_date} удалена.",
        "items_removed": removed_rows,
        **rtsp_monitor.history_payload(
            stream_ids=selected_stream_ids,
            limit=DEFAULT_SHIFT_HISTORY_LIMIT,
        ),
    }


@app.get("/rtsp-monitor/history/export")
@app.get("/api/rtsp-monitor/history/export")
async def export_rtsp_monitor_history(
    stream_id: str | None = Query(default=None),
    stream_ids: list[str] | None = Query(default=None),
    date_from: date_value | None = Query(default=None),
    date_to: date_value | None = Query(default=None),
) -> StreamingResponse:
    try:
        validate_date_range(date_from, date_to)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history_payload = rtsp_monitor.history_payload(
        stream_ids=normalize_stream_ids([stream_id, *(stream_ids or [])], limit=RTSP_MONITOR_LIMIT),
        limit=MAX_SHIFT_HISTORY_EXPORT_ROWS,
        date_from=date_from,
        date_to=date_to,
    )
    workbook_bytes = build_shift_history_workbook(
        history_payload["items"],
        streams=history_payload.get("streams"),
    )
    filename = build_history_export_filename(date_from=date_from, date_to=date_to)

    return StreamingResponse(
        io.BytesIO(workbook_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.post("/upload")
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Перед загрузкой выберите видеофайл.")

    analytics.reset()
    target = manager.prepare_upload(file.filename)

    try:
        with target.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {exc}") from exc
    finally:
        await file.close()

    manager.set_uploaded_file(target)
    return build_response(f"Файл {file.filename} загружен.")


@app.post("/start")
@app.post("/api/start")
async def start_stream() -> dict[str, Any]:
    try:
        manager.start()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось запустить RTSP-поток: {exc}") from exc

    analytics.reset(wait=False, clear_result=True)
    return build_response("RTSP-поток запущен. Теперь его можно сохранить в историю и запустить анализ.")


@app.post("/analyze")
@app.post("/api/analyze")
async def analyze_stream() -> dict[str, Any]:
    stream_status = manager.status()
    if not stream_status["running"]:
        raise HTTPException(status_code=400, detail="Сначала запустите RTSP-поток, потом аналитику.")

    try:
        analytics.start(INTERNAL_RTSP_URL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Не удалось запустить аналитику мешков: {exc}") from exc

    return build_response("Аналитика мешков по RTSP перезапущена.")


@app.post("/stop")
@app.post("/api/stop")
async def stop_stream() -> dict[str, Any]:
    analytics.reset()
    stopped, message = manager.stop()
    if not stopped:
        raise HTTPException(status_code=409, detail=message)

    return build_response(message)


@app.get("/detections.js", include_in_schema=False)
async def detections_script() -> FileResponse:
    return FileResponse(
        FRONTEND_DETECTIONS_SCRIPT,
        media_type="application/javascript",
        headers=NO_CACHE_HEADERS,
    )


@app.get("/detections")
@app.get("/api/detections")
async def list_detections(
    date_from: date_value | None = Query(default=None),
    date_to: date_value | None = Query(default=None),
    stream_id: list[str] | None = Query(default=None),
    limit: int = Query(default=DEFAULT_DETECTION_LIMIT, ge=1, le=MAX_DETECTION_LIMIT),
) -> dict[str, Any]:
    try:
        validate_date_range(date_from, date_to)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return detection_archive.grouped(
        date_from=date_from,
        date_to=date_to,
        stream_ids=stream_id,
        limit=limit,
    )


@app.delete("/detections")
@app.delete("/api/detections")
async def clear_detections(stream_id: str | None = Query(default=None)) -> dict[str, Any]:
    removed = detection_archive.clear(stream_id=stream_id)
    return {"removed": removed, "message": f"Удалено кадров: {removed}."}


@app.get("/detections/image/{bucket_date}/{filename}", include_in_schema=False)
@app.get("/api/detections/image/{bucket_date}/{filename}", include_in_schema=False)
async def detection_image(bucket_date: str, filename: str) -> FileResponse:
    safe_date = Path(bucket_date).name
    safe_name = Path(filename).name
    target = DETECTION_FRAME_DIR / safe_date / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Кадр не найден.")

    return FileResponse(target, media_type="image/jpeg")
