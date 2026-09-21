# RTSP File Stream Tester

Минимальный локальный проект для тестирования RTSP-потока из видеофайла.

Что делает проект:

- загружает видео через веб-интерфейс;
- запускает FFmpeg и публикует поток в MediaMTX;
- автоматически запускает аналитику по RTSP-потоку в реальном времени через YOLO-модель `best.pt`;
- считает количество мешков на видео и показывает результат в UI и API;
- отдает фиксированный RTSP URL: `rtsp://localhost:8554/teststream`;
- крутит файл по кругу;
- пишет логи FFmpeg и показывает их в UI и через `/api/status`.

## Состав проекта

- `docker-compose.yml` — запуск backend и MediaMTX;
- `backend` — FastAPI + управление процессом FFmpeg;
- `models/best.pt` — локальная модель для детекции мешков;
- `frontend` — простой HTML/JS интерфейс;
- `mediamtx/mediamtx.yml` — конфиг MediaMTX;
- `data/uploads` — загруженный файл и лог `ffmpeg.log`.

## Запуск

Требования:

- Docker Desktop или Docker Engine с Compose plugin;
- свободные порты `18000` и `8554`.

Старт:

```bash
docker compose up --build
```

После запуска:

- веб-интерфейс: `http://localhost:18000`
- RTSP URL: `rtsp://localhost:8554/teststream`

Первый билд теперь заметно тяжелее, потому что внутри backend ставится `Ultralytics + Torch` для аналитики.

Остановить проект:

```bash
docker compose down
```

## Как использовать

1. Откройте `http://localhost:18000`.
2. Нажмите `Upload video` и выберите файл.
3. Нажмите `Start RTSP`.
4. Backend автоматически стартует аналитику по RTSP и через несколько секунд покажет количество мешков.
5. При необходимости нажмите `Run analytics`, чтобы пересчитать результат заново.
6. Заберите URL `rtsp://localhost:8554/teststream` и вставьте его в другой проект.
7. Чтобы остановить публикацию, нажмите `Stop RTSP`.

## Проверка RTSP

### VLC

1. Откройте VLC.
2. Выберите `Media` → `Open Network Stream`.
3. Вставьте:

```text
rtsp://localhost:8554/teststream
```

### ffplay

```bash
ffplay -rtsp_transport tcp rtsp://localhost:8554/teststream
```

`MediaMTX` в этом проекте настроен только на RTSP-over-TCP, поэтому `ffplay` лучше запускать именно с `-rtsp_transport tcp`.

## API

Доступны оба варианта маршрутов: короткие (`/upload`, `/start`, `/stop`, `/status`) и такие же с префиксом `/api`.

- `POST /upload` или `POST /api/upload` — загрузка файла (`multipart/form-data`, поле `file`)
- `POST /start` или `POST /api/start` — запуск FFmpeg, публикации в RTSP и автоматический запуск live-аналитики по RTSP-потоку
- `POST /analyze` или `POST /api/analyze` — повторный запуск live-аналитики по уже запущенному RTSP-потоку
- `POST /stop` или `POST /api/stop` — остановка FFmpeg
- `GET /status` или `GET /api/status` — текущий статус RTSP и аналитики, RTSP URL и хвост лога FFmpeg

Пример ответа `/api/status`:

```json
{
  "uploaded": true,
  "uploaded_file": "input.mp4",
  "uploaded_size_bytes": 1234567,
  "running": true,
  "ffmpeg_pid": 14,
  "started_at": "2026-03-26T10:15:30.000000+00:00",
  "stream_name": "teststream",
  "rtsp_url": "rtsp://localhost:8554/teststream",
  "last_error": null,
  "ffmpeg_log_tail": "...",
  "analytics": {
    "enabled": true,
    "state": "done",
    "message": "Detected 3 bag(s) from 12 sampled frames.",
    "bag_count": 3,
    "max_bag_count": 3,
    "frames_processed": 12,
    "sample_counts": [3, 3, 3, 3, 3],
    "last_error": null
  }
}
```

## Что уже учтено

- backend умеет останавливать предыдущий FFmpeg-процесс;
- поток запускается в цикле через `-stream_loop -1`;
- видео всегда публикуется как `H.264`, аудио как `AAC`, что упрощает совместимость;
- YOLO-модель из `models/best.pt` загружается внутри backend и считает мешки по живому RTSP-потоку;
- результат аналитики отдается через API и показывается в веб-интерфейсе;
- ошибки FFmpeg попадают в `data/uploads/ffmpeg.log`;
- frontend показывает текущий статус и лог.

## Полезные команды

Посмотреть логи контейнеров:

```bash
docker compose logs -f
```

Посмотреть только backend:

```bash
docker compose logs -f backend
```

Перезапустить backend:

```bash
docker compose restart backend
```

## Примечание для Windows

Если Docker Desktop на Windows начнет сбоить из-за пути проекта с не-ASCII символами, перенесите проект в каталог с ASCII-именем, например `C:\projects\rtsp-stream-test`, и запустите снова.
