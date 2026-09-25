# Доставка правки без перезаливки образа

Образ весит 648 МБ, и гнать его целиком ради изменения в коде долго. Compose
умеет подменять отдельный файл внутри контейнера свежей версией с диска —
этого достаточно для правок в логике подсчёта.

## Как применить

Всё делается в папке, где лежит `docker-compose.yml` (обычно `soda-bag-counter`).

1. Скачайте в эту папку два файла:

   - `docker-compose.override.yml` — отсюда же, из `artifacts/hotfix/`
   - `analytics.py` — из `backend/app/analytics.py` в корне репозитория

   Прямые ссылки:

   ```
   https://raw.githubusercontent.com/SerovaVladislava/soda-bag-counter/main/artifacts/hotfix/docker-compose.override.yml
   https://raw.githubusercontent.com/SerovaVladislava/soda-bag-counter/main/backend/app/analytics.py
   ```

2. Примените:

   ```bash
   docker compose up -d
   ```

   Compose подхватит `docker-compose.override.yml` автоматически — указывать
   его в команде не нужно.

3. Проверьте, что правка на месте:

   ```bash
   docker compose exec backend python -c "from app.analytics import ZONE_MIN_PRESENT_SECONDS; print(ZONE_MIN_PRESENT_SECONDS)"
   ```

   Команда обращается к сервису, а не к имени контейнера, поэтому работает и
   на старой установке (контейнер `rtsp-backend`), и на новой, где Compose
   именует контейнеры по проекту (`rtsp-bag-counter-backend-1`).

   Должно напечатать `10.0`. Если печатает `3.5` — файл не подхватился,
   проверьте, что `analytics.py` лежит рядом с `docker-compose.yml`.

4. После `docker compose up -d` контейнер пересоздаётся, и мониторинг
   останавливается — он не восстанавливается сам. Откройте интерфейс,
   раскройте «Управление RTSP-потоками» и нажмите «Анализировать».

## Журнал проб для калибровки

Живой цикл пишет каждую пробу в `data/uploads/zone_samples.csv`: время,
заполнение зоны, контрольная зона, базовая линия, решение, состояние.
Файл ограничен 25 МБ (старый переименовывается в `.1`). Если счёт
расходится с реальностью, пришлите этот файл — по нему видно, что именно
видела система в момент выгрузки.

   Заодно проверьте, что поток не замер:

   ```bash
   docker compose exec backend python -c "from app.analytics import STALL_SECONDS; print(STALL_SECONDS)"
   ```

   Должно напечатать `90.0` — столько секунд неподвижной картинки система
   считает признаком зависшего источника и сообщает об этом в интерфейсе.

## Как откатиться

Удалите `docker-compose.override.yml` и `analytics.py`, затем `docker compose up -d`.
Контейнер вернётся к коду из образа.

## Когда так делать не стоит

Подмена файла — временная мера между релизами. Если правок накопилось много
или менялись зависимости, лучше собрать новый образ: подменённый файл должен
оставаться совместимым с остальным кодом внутри образа.
