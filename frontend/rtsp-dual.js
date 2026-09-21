(() => {
  const replaceNode = (node) => {
    if (!node) return null;
    const clone = node.cloneNode(true);
    node.replaceWith(clone);
    return clone;
  };

  const byId = (id) => document.getElementById(id);
  const rtspTabButton = byId("rtspTabButton");
  const rtspPanel = byId("rtspPanel");
  const monitorGrid = byId("monitorStatePill")?.closest(".grid.two");

  const rtspForm = replaceNode(byId("rtspForm"));
  const rtspNameInput = byId("rtspNameInput");
  const rtspUrlInput = byId("rtspUrlInput");
  const addRtspButton = byId("addRtspButton");
  const refreshRtspButton = byId("refreshRtspButton");
  const rtspMessage = byId("rtspMessage");
  const rtspHistoryEmpty = byId("rtspHistoryEmpty");
  const rtspHistoryTableWrap = byId("rtspHistoryTableWrap");
  const rtspHistoryBody = replaceNode(byId("rtspHistoryBody"));

  const shiftScheduleHint = byId("shiftScheduleHint");
  const shiftDateFromInput = byId("shiftDateFromInput");
  const shiftDateToInput = byId("shiftDateToInput");
  const applyShiftHistoryFilterButton = replaceNode(byId("applyShiftHistoryFilterButton"));
  const exportShiftHistoryButton = replaceNode(byId("exportShiftHistoryButton"));
  const shiftHistoryMessage = byId("shiftHistoryMessage");
  const shiftHistoryEmpty = byId("shiftHistoryEmpty");
  const shiftHistoryTableWrap = byId("shiftHistoryTableWrap");
  const shiftHistoryBody = replaceNode(byId("shiftHistoryBody"));
  const shiftHistoryHeadRow = document.querySelector("#shiftHistoryTableWrap thead");

  let rtspStreams = [];
  let monitorStatusPayload = null;
  let shiftHistoryItems = [];
  let historyStreams = [];
  let editingShiftDate = "";

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const formatDateTime = (value) => {
    if (!value) return "-";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ru-RU");
  };

  const formatCount = (value) =>
    Number.isFinite(Number(value)) ? String(Math.max(Number(value), 0)) : "0";

  const setMessage = (element, text, tone = "") => {
    if (!element) return;
    element.textContent = text || "";
    element.className = "message";
    if (tone) element.classList.add(tone);
  };

  const toIsoDate = (value) => {
    if (!(value instanceof Date) || Number.isNaN(value.getTime())) return "";
    const adjustedDate = new Date(value.getTime() - value.getTimezoneOffset() * 60000);
    return adjustedDate.toISOString().slice(0, 10);
  };

  const getSelectedHistoryStreams = () =>
    Array.isArray(historyStreams) ? historyStreams.filter((stream) => stream && stream.id) : [];

  const getDisplayHistoryStreams = () => {
    const streams = getSelectedHistoryStreams().slice(0, 2);
    while (streams.length < 2) {
      streams.push({ id: "", name: `Поток ${streams.length + 1}`, placeholder: true });
    }
    return streams;
  };

  const getShiftHistoryFilters = () => ({
    dateFrom: shiftDateFromInput.value || "",
    dateTo: shiftDateToInput.value || "",
  });

  const validateShiftHistoryFilters = () => {
    const { dateFrom, dateTo } = getShiftHistoryFilters();
    if (dateFrom && dateTo && dateFrom > dateTo) {
      throw new Error("Дата начала периода не может быть позже даты окончания.");
    }
    return { dateFrom, dateTo };
  };

  const buildShiftHistoryParams = () => {
    const { dateFrom, dateTo } = validateShiftHistoryFilters();
    const params = new URLSearchParams({ limit: "90" });
    if (dateFrom) params.set("date_from", dateFrom);
    if (dateTo) params.set("date_to", dateTo);
    getSelectedHistoryStreams().forEach((stream) => params.append("stream_ids", stream.id));
    return params;
  };

  const fetchJson = async (url, options = {}) => {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || data.message || "Ошибка запроса.");
    }
    return data;
  };

  const renderShiftSchedule = (schedule) => {
    if (!shiftScheduleHint) return;
    if (!schedule) {
      shiftScheduleHint.textContent =
        "Смены считаются по времени площадки: дневная 07:00-19:00, ночная 19:00-07:00.";
      return;
    }

    shiftScheduleHint.textContent =
      `Смены считаются по времени ${schedule.timezone}: ` +
      `${schedule.day_shift?.label || "Дневная"} ${schedule.day_shift?.hours || "07:00-19:00"}, ` +
      `${schedule.night_shift?.label || "Ночная"} ${schedule.night_shift?.hours || "19:00-07:00"}.`;
  };

  const resolveMonitorPill = (monitor) => {
    const analytics = monitor?.analytics || {};
    if (!monitor) return { text: "Свободно", tone: "warn" };
    if (analytics.state === "running") return { text: "Считает мешки", tone: "ok" };
    if (analytics.state === "waiting") return { text: "Ожидание потока", tone: "warn" };
    if (analytics.state === "error") return { text: "Ошибка", tone: "bad" };
    if (analytics.state === "done") return { text: "Готово", tone: "ok" };
    return { text: "Ожидание", tone: "warn" };
  };

  const renderRtspHistory = (items) => {
    const activeIds = new Set(
      (monitorStatusPayload?.monitors || []).map((monitor) => monitor?.stream?.id || "").filter(Boolean)
    );
    const hasItems = Array.isArray(items) && items.length > 0;
    rtspHistoryEmpty.hidden = hasItems;
    rtspHistoryTableWrap.hidden = !hasItems;

    if (!hasItems) {
      rtspHistoryBody.innerHTML = "";
      return;
    }

    rtspHistoryBody.innerHTML = items
      .map((item) => {
        const isActive = activeIds.has(item.id);
        return `
          <tr>
            <td>${escapeHtml(item.name || "-")}</td>
            <td>
              <a class="table-link mono" href="${escapeHtml(item.url || "#")}" target="_blank" rel="noreferrer">
                ${escapeHtml(item.url || "-")}
              </a>
            </td>
            <td>${escapeHtml(formatDateTime(item.created_at))}</td>
            <td>
              <div class="actions">
                <button
                  class="secondary"
                  type="button"
                  data-action="analyze"
                  data-stream-id="${escapeHtml(item.id || "")}"
                  ${isActive ? "disabled" : ""}
                >${isActive ? "В работе" : "Анализировать"}</button>
                <button class="secondary" type="button" data-action="delete" data-stream-id="${escapeHtml(item.id || "")}">Удалить</button>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  };

  const loadRtspHistory = async () => {
    const data = await fetchJson("/api/rtsp-streams");
    rtspStreams = data.items || [];
    renderRtspHistory(rtspStreams);
    return rtspStreams;
  };

  const buildMonitorCard = (monitor, index) => {
    const slotNumber = index + 1;

    if (!monitor) {
      return `
        <article class="card monitor-card">
          <div class="section-head">
            <div class="monitor-title">
              <span class="monitor-slot">Монитор ${slotNumber}</span>
              <span class="monitor-name">Слот свободен</span>
            </div>
            <span class="pill" data-state="warn"><span class="live-dot"></span>Свободно</span>
          </div>
          <div class="monitor-empty">
            Откройте «Управление RTSP-потоками» вверху страницы и нажмите «Анализировать»
            напротив нужного потока.
          </div>
        </article>
      `;
    }

    const pill = resolveMonitorPill(monitor);
    const stream = monitor.stream || {};
    const analytics = monitor.analytics || {};
    const summary = monitor.today_summary || {};
    const details =
      analytics.message || analytics.last_error || "Мониторинг запущен, ожидаются данные от RTSP-потока.";
    const bagCount = Number.isInteger(analytics.bag_count) ? String(analytics.bag_count) : "-";
    const frames = Number.isInteger(analytics.frames_processed)
      ? Number(analytics.frames_processed).toLocaleString("ru-RU")
      : "-";

    return `
      <article class="card monitor-card">
        <div class="section-head">
          <div class="monitor-title">
            <span class="monitor-slot">Монитор ${slotNumber}</span>
            <span class="monitor-name">${escapeHtml(stream.name || "Без названия")}</span>
          </div>
          <span class="pill" data-state="${escapeHtml(pill.tone)}"><span class="live-dot"></span>${escapeHtml(pill.text)}</span>
        </div>

        <span class="monitor-url mono">${escapeHtml(stream.url || "-")}</span>

        <div class="monitor-headline">
          <span class="value-label">Посчитано с начала анализа</span>
          <div class="summary-number">${escapeHtml(bagCount)}</div>
        </div>

        <div class="monitor-shifts">
          <span class="shift-chip">Дневная<b>${escapeHtml(formatCount(summary.day_count))}</b></span>
          <span class="shift-chip">Ночная<b>${escapeHtml(formatCount(summary.night_count))}</b></span>
          <span class="shift-chip">Всего<b>${escapeHtml(formatCount(summary.total_count))}</b></span>
        </div>

        <div class="monitor-facts">
          <span>Обработано кадров: <b>${escapeHtml(frames)}</b></span>
          <span>Запущен: <b>${escapeHtml(formatDateTime(monitor.started_at))}</b></span>
          <span>Последнее обновление: <b>${escapeHtml(formatDateTime(analytics.last_result_at))}</b></span>
        </div>

        <p class="monitor-note">${escapeHtml(details)}</p>

        <div class="actions">
          <button class="secondary" type="button" data-action="stop-monitor" data-stream-id="${escapeHtml(stream.id || "")}">Остановить</button>
        </div>
      </article>
    `;
  };

  let setupToggledByUser = false;
  let syncingSetup = false;

  // Assigning `open` fires a toggle event of its own, so the flag has to be
  // suppressed while we do it - otherwise the first automatic open would look
  // like an operator action and disable the rule for good.
  const syncSetupDisclosure = (monitors) => {
    const setup = byId("streamSetup");
    if (!setup || setupToggledByUser) return;

    const shouldOpen = (monitors || []).filter(Boolean).length === 0;
    if (setup.open === shouldOpen) return;

    syncingSetup = true;
    setup.open = shouldOpen;
    window.setTimeout(() => {
      syncingSetup = false;
    }, 0);
  };

  const renderShiftSummary = (monitors, limit) => {
    const summaryDay = byId("summaryDayCount");
    const summaryNight = byId("summaryNightCount");
    const summaryTotal = byId("summaryTotalCount");
    const summaryDate = byId("summaryBusinessDate");
    const summaryPill = byId("summaryActivePill");
    if (!summaryDay || !summaryNight || !summaryTotal) return;

    const active = (monitors || []).filter(Boolean);
    let day = 0;
    let night = 0;
    let total = 0;
    let businessDate = "";

    active.forEach((monitor) => {
      const summary = monitor.today_summary || {};
      day += Number(summary.day_count) || 0;
      night += Number(summary.night_count) || 0;
      total += Number(summary.total_count) || 0;
      if (!businessDate && summary.date) businessDate = String(summary.date);
    });

    summaryDay.textContent = String(day);
    summaryNight.textContent = String(night);
    summaryTotal.textContent = String(total);
    if (summaryDate) summaryDate.textContent = businessDate || "-";

    if (summaryPill) {
      const running = active.filter((monitor) => (monitor.analytics || {}).state === "running").length;
      if (!active.length) {
        summaryPill.textContent = "Нет активных потоков";
        summaryPill.dataset.state = "warn";
      } else {
        summaryPill.textContent = `Активно ${running || active.length} из ${limit}`;
        summaryPill.dataset.state = running ? "ok" : "warn";
      }
    }
  };

  const renderMonitorStatus = (payload) => {
    monitorStatusPayload = payload || { monitor_limit: 2, monitors: [] };
    const limit = Math.max(Number(monitorStatusPayload.monitor_limit) || 2, 2);
    const monitors = Array.isArray(monitorStatusPayload.monitors) ? monitorStatusPayload.monitors : [];

    if (monitorGrid) {
      monitorGrid.innerHTML = Array.from({ length: limit }, (_, index) =>
        buildMonitorCard(monitors[index] || null, index)
      ).join("");
    }

    renderShiftSummary(monitors, limit);
    syncSetupDisclosure(monitors);
    renderShiftSchedule(monitorStatusPayload.shift_schedule || null);
    renderRtspHistory(rtspStreams);
  };

  const renderShiftHistoryHead = () => {
    if (!shiftHistoryHeadRow) return;
    const groupCells = ['<th rowspan="2">Дата</th>'];
    const subCells = [];
    getDisplayHistoryStreams().forEach((stream, index) => {
      const fallbackName = `RTSP поток ${index + 1}`;
      const streamLabel =
        stream.name && !stream.placeholder ? `${fallbackName}: ${stream.name}` : fallbackName;
      groupCells.push(`<th colspan="3">${escapeHtml(streamLabel)}</th>`);
      subCells.push("<th>Дневная</th>");
      subCells.push("<th>Ночная</th>");
      subCells.push('<th class="history-total-column subtle">Всего</th>');
    });
    groupCells.push('<th rowspan="2">Действия</th>');
    shiftHistoryHeadRow.innerHTML = `
      <tr>${groupCells.join("")}</tr>
      <tr>${subCells.join("")}</tr>
    `;
  };

  const getHistoryEntry = (item, streamId) => {
    const entry = item?.entries?.[streamId];
    return {
      day_count: Math.max(Number(entry?.day_count) || 0, 0),
      night_count: Math.max(Number(entry?.night_count) || 0, 0),
      total_count: Math.max(Number(entry?.total_count) || 0, 0),
    };
  };

  const renderShiftHistory = (items) => {
    renderShiftHistoryHead();
    const hasItems = Array.isArray(items) && items.length > 0;
    shiftHistoryItems = hasItems ? [...items] : [];
    shiftHistoryEmpty.hidden = hasItems;
    shiftHistoryTableWrap.hidden = !hasItems;

    if (!hasItems) {
      shiftHistoryBody.innerHTML = "";
      return;
    }

    const streams = getDisplayHistoryStreams();
    shiftHistoryBody.innerHTML = items
      .map((item) => {
        const cells = streams
          .map((stream) => {
            const entry = getHistoryEntry(item, stream.id);
            if (editingShiftDate === item.date && !stream.placeholder) {
              return `
                <td><input type="number" min="0" step="1" data-stream-id="${escapeHtml(stream.id)}" data-field="day_count" value="${escapeHtml(String(entry.day_count))}" /></td>
                <td><input type="number" min="0" step="1" data-stream-id="${escapeHtml(stream.id)}" data-field="night_count" value="${escapeHtml(String(entry.night_count))}" /></td>
                <td class="history-total-column value-cell">${escapeHtml(String(entry.total_count))}</td>
              `;
            }
            if (editingShiftDate === item.date && stream.placeholder) {
              return '<td>-</td><td>-</td><td class="history-total-column value-cell">-</td>';
            }
            return `
              <td>${escapeHtml(String(entry.day_count))}</td>
              <td>${escapeHtml(String(entry.night_count))}</td>
              <td class="history-total-column value-cell">${escapeHtml(String(entry.total_count))}</td>
            `;
          })
          .join("");

        return `
          <tr>
            <td>${escapeHtml(item.date || "-")}</td>
            ${cells}
            <td>
              <div class="actions">
                ${
                  editingShiftDate === item.date
                    ? `<button class="secondary" type="button" data-action="save-row" data-date="${escapeHtml(item.date || "")}">Сохранить</button>
                       <button class="secondary" type="button" data-action="cancel-row" data-date="${escapeHtml(item.date || "")}">Отмена</button>`
                    : `<button class="secondary" type="button" data-action="edit-row" data-date="${escapeHtml(item.date || "")}">Редактировать</button>
                       <button class="secondary" type="button" data-action="delete-row" data-date="${escapeHtml(item.date || "")}">Удалить</button>`
                }
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  };

  const refreshMonitorStatus = async ({ quiet = false } = {}) => {
    try {
      const data = await fetchJson("/api/rtsp-monitor/status");
      renderMonitorStatus(data);
      return data;
    } catch (error) {
      if (!quiet) setMessage(rtspMessage, error.message, "error");
      throw error;
    }
  };

  const refreshShiftHistory = async ({ quiet = false } = {}) => {
    try {
      const params = buildShiftHistoryParams();
      const data = await fetchJson(`/api/rtsp-monitor/history?${params.toString()}`);
      historyStreams = Array.isArray(data.streams) ? data.streams.slice(0, 2) : [];
      if (editingShiftDate && !(data.items || []).some((item) => item.date === editingShiftDate)) {
        editingShiftDate = "";
      }
      renderShiftHistory(data.items || []);
      renderShiftSchedule(data.shift_schedule || null);
      return data;
    } catch (error) {
      if (!quiet) setMessage(shiftHistoryMessage, error.message, "error");
      throw error;
    }
  };

  const exportShiftHistory = async () => {
    const params = buildShiftHistoryParams();
    params.delete("limit");
    const response = await fetch(`/api/rtsp-monitor/history/export?${params.toString()}`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || data.message || "Ошибка выгрузки.");
    }
    const blob = await response.blob();
    const contentDisposition = response.headers.get("Content-Disposition") || "";
    const filenameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
    const filename = filenameMatch?.[1] || "bag_shift_history.xlsx";
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(downloadUrl);
    return filename;
  };

  const updateShiftHistoryRow = (date, entries) =>
    fetchJson(`/api/rtsp-monitor/history/${encodeURIComponent(date)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        entries: entries.map((entry) => ({
          stream_id: entry.stream_id,
          day_count: Math.max(Number(entry.day_count) || 0, 0),
          night_count: Math.max(Number(entry.night_count) || 0, 0),
        })),
      }),
    });

  const deleteShiftHistoryRow = (date) => {
    const params = new URLSearchParams();
    getSelectedHistoryStreams().forEach((stream) => params.append("stream_ids", stream.id));
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return fetchJson(`/api/rtsp-monitor/history/${encodeURIComponent(date)}${suffix}`, {
      method: "DELETE",
    });
  };

  const startMonitorForStream = async (streamId) => {
    const data = await fetchJson("/api/rtsp-monitor/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stream_id: streamId }),
    });
    renderMonitorStatus(data);
    await refreshShiftHistory({ quiet: true });
    return data;
  };

  const stopMonitorForStream = async (streamId) => {
    const params = new URLSearchParams();
    if (streamId) params.set("stream_id", streamId);
    const suffix = params.toString() ? `?${params.toString()}` : "";
    const data = await fetchJson(`/api/rtsp-monitor/stop${suffix}`, { method: "POST" });
    renderMonitorStatus(data);
    await refreshShiftHistory({ quiet: true });
    return data;
  };

  const applyUiLabels = () => {
    const fromLabel = shiftDateFromInput?.closest("label")?.querySelector(".field-label");
    const toLabel = shiftDateToInput?.closest("label")?.querySelector(".field-label");
    const streamsCard = byId("rtspStreamsCard");
    const monitorTitle = streamsCard?.querySelector("h2");
    const monitorPill = streamsCard?.querySelector(".pill");

    if (fromLabel) fromLabel.textContent = "С даты";
    if (toLabel) toLabel.textContent = "До даты";
    if (applyShiftHistoryFilterButton) applyShiftHistoryFilterButton.textContent = "Показать";
    if (exportShiftHistoryButton) exportShiftHistoryButton.textContent = "Выгрузить Excel";
    if (monitorTitle) monitorTitle.textContent = "Параллельный мониторинг RTSP-потоков";
    if (monitorPill) monitorPill.textContent = "2 потока одновременно";
    if (shiftHistoryEmpty) {
      shiftHistoryEmpty.textContent =
        "История по сменам появится после запуска анализа RTSP-потоков или из демо-данных.";
    }
  };

  rtspHistoryBody?.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action][data-stream-id]");
    if (!button) return;
    const streamId = button.dataset.streamId || "";
    const action = button.dataset.action || "";
    if (!streamId || !action) return;

    if (action === "analyze") {
      button.disabled = true;
      setMessage(rtspMessage, "Запускаю анализ выбранного RTSP-потока...", "warn");
      try {
        const data = await startMonitorForStream(streamId);
        setMessage(rtspMessage, data.message || "Анализ RTSP-потока запущен.", "ok");
      } catch (error) {
        setMessage(rtspMessage, error.message, "error");
      } finally {
        button.disabled = false;
      }
      return;
    }

    if (action === "delete") {
      button.disabled = true;
      setMessage(rtspMessage, "Удаляю RTSP-поток из реестра...", "warn");
      try {
        const data = await fetchJson(`/api/rtsp-streams/${encodeURIComponent(streamId)}`, { method: "DELETE" });
        rtspStreams = data.items || [];
        renderRtspHistory(rtspStreams);
        await refreshMonitorStatus({ quiet: true });
        await refreshShiftHistory({ quiet: true });
        setMessage(rtspMessage, data.message || "RTSP-поток удален.", "ok");
      } catch (error) {
        setMessage(rtspMessage, error.message, "error");
      } finally {
        button.disabled = false;
      }
    }
  });

  monitorGrid?.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action='stop-monitor'][data-stream-id]");
    if (!button) return;
    const streamId = button.dataset.streamId || "";
    if (!streamId) return;
    button.disabled = true;
    setMessage(rtspMessage, "Останавливаю мониторинг RTSP-потока...", "warn");
    try {
      const data = await stopMonitorForStream(streamId);
      setMessage(rtspMessage, data.message || "Мониторинг RTSP-потока остановлен.", "ok");
    } catch (error) {
      setMessage(rtspMessage, error.message, "error");
    } finally {
      button.disabled = false;
    }
  });

  shiftHistoryBody?.addEventListener("click", async (event) => {
    const button = event.target.closest("button[data-action][data-date]");
    if (!button) return;
    const bucketDate = button.dataset.date || "";
    const action = button.dataset.action || "";
    if (!bucketDate || !action) return;

    if (action === "edit-row") {
      editingShiftDate = bucketDate;
      renderShiftHistory(shiftHistoryItems);
      return;
    }

    if (action === "cancel-row") {
      editingShiftDate = "";
      renderShiftHistory(shiftHistoryItems);
      return;
    }

    if (action === "save-row") {
      const row = button.closest("tr");
      const entries = getSelectedHistoryStreams().map((stream) => {
        const dayInput = row?.querySelector(`input[data-stream-id="${stream.id}"][data-field="day_count"]`);
        const nightInput = row?.querySelector(`input[data-stream-id="${stream.id}"][data-field="night_count"]`);
        return {
          stream_id: stream.id,
          day_count: dayInput instanceof HTMLInputElement ? dayInput.value : "0",
          night_count: nightInput instanceof HTMLInputElement ? nightInput.value : "0",
        };
      });

      if (!entries.length) {
        setMessage(shiftHistoryMessage, "Не удалось прочитать значения строки.", "error");
        return;
      }

      button.disabled = true;
      setMessage(shiftHistoryMessage, `Сохраняю строку за ${bucketDate}...`, "warn");
      try {
        const data = await updateShiftHistoryRow(bucketDate, entries);
        editingShiftDate = "";
        await refreshShiftHistory({ quiet: true });
        await refreshMonitorStatus({ quiet: true });
        setMessage(shiftHistoryMessage, data.message || `Строка за ${bucketDate} обновлена.`, "ok");
      } catch (error) {
        setMessage(shiftHistoryMessage, error.message, "error");
      } finally {
        button.disabled = false;
      }
      return;
    }

    if (action === "delete-row") {
      button.disabled = true;
      setMessage(shiftHistoryMessage, `Удаляю строку за ${bucketDate}...`, "warn");
      try {
        const data = await deleteShiftHistoryRow(bucketDate);
        if (editingShiftDate === bucketDate) editingShiftDate = "";
        await refreshShiftHistory({ quiet: true });
        await refreshMonitorStatus({ quiet: true });
        setMessage(shiftHistoryMessage, data.message || `Строка за ${bucketDate} удалена.`, "ok");
      } catch (error) {
        setMessage(shiftHistoryMessage, error.message, "error");
      } finally {
        button.disabled = false;
      }
    }
  });

  byId("streamSetup")?.addEventListener("toggle", () => {
    if (syncingSetup) return;
    setupToggledByUser = true;
  });

  rtspTabButton?.addEventListener("click", () => {
    rtspTabButton.classList.add("active");
    rtspPanel?.classList.add("active");
  });

  refreshRtspButton?.addEventListener("click", async () => {
    refreshRtspButton.disabled = true;
    setMessage(rtspMessage, "Обновляю реестр RTSP-потоков...", "warn");
    try {
      await loadRtspHistory();
      await refreshMonitorStatus({ quiet: true });
      await refreshShiftHistory({ quiet: true });
      setMessage(rtspMessage, "Реестр RTSP-потоков обновлен.", "ok");
    } catch (error) {
      setMessage(rtspMessage, error.message, "error");
    } finally {
      refreshRtspButton.disabled = false;
    }
  });

  rtspForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = { name: rtspNameInput.value.trim(), url: rtspUrlInput.value.trim() };
    if (!payload.name || !payload.url) {
      setMessage(rtspMessage, "Заполните название потока и RTSP-ссылку.", "error");
      return;
    }

    addRtspButton.disabled = true;
    setMessage(rtspMessage, "Добавляю RTSP-поток...", "warn");
    try {
      const data = await fetchJson("/api/rtsp-streams", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      rtspStreams = data.items || [];
      renderRtspHistory(rtspStreams);
      await refreshShiftHistory({ quiet: true });
      setMessage(rtspMessage, data.message || "RTSP-поток добавлен.", "ok");
      rtspNameInput.value = "";
      rtspUrlInput.value = "";
    } catch (error) {
      setMessage(rtspMessage, error.message, "error");
    } finally {
      addRtspButton.disabled = false;
    }
  });

  applyShiftHistoryFilterButton?.addEventListener("click", async () => {
    applyShiftHistoryFilterButton.disabled = true;
    setMessage(shiftHistoryMessage, "Обновляю историю по выбранному периоду...", "warn");
    try {
      await refreshShiftHistory({ quiet: true });
      setMessage(shiftHistoryMessage, "История по выбранному периоду обновлена.", "ok");
    } catch (error) {
      setMessage(shiftHistoryMessage, error.message, "error");
    } finally {
      applyShiftHistoryFilterButton.disabled = false;
    }
  });

  exportShiftHistoryButton?.addEventListener("click", async () => {
    exportShiftHistoryButton.disabled = true;
    setMessage(shiftHistoryMessage, "Готовлю Excel-файл с историей...", "warn");
    try {
      const filename = await exportShiftHistory();
      setMessage(shiftHistoryMessage, `Файл ${filename} готов к скачиванию.`, "ok");
    } catch (error) {
      setMessage(shiftHistoryMessage, error.message, "error");
    } finally {
      exportShiftHistoryButton.disabled = false;
    }
  });

  window.refreshMonitorStatus = refreshMonitorStatus;
  window.refreshShiftHistory = refreshShiftHistory;
  window.renderMonitorStatus = renderMonitorStatus;
  window.renderShiftHistory = renderShiftHistory;
  window.loadRtspHistory = loadRtspHistory;

  rtspTabButton?.classList.add("active");
  rtspPanel?.classList.add("active");
  shiftDateToInput.value = toIsoDate(new Date());
  shiftDateFromInput.value = toIsoDate(new Date(new Date().getFullYear(), new Date().getMonth(), 1));
  applyUiLabels();
  renderMonitorStatus({ monitor_limit: 2, monitors: [] });
  renderShiftHistory([]);

  (async () => {
    try {
      await loadRtspHistory();
      await refreshMonitorStatus({ quiet: true });
      await refreshShiftHistory({ quiet: true });
    } catch (error) {
      setMessage(rtspMessage, error.message, "error");
    }
  })();
})();
