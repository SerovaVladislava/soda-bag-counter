(() => {
  const byId = (id) => document.getElementById(id);

  const tabButtons = Array.from(document.querySelectorAll(".tab-button"));
  const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));

  const detectionsTabButton = byId("detectionsTabButton");
  const rtspTabButton = byId("rtspTabButton");
  const detectionsPanel = byId("detectionsPanel");

  if (!detectionsTabButton || !detectionsPanel) return;

  const dateFromInput = byId("detectionsDateFrom");
  const dateToInput = byId("detectionsDateTo");
  const streamFilter = byId("detectionsStreamFilter");
  const applyButton = byId("detectionsApplyButton");
  const refreshButton = byId("detectionsRefreshButton");
  const clearButton = byId("detectionsClearButton");
  const messageBox = byId("detectionsMessage");

  const totalBox = byId("detectionsTotal");
  const daysBox = byId("detectionsDays");
  const retentionBox = byId("detectionsRetention");
  const emptyState = byId("detectionsEmpty");
  const groupsBox = byId("detectionsGroups");

  const AUTO_REFRESH_MS = 15000;
  let autoRefreshTimer = null;
  let knownStreams = [];

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const setMessage = (text, tone = "") => {
    if (!messageBox) return;
    messageBox.textContent = text || "";
    messageBox.className = tone ? `message ${tone}` : "message";
  };

  const request = async (url, options = {}) => {
    const response = await fetch(url, options);
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `Ошибка запроса (${response.status}).`;
      throw new Error(detail);
    }
    return payload;
  };

  const formatDateHeading = (value) => {
    if (!value) return "Без даты";
    const parsed = new Date(`${value}T00:00:00`);
    if (Number.isNaN(parsed.getTime())) return value;
    const formatted = parsed.toLocaleDateString("ru-RU", {
      day: "2-digit",
      month: "long",
      year: "numeric",
    });
    const weekday = parsed.toLocaleDateString("ru-RU", { weekday: "long" });
    return `${formatted}, ${weekday}`;
  };

  const plural = (count, one, few, many) => {
    const n = Math.abs(Number(count) || 0) % 100;
    const n1 = n % 10;
    if (n > 10 && n < 20) return many;
    if (n1 > 1 && n1 < 5) return few;
    if (n1 === 1) return one;
    return many;
  };

  const activatePanel = (panelId, button) => {
    tabButtons.forEach((item) => item.classList.toggle("active", item === button));
    tabPanels.forEach((panel) => panel.classList.toggle("active", panel.id === panelId));
  };

  const buildQuery = () => {
    const params = new URLSearchParams();
    if (dateFromInput && dateFromInput.value) params.set("date_from", dateFromInput.value);
    if (dateToInput && dateToInput.value) params.set("date_to", dateToInput.value);
    if (streamFilter && streamFilter.value) params.set("stream_id", streamFilter.value);
    const query = params.toString();
    return query ? `/api/detections?${query}` : "/api/detections";
  };

  const syncStreamOptions = (streams) => {
    if (!streamFilter) return;
    const items = Array.isArray(streams) ? streams : [];
    const signature = items.map((item) => `${item.id}:${item.name}`).join("|");
    const knownSignature = knownStreams.map((item) => `${item.id}:${item.name}`).join("|");
    if (signature === knownSignature) return;

    knownStreams = items;
    const current = streamFilter.value;
    streamFilter.innerHTML =
      '<option value="">Все потоки</option>' +
      items
        .map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)}</option>`)
        .join("");
    if (current && items.some((item) => item.id === current)) {
      streamFilter.value = current;
    }
  };

  const renderCard = (item) => {
    const url = `/api/detections/image/${encodeURIComponent(String(item.filename || "").split("/")[0])}/${encodeURIComponent(
      String(item.filename || "").split("/").pop(),
    )}`;
    const shift = item.shift === "night" ? "ночь" : "день";
    const fill = Number.isFinite(Number(item.fill)) ? Number(item.fill).toFixed(3) : "-";
    return `<figure class="detection-card">
        <a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">
          <img src="${escapeHtml(url)}" alt="Кадр с найденным мешком" loading="lazy" />
        </a>
        <figcaption class="detection-card-body">
          <div class="detection-time">${escapeHtml(item.local_time || "--:--:--")}<span
            class="shift-tag" data-shift="${escapeHtml(item.shift || "day")}">${escapeHtml(shift)}</span></div>
          <div class="detection-meta">${escapeHtml(item.stream_name || "RTSP-поток")}</div>
          <div class="detection-meta">мешок №${escapeHtml(String(item.bag_index ?? "-"))} · заполнение ${escapeHtml(fill)}</div>
        </figcaption>
      </figure>`;
  };

  const renderGroups = (payload) => {
    const groups = Array.isArray(payload?.groups) ? payload.groups : [];
    const total = Number(payload?.total) || 0;

    if (totalBox) totalBox.textContent = String(total);
    if (daysBox) daysBox.textContent = String(groups.length);
    if (retentionBox) {
      const days = Number(payload?.retention_days);
      retentionBox.textContent = Number.isFinite(days)
        ? `${days} ${plural(days, "день", "дня", "дней")}`
        : "-";
    }

    syncStreamOptions(payload?.streams);

    if (!groups.length) {
      if (groupsBox) {
        groupsBox.hidden = true;
        groupsBox.innerHTML = "";
      }
      if (emptyState) emptyState.hidden = false;
      return;
    }

    if (emptyState) emptyState.hidden = true;
    if (!groupsBox) return;

    groupsBox.hidden = false;
    groupsBox.innerHTML = groups
      .map((group) => {
        const items = Array.isArray(group.items) ? group.items : [];
        const count = Number(group.total) || items.length;
        const meta = `${count} ${plural(count, "кадр", "кадра", "кадров")} · день ${
          Number(group.day_count) || 0
        } · ночь ${Number(group.night_count) || 0}`;
        return `<section class="detection-day">
            <header class="detection-day-head">
              <span class="detection-day-title">${escapeHtml(formatDateHeading(group.date))}</span>
              <span class="detection-day-meta">${escapeHtml(meta)}</span>
            </header>
            <div class="detection-grid">${items.map(renderCard).join("")}</div>
          </section>`;
      })
      .join("");
  };

  const loadDetections = async ({ quiet = false } = {}) => {
    try {
      const payload = await request(buildQuery());
      renderGroups(payload);
      if (!quiet) {
        const total = Number(payload?.total) || 0;
        setMessage(
          total
            ? `Показано ${total} ${plural(total, "кадр", "кадра", "кадров")}.`
            : "За выбранный период кадров нет.",
          total ? "ok" : "warn",
        );
      }
      return payload;
    } catch (error) {
      if (!quiet) setMessage(error.message, "error");
      return null;
    }
  };

  const clearArchive = async () => {
    const streamId = streamFilter && streamFilter.value ? streamFilter.value : "";
    const scope = streamId ? "выбранного потока" : "всех потоков";
    if (!window.confirm(`Удалить сохранённые кадры для ${scope}? Действие необратимо.`)) return;

    if (clearButton) clearButton.disabled = true;
    try {
      const url = streamId
        ? `/api/detections?stream_id=${encodeURIComponent(streamId)}`
        : "/api/detections";
      const payload = await request(url, { method: "DELETE" });
      setMessage(payload?.message || "Архив очищен.", "ok");
      await loadDetections({ quiet: true });
    } catch (error) {
      setMessage(error.message, "error");
    } finally {
      if (clearButton) clearButton.disabled = false;
    }
  };

  const startAutoRefresh = () => {
    if (autoRefreshTimer !== null) return;
    autoRefreshTimer = setInterval(() => {
      if (!detectionsPanel.classList.contains("active")) return;
      loadDetections({ quiet: true }).catch(() => {});
    }, AUTO_REFRESH_MS);
  };

  detectionsTabButton.addEventListener("click", () => {
    activatePanel("detectionsPanel", detectionsTabButton);
    loadDetections({ quiet: true }).catch(() => {});
  });

  rtspTabButton?.addEventListener("click", () => activatePanel("rtspPanel", rtspTabButton));

  applyButton?.addEventListener("click", () => loadDetections());
  refreshButton?.addEventListener("click", () => loadDetections());
  clearButton?.addEventListener("click", clearArchive);
  streamFilter?.addEventListener("change", () => loadDetections({ quiet: true }));

  loadDetections({ quiet: true }).catch(() => {});
  startAutoRefresh();
})();
