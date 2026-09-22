from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable

import cv2
import numpy as np
from ultralytics import YOLO

BAG_LABEL_HINTS = {"bag", "bags", "sack", "sacks"}

# Live detection captures: one annotated still per counted bag, downscaled so a
# long-running archive stays a manageable size on disk.
# Замерший поток неотличим от пустой зоны: аналитика продолжает исправно
# брать пробы и честно сообщать, что мешка нет. Поэтому кадры сравниваются
# между пробами: если картинка перестала меняться, это не отсутствие мешков,
# а мёртвый источник, и об этом надо сказать вслух.
STALL_SIGNATURE_SIZE = 32
# Решение принимается по МЕДИАНЕ различий за окно, а не по отдельной пробе.
# Измерено на настоящем RTSP: застывшая картинка даёт медиану 0.004, живая
# сцена - 0.778, разрыв в двести раз. Но отдельные пробы разделить нельзя:
# у застывшего потока бывают всплески до 3.5 из-за битых макроблоков при
# потере пакетов, и они перекрывают весь диапазон живой сцены. Медиана к
# таким выбросам невосприимчива.
STALL_DIFF_THRESHOLD = 0.1
STALL_SECONDS = 90.0
STALL_MIN_SAMPLES = 10

DETECTION_CAPTURE_WIDTH = 1280
DETECTION_CAPTURE_JPEG_QUALITY = 80

ZoneBox = tuple[float, float, float, float]

# The only region that can produce a count: the column of air above the hopper
# mouth, full-res (1270,0)-(1762,412) at 2592x1944.  Every measured floor or
# platform bag has its bounding-box top at y >= 990 px, so a bag resting on the
# floor contributes fill 0.000 here at any threshold.
# NOTE: the permanent white soda-caked A-frame sling on the hopper rim,
# (1430,300)-(1720,560), is NOT fully below this zone - its top 112 rows fall
# inside it and account for about 0.6 percent of the zone area.  That is far
# below fill_enter so it cannot count on its own, but the zone must not be
# widened downwards on the assumption that the fixture is excluded.
ZONE_ABOVE_HOPPER: ZoneBox = (0.490, 0.000, 0.680, 0.212)
ZONE_CONTROL: ZoneBox = (0.050, 0.020, 0.300, 0.250)
ZONE_HOPPER_MOUTH: ZoneBox = (0.475, 0.232, 0.690, 0.310)
ZONE_HOPPER_BODY: ZoneBox = (0.505, 0.310, 0.670, 0.480)
ZONE_FLOOR_RIGHT: ZoneBox = (0.550, 0.520, 1.000, 1.000)
ZONE_FLOOR_CENTRE: ZoneBox = (0.330, 0.600, 0.560, 1.000)
ZONE_TRANSIT_RIGHT: ZoneBox = (0.700, 0.000, 1.000, 0.520)

ZONE_GRAY_LEVEL = 140
ZONE_SATURATION_MAX = 60
ZONE_FILL_ENTER = 0.21
ZONE_FILL_STAY = 0.17
ZONE_CONTROL_RATIO = 3.0
ZONE_BASELINE_WINDOW = 300
ZONE_BASELINE_PERCENTILE = 20.0
ZONE_BASELINE_MIN_SAMPLES = 30
ZONE_BASELINE_MARGIN = 0.06
ZONE_MIN_PRESENT_HITS = 2
# A commit needs BOTH a minimum number of samples and a minimum amount of
# WALL-CLOCK presence.  The sample count alone is meaningless: the sampling
# interval is derived from a measured frame rate that can be wrong in either
# direction, and two samples can span 0.13 s or 8 s.
#
# The threshold is what separates an unloading from a bag merely being carried
# past the camera.  The zone is flat, so a bag swinging a metre from the lens
# covers it exactly like a bag hanging over the hopper ten metres away - there
# is no depth to tell them apart.  Duration does tell them apart: measured over
# four recordings (40 min, 06-08 Sep) a crane moving bags to the floor occupies
# the zone for 4 s, while the eight real unloadings occupy it for 90-224 s.
# Every value from 10 s to 60 s yields the same eight episodes, so 20 s sits in
# the middle of a wide plateau: five times the transit, four times below the
# shortest real unloading.
ZONE_MIN_PRESENT_SECONDS = 20.0
ZONE_ABSENCE_SECONDS = 20.0
ZONE_SAMPLE_PERIOD_SECONDS = 2.0
ZONE_MAX_SAMPLE_PERIOD_SECONDS = 5.0
ZONE_MAX_COMMITS_PER_HOUR = 30
ZONE_MAX_EPISODE_SECONDS = 1800.0
ZONE_STUCK_CLEAR_SECONDS = 60.0
ZONE_MAX_TRACKED_EPISODES = 512
ZONE_PENDING_GAP_FACTOR = 2.5
ZONE_RATE_WINDOW_SECONDS = 3600.0
# The counter measures the sampling interval it is actually being driven at,
# instead of trusting the nominal period it was configured with.  Every gap
# based rule is then expressed against max(nominal, realized).
ZONE_SPACING_WINDOW = 9
ZONE_SPACING_MIN_SAMPLES = 3
ZONE_SPACING_MAX_SECONDS = 300.0


def _zone_bounds(frame_width: int, frame_height: int, box: ZoneBox) -> tuple[int, int, int, int]:
    x1 = int(box[0] * frame_width)
    y1 = int(box[1] * frame_height)
    x2 = int(box[2] * frame_width)
    y2 = int(box[3] * frame_height)
    return (
        max(min(x1, frame_width - 1), 0),
        max(min(y1, frame_height - 1), 0),
        max(min(x2, frame_width), 1),
        max(min(y2, frame_height), 1),
    )


def _zone_fill(frame: Any, box: ZoneBox, gray_level: int, saturation_max: int) -> float:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = _zone_bounds(frame_width, frame_height, box)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(((gray > gray_level) & (hsv[:, :, 1] < saturation_max)).mean())


@dataclass(frozen=True)
class FrameDetection:
    frame_position: int
    count: int
    confidence_sum: float
    top_confidence: float


@dataclass(frozen=True)
class SpatialDetection:
    frame_position: int
    center_x: float
    center_y: float
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class BagCandidate:
    frame_position: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    band: str
    source: str


@dataclass(frozen=True)
class ZoneObservation:
    frame_position: int
    timestamp: float
    fill: float
    control_fill: float
    baseline: float
    present: bool
    committed: bool
    state: str
    candidate: BagCandidate | None


def _frame_signature(frame: Any) -> Any:
    """Маленький слепок кадра: хватает, чтобы отличить движение от заморозки."""
    small = cv2.resize(
        frame,
        (STALL_SIGNATURE_SIZE, STALL_SIGNATURE_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


@dataclass(frozen=True)
class DetectionCapture:
    bag_index: int
    frame_position: int
    fill: float
    captured_at: datetime
    image_jpeg: bytes
    image_width: int
    image_height: int


class SampleClock:
    def __init__(self, source_kind: str, fps_effective: float) -> None:
        self.source_kind = source_kind
        self.fps_effective = max(float(fps_effective), 1e-6)
        self.use_position_msec = source_kind != "rtsp"
        self._origin: float | None = None
        self._last_value = -1.0

    def update_fps(self, fps_effective: float) -> None:
        self.fps_effective = max(float(fps_effective), 1e-6)

    def stamp(self, capture: cv2.VideoCapture, sequential_index: int) -> float:
        if self.source_kind == "rtsp":
            now = time.monotonic()
            if self._origin is None:
                self._origin = now
            return now - self._origin

        if self.use_position_msec:
            try:
                position_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            except Exception:
                position_msec = 0.0
            value = position_msec / 1000.0
            if value > self._last_value:
                self._last_value = value
                return value
            self.use_position_msec = False

        return float(sequential_index) / self.fps_effective


class HopperZoneEpisodeCounter:
    def __init__(
        self,
        zone: ZoneBox = ZONE_ABOVE_HOPPER,
        control_zone: ZoneBox = ZONE_CONTROL,
        gray_level: int = ZONE_GRAY_LEVEL,
        saturation_max: int = ZONE_SATURATION_MAX,
        fill_enter: float = ZONE_FILL_ENTER,
        fill_stay: float = ZONE_FILL_STAY,
        control_ratio: float = ZONE_CONTROL_RATIO,
        baseline_window: int = ZONE_BASELINE_WINDOW,
        baseline_percentile: float = ZONE_BASELINE_PERCENTILE,
        baseline_min_samples: int = ZONE_BASELINE_MIN_SAMPLES,
        baseline_margin: float = ZONE_BASELINE_MARGIN,
        min_present_hits: int = ZONE_MIN_PRESENT_HITS,
        min_present_seconds: float = ZONE_MIN_PRESENT_SECONDS,
        absence_seconds: float = ZONE_ABSENCE_SECONDS,
        sample_period_seconds: float = ZONE_SAMPLE_PERIOD_SECONDS,
        max_commits_per_hour: int = ZONE_MAX_COMMITS_PER_HOUR,
        max_episode_seconds: float = ZONE_MAX_EPISODE_SECONDS,
        stuck_clear_seconds: float = ZONE_STUCK_CLEAR_SECONDS,
        max_tracked_episodes: int = ZONE_MAX_TRACKED_EPISODES,
        spacing_window: int = ZONE_SPACING_WINDOW,
        spacing_min_samples: int = ZONE_SPACING_MIN_SAMPLES,
    ) -> None:
        self.zone = zone
        self.control_zone = control_zone
        self.gray_level = int(gray_level)
        self.saturation_max = int(saturation_max)
        self.fill_enter = float(fill_enter)
        self.fill_stay = min(float(fill_stay), float(fill_enter))
        self.control_ratio = max(float(control_ratio), 0.0)
        self.baseline_percentile = float(baseline_percentile)
        self.baseline_min_samples = max(int(baseline_min_samples), 1)
        self.baseline_margin = float(baseline_margin)
        self.min_present_hits = max(int(min_present_hits), 1)
        self.min_present_seconds = max(float(min_present_seconds), 0.0)
        self.absence_seconds = max(float(absence_seconds), 0.0)
        self.sample_period_seconds = max(float(sample_period_seconds), 1e-3)
        self.max_commits_per_hour = max(int(max_commits_per_hour), 1)
        self.max_episode_seconds = max(float(max_episode_seconds), 0.0)
        self.stuck_clear_seconds = max(float(stuck_clear_seconds), 0.0)
        self.max_tracked_episodes = max(int(max_tracked_episodes), 1)
        self.spacing_min_samples = max(int(spacing_min_samples), 1)

        self.count = 0
        self.state = "idle"
        self.degraded = False
        self.last_fill = 0.0
        self.last_control_fill = 0.0
        self.last_baseline = 0.0
        self.present_samples = 0
        self.pending_hits = 0

        self._fills: deque[float] = deque(maxlen=max(int(baseline_window), 1))
        self._commit_timestamps: deque[float] = deque(maxlen=self.max_commits_per_hour + 1)
        self._episodes: deque[dict[str, Any]] = deque(maxlen=self.max_tracked_episodes)
        self._open_episode: dict[str, Any] | None = None
        self._last_present_t = 0.0
        self._prev_present_t: float | None = None
        self._first_present_t: float | None = None
        self._episode_start_t = 0.0
        self._suppressed_until_clear = False
        self._clear_since: float | None = None
        self._gaps: deque[float] = deque(maxlen=max(int(spacing_window), 1))
        self._last_sample_t: float | None = None

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return [dict(episode) for episode in self._episodes]

    @property
    def realized_sample_period_seconds(self) -> float:
        """Median spacing of the samples this counter is actually being fed.

        The nominal ``sample_period_seconds`` is only a request: the live loop
        derives its stride from a measured frame rate that can be wrong by a
        factor of four in either direction (this camera declares 60 fps on a
        true 15 fps stream).  Every gap rule below is therefore evaluated
        against ``effective_sample_period_seconds`` and never against the
        nominal value alone.
        """
        if len(self._gaps) < self.spacing_min_samples:
            return self.sample_period_seconds
        gaps = sorted(self._gaps)
        middle = len(gaps) // 2
        if len(gaps) % 2:
            median = gaps[middle]
        else:
            median = 0.5 * (gaps[middle - 1] + gaps[middle])
        return max(float(median), 1e-3)

    @property
    def effective_sample_period_seconds(self) -> float:
        return max(self.sample_period_seconds, self.realized_sample_period_seconds)

    def set_sample_period(self, sample_period_seconds: float) -> None:
        self.sample_period_seconds = max(float(sample_period_seconds), 1e-3)

    def _track_spacing(self, timestamp: float) -> None:
        previous = self._last_sample_t
        self._last_sample_t = timestamp
        if previous is None:
            return
        gap = timestamp - previous
        # A non-positive gap means the clock restarted (file rewind, reconnect)
        # and an absurd gap means the stream stalled; neither describes the
        # steady-state sampling interval, so neither is allowed to poison it.
        if 0.0 < gap <= ZONE_SPACING_MAX_SECONDS:
            self._gaps.append(float(gap))
        elif gap <= 0.0:
            self._rebase_clock(timestamp)

    def _rebase_clock(self, timestamp: float) -> None:
        """Move the episode timers into the new clock epoch.

        Without this, anchors left in the old, larger epoch make every
        ``timestamp - anchor`` negative, so the absence timer and the stuck
        episode guard can never fire and an open episode would stay ACTIVE
        forever, blocking every later count.  Rebasing restarts the timers from
        the restart moment, which is the conservative reading: the same bag may
        still be in the zone.
        """
        self._last_present_t = min(self._last_present_t, timestamp)
        self._episode_start_t = min(self._episode_start_t, timestamp)
        if self._prev_present_t is not None and self._prev_present_t > timestamp:
            self._prev_present_t = None
        if self._first_present_t is not None and self._first_present_t > timestamp:
            self._first_present_t = None
            self.pending_hits = 0
        if self._clear_since is not None and self._clear_since > timestamp:
            self._clear_since = timestamp
        while self._commit_timestamps and self._commit_timestamps[-1] > timestamp:
            self._commit_timestamps.pop()

    def observe(self, frame: Any, frame_position: int, timestamp: float) -> ZoneObservation:
        self._track_spacing(timestamp)
        fill = _zone_fill(frame, self.zone, self.gray_level, self.saturation_max)
        control_fill = (
            _zone_fill(frame, self.control_zone, self.gray_level, self.saturation_max)
            if self.control_ratio > 0.0
            else 0.0
        )

        self._fills.append(fill)
        baseline = (
            float(np.percentile(np.asarray(self._fills, dtype=np.float32), self.baseline_percentile))
            if len(self._fills) >= self.baseline_min_samples
            else 0.0
        )

        threshold = self.fill_stay if self.state == "active" else self.fill_enter
        present = fill >= threshold
        if present and self.control_ratio > 0.0:
            present = fill >= self.control_ratio * max(control_fill, 1e-6)

        self._update_stuck_suppression(fill=fill, timestamp=timestamp)
        commit_allowed = (not self._suppressed_until_clear) and fill >= baseline + self.baseline_margin
        committed = False

        if self.state == "active":
            if present:
                self._last_present_t = timestamp
                self._touch_episode(frame_position=frame_position, timestamp=timestamp, fill=fill)
            elif timestamp - self._last_present_t >= self.absence_seconds:
                self._close_episode()
            if self.state == "active" and self.max_episode_seconds > 0.0:
                if timestamp - self._episode_start_t >= self.max_episode_seconds:
                    self._close_episode()
                    self._suppressed_until_clear = True
                    self._clear_since = None
        elif present and commit_allowed:
            gap_limit = ZONE_PENDING_GAP_FACTOR * self.effective_sample_period_seconds
            if (
                self._prev_present_t is not None
                and timestamp - self._prev_present_t > gap_limit
            ):
                self.pending_hits = 0
                self._first_present_t = None
            if self._first_present_t is None:
                self._first_present_t = timestamp
            self.pending_hits += 1
            self._prev_present_t = timestamp
            # Two conditions, and the seconds one is what actually rejects
            # steam: however dense or sparse the sampling turns out to be, the
            # zone must stay occupied for min_present_seconds of real time.
            present_span = timestamp - self._first_present_t
            if (
                self.pending_hits >= self.min_present_hits
                and present_span >= self.min_present_seconds
            ):
                committed = self._commit(frame_position=frame_position, timestamp=timestamp, fill=fill)
        else:
            self.pending_hits = 0
            self._prev_present_t = None
            self._first_present_t = None

        self.last_fill = fill
        self.last_control_fill = control_fill
        self.last_baseline = baseline
        if present:
            self.present_samples += 1

        return ZoneObservation(
            frame_position=int(frame_position),
            timestamp=float(timestamp),
            fill=float(fill),
            control_fill=float(control_fill),
            baseline=float(baseline),
            present=bool(present),
            committed=bool(committed),
            state=self.state,
            candidate=self._build_candidate(frame=frame, frame_position=frame_position, fill=fill),
        )

    def finish(self) -> None:
        self._close_episode()

    def _commit(self, frame_position: int, timestamp: float, fill: float) -> bool:
        self.pending_hits = 0
        self._prev_present_t = None
        self._first_present_t = None
        self.state = "active"
        self._last_present_t = timestamp
        self._episode_start_t = timestamp

        if self._rate_capped(timestamp):
            self.degraded = True
            self._open_episode = None
            return False

        # The rate cap is a rolling window, so it un-trips by itself.  Clearing
        # the latch here keeps the operator message from claiming the count is
        # frozen while it is visibly rising again.
        self.degraded = False
        self._commit_timestamps.append(timestamp)
        self.count += 1
        self._open_episode = {
            "index": int(self.count),
            "start_frame": int(frame_position),
            "end_frame": int(frame_position),
            "start_timestamp": float(timestamp),
            "end_timestamp": float(timestamp),
            "hit_count": 1,
            "peak_fill": float(fill),
            "representative_frame": int(frame_position),
        }
        return True

    def _rate_capped(self, timestamp: float) -> bool:
        while self._commit_timestamps and timestamp - self._commit_timestamps[0] > ZONE_RATE_WINDOW_SECONDS:
            self._commit_timestamps.popleft()
        return len(self._commit_timestamps) >= self.max_commits_per_hour

    def _touch_episode(self, frame_position: int, timestamp: float, fill: float) -> None:
        if self._open_episode is None:
            return
        self._open_episode["end_frame"] = int(frame_position)
        self._open_episode["end_timestamp"] = float(timestamp)
        self._open_episode["hit_count"] = int(self._open_episode["hit_count"]) + 1
        if fill >= float(self._open_episode["peak_fill"]):
            self._open_episode["peak_fill"] = float(fill)
            self._open_episode["representative_frame"] = int(frame_position)

    def _close_episode(self) -> None:
        if self._open_episode is not None:
            self._episodes.append(self._open_episode)
            self._open_episode = None
        self.state = "idle"
        self.pending_hits = 0
        self._prev_present_t = None
        self._first_present_t = None

    def _update_stuck_suppression(self, fill: float, timestamp: float) -> None:
        if not self._suppressed_until_clear:
            return
        if fill < self.fill_stay:
            if self._clear_since is None:
                self._clear_since = timestamp
            elif timestamp - self._clear_since >= self.stuck_clear_seconds:
                self._suppressed_until_clear = False
                self._clear_since = None
        else:
            self._clear_since = None

    def _build_candidate(self, frame: Any, frame_position: int, fill: float) -> BagCandidate | None:
        if fill < self.fill_stay:
            return None
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = _zone_bounds(frame_width, frame_height, self.zone)
        return BagCandidate(
            frame_position=int(frame_position),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            score=float(fill),
            band="top",
            source="hopper_zone",
        )


class BagAnalyticsManager:
    def __init__(
        self,
        model_path: Path,
        coarse_confidence: float = 0.05,
        fine_confidence: float = 0.10,
        coarse_imgsz: int = 640,
        fine_imgsz: int = 960,
        coarse_samples: int = 48,
        fine_samples: int = 5,
        candidate_windows: int = 5,
        fallback_confidence: float = 0.01,
        tile_confidence: float = 0.04,
        live_frame_stride: int = 15,
        live_history_size: int = 5,
        live_open_timeout_seconds: float = 15.0,
        live_read_retry_seconds: float = 0.5,
        live_max_read_failures: int = 6,
        live_event_gap_samples: int = 12,
        live_presence_score_threshold: float = 25000.0,
        video_event_sample_seconds: float = 5.0,
        video_event_merge_gap_seconds: float = 40.0,
        video_event_min_hits: int = 2,
        video_event_min_single_score: float = 20000.0,
        zone_above_hopper: ZoneBox = ZONE_ABOVE_HOPPER,
        zone_control: ZoneBox = ZONE_CONTROL,
        zone_gray_level: int = ZONE_GRAY_LEVEL,
        zone_saturation_max: int = ZONE_SATURATION_MAX,
        zone_fill_enter: float = ZONE_FILL_ENTER,
        zone_fill_stay: float = ZONE_FILL_STAY,
        zone_control_ratio: float = ZONE_CONTROL_RATIO,
        zone_baseline_window: int = ZONE_BASELINE_WINDOW,
        zone_baseline_percentile: float = ZONE_BASELINE_PERCENTILE,
        zone_baseline_min_samples: int = ZONE_BASELINE_MIN_SAMPLES,
        zone_baseline_margin: float = ZONE_BASELINE_MARGIN,
        zone_min_present_hits: int = ZONE_MIN_PRESENT_HITS,
        zone_min_present_seconds: float = ZONE_MIN_PRESENT_SECONDS,
        zone_absence_seconds: float = ZONE_ABSENCE_SECONDS,
        zone_sample_period_seconds: float = ZONE_SAMPLE_PERIOD_SECONDS,
        zone_max_sample_period_seconds: float = ZONE_MAX_SAMPLE_PERIOD_SECONDS,
        zone_max_commits_per_hour: int = ZONE_MAX_COMMITS_PER_HOUR,
        zone_max_episode_seconds: float = ZONE_MAX_EPISODE_SECONDS,
        zone_stuck_clear_seconds: float = ZONE_STUCK_CLEAR_SECONDS,
        zone_max_tracked_episodes: int = ZONE_MAX_TRACKED_EPISODES,

        stall_seconds: float = STALL_SECONDS,
        detection_sink: Callable[[DetectionCapture], None] | None = None,
        detection_capture_width: int = DETECTION_CAPTURE_WIDTH,
        detection_jpeg_quality: int = DETECTION_CAPTURE_JPEG_QUALITY,        fps_fallback: float = 15.0,
        fps_min_plausible: float = 1.0,
        fps_max_plausible: float = 60.0,
        fps_probe_seconds: float = 10.0,
        fps_probe_frames: int = 150,
        fps_warmup_seconds: float = 10.0,
        fps_file_probe_frames: int = 200,
        fps_reestimate_seconds: float = 60.0,
        fps_reestimate_drift: float = 0.30,
        max_sample_stride: int = 120,
        video_summary_points: int = 720,
        overrun_escalate_seconds: float = 30.0,
        overrun_relax_seconds: float = 120.0,
    ) -> None:
        self.model_path = model_path
        self.coarse_confidence = coarse_confidence
        self.fine_confidence = fine_confidence
        self.coarse_imgsz = coarse_imgsz
        self.fine_imgsz = fine_imgsz
        self.coarse_samples = coarse_samples
        self.fine_samples = fine_samples
        self.candidate_windows = candidate_windows
        self.fallback_confidence = fallback_confidence
        self.tile_confidence = tile_confidence
        self.live_frame_stride = max(int(live_frame_stride), 1)
        self.live_history_size = live_history_size
        self.live_open_timeout_seconds = live_open_timeout_seconds
        self.live_read_retry_seconds = live_read_retry_seconds
        self.live_max_read_failures = live_max_read_failures
        self.live_event_gap_samples = max(int(live_event_gap_samples), 1)
        self.live_presence_score_threshold = max(float(live_presence_score_threshold), 1.0)
        self.video_event_sample_seconds = max(float(video_event_sample_seconds), 0.5)
        self.video_event_merge_gap_seconds = max(
            float(video_event_merge_gap_seconds),
            self.video_event_sample_seconds,
        )
        self.video_event_min_hits = max(int(video_event_min_hits), 1)
        self.video_event_min_single_score = max(float(video_event_min_single_score), 1.0)
        self.zone_above_hopper = zone_above_hopper
        self.zone_control = zone_control
        self.zone_gray_level = int(zone_gray_level)
        self.zone_saturation_max = int(zone_saturation_max)
        self.zone_fill_enter = float(zone_fill_enter)
        self.zone_fill_stay = min(float(zone_fill_stay), float(zone_fill_enter))
        self.zone_control_ratio = max(float(zone_control_ratio), 0.0)
        self.zone_baseline_window = max(int(zone_baseline_window), 1)
        self.zone_baseline_percentile = float(zone_baseline_percentile)
        self.zone_baseline_min_samples = max(int(zone_baseline_min_samples), 1)
        self.zone_baseline_margin = float(zone_baseline_margin)
        self.zone_min_present_hits = max(int(zone_min_present_hits), 1)
        self.zone_min_present_seconds = max(float(zone_min_present_seconds), 0.0)
        self.zone_absence_seconds = max(float(zone_absence_seconds), 0.0)
        self.zone_sample_period_seconds = max(float(zone_sample_period_seconds), 0.1)
        self.zone_max_sample_period_seconds = max(
            float(zone_max_sample_period_seconds),
            self.zone_sample_period_seconds,
        )
        self.zone_max_commits_per_hour = max(int(zone_max_commits_per_hour), 1)
        self.zone_max_episode_seconds = max(float(zone_max_episode_seconds), 0.0)
        self.zone_stuck_clear_seconds = max(float(zone_stuck_clear_seconds), 0.0)
        self.zone_max_tracked_episodes = max(int(zone_max_tracked_episodes), 1)
        self.fps_fallback = max(float(fps_fallback), 1.0)
        self.fps_min_plausible = max(float(fps_min_plausible), 0.01)
        self.fps_max_plausible = max(float(fps_max_plausible), self.fps_min_plausible)
        self.fps_probe_seconds = max(float(fps_probe_seconds), 0.5)
        self.fps_probe_frames = max(int(fps_probe_frames), 2)
        self.fps_warmup_seconds = max(float(fps_warmup_seconds), 0.0)
        self.fps_file_probe_frames = max(int(fps_file_probe_frames), 2)
        self.fps_reestimate_seconds = max(float(fps_reestimate_seconds), 5.0)
        self.fps_reestimate_drift = max(float(fps_reestimate_drift), 0.01)
        self.max_sample_stride = max(int(max_sample_stride), 1)
        self.video_summary_points = max(int(video_summary_points), 2)
        self.overrun_escalate_seconds = max(float(overrun_escalate_seconds), 1.0)
        self.overrun_relax_seconds = max(float(overrun_relax_seconds), 1.0)

        self.active_frame_stride = self.live_frame_stride
        self.active_sample_period_seconds = self.zone_sample_period_seconds
        self.fps_effective: float | None = None
        self.fps_source: str | None = None
        self.stream_stalled = False
        self.last_zone_fill: float | None = None
        self.last_zone_baseline: float | None = None
        self.episode_state = "idle"

        self.stall_seconds = max(float(stall_seconds), 0.0)
        self.detection_sink = detection_sink
        self.detection_capture_width = max(int(detection_capture_width), 320)
        self.detection_jpeg_quality = min(max(int(detection_jpeg_quality), 40), 95)

        self.lock = threading.Lock()
        self.model: YOLO | None = None
        self.thread: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.run_id = 0
        # start() publishes self.thread under the lock but can only call
        # thread.start() after releasing it (it first joins the previous
        # thread).  Without this flag a status() poll landing in that window
        # sees a not-yet-alive thread and declares the run dead.
        self.start_pending = False

        self.state = "idle"
        self.message = "Аналитика верхнего мешка запустится после старта RTSP-потока."
        self.bag_count: int | None = None
        self.max_bag_count: int | None = None
        self.frames_processed = 0
        self.sample_counts: list[int] = []
        self.class_names: list[str] = []
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.last_result_at: datetime | None = None
        self.last_error: str | None = None
        self.source_kind: str | None = None

    def start(self, source: str) -> dict[str, Any]:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        if not source:
            raise ValueError("Нужно указать источник для аналитики.")

        source_kind = self._resolve_source_kind(source)
        if source_kind == "file" and not Path(source).exists():
            raise FileNotFoundError(f"Видеофайл не найден: {source}")

        previous_thread: threading.Thread | None = None
        previous_stop_event: threading.Event | None = None

        with self.lock:
            previous_thread = self.thread
            previous_stop_event = self.stop_event
            if previous_stop_event is not None:
                previous_stop_event.set()

            self.run_id += 1
            current_run_id = self.run_id
            self.stop_event = threading.Event()
            self.thread = threading.Thread(
                target=self._run,
                args=(current_run_id, source, source_kind, self.stop_event),
                daemon=True,
            )
            self.state = "running"
            self.message = (
                "Аналитика RTSP-потока запущена: считаю мешки, опускаемые краном в бункер "
                f"(проба зоны над бункером каждые {self.zone_sample_period_seconds:.0f} с)..."
                if source_kind == "rtsp"
                else "Идет обработка источника по мешкам, опускаемым в бункер..."
            )
            self.bag_count = None
            self.max_bag_count = None
            self.frames_processed = 0
            self.sample_counts = []
            self.active_sample_period_seconds = self.zone_sample_period_seconds
            self.fps_effective = None
            self.fps_source = None
            self.stream_stalled = False
            self.last_zone_fill = None
            self.last_zone_baseline = None
            self.episode_state = "idle"
            self.started_at = datetime.now(timezone.utc)
            self.completed_at = None
            self.last_result_at = None
            self.last_error = None
            self.source_kind = source_kind
            self.start_pending = True

            snapshot = self._snapshot_locked()
            thread = self.thread

        try:
            if previous_thread is not None and previous_thread.is_alive():
                previous_thread.join(timeout=1)

            if thread is not None:
                thread.start()
        finally:
            with self.lock:
                if self.run_id == current_run_id:
                    self.start_pending = False

        return snapshot

    def reset(self, wait: bool = True, clear_result: bool = True) -> dict[str, Any]:
        thread: threading.Thread | None = None
        stop_event: threading.Event | None = None

        with self.lock:
            self.run_id += 1
            thread = self.thread
            stop_event = self.stop_event
            self.thread = None
            self.stop_event = None
            self.start_pending = False

            if clear_result:
                self.state = "idle"
                self.message = "Аналитика верхнего мешка запустится после старта RTSP-потока."
                self.bag_count = None
                self.max_bag_count = None
                self.frames_processed = 0
                self.sample_counts = []
                self.active_sample_period_seconds = self.zone_sample_period_seconds
                self.fps_effective = None
                self.fps_source = None
                self.stream_stalled = False
                self.last_zone_fill = None
                self.last_zone_baseline = None
                self.episode_state = "idle"
                self.started_at = None
                self.completed_at = None
                self.last_result_at = None
                self.last_error = None
                self.source_kind = None

            snapshot = self._snapshot_locked()

        if stop_event is not None:
            stop_event.set()
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=2)

        return snapshot

    def shutdown(self) -> None:
        self.reset(wait=True, clear_result=True)

    def status(self) -> dict[str, Any]:
        with self.lock:
            if (
                not self.start_pending
                and self.thread is not None
                and not self.thread.is_alive()
                and self.state == "running"
            ):
                self.state = "error"
                self.message = "Поток аналитики остановился неожиданно."
                if not self.last_error:
                    self.last_error = "Поток аналитики остановился неожиданно."
                self.thread = None
                self.stop_event = None

            return self._snapshot_locked()

    def _run(
        self,
        run_id: int,
        source: str,
        source_kind: str,
        stop_event: threading.Event,
    ) -> None:
        capture: cv2.VideoCapture | None = None

        try:
            names_map = self._load_class_names()
            capture = self._wait_for_capture(
                run_id=run_id,
                source=source,
                source_kind=source_kind,
                stop_event=stop_event,
            )
            if capture is None:
                return

            counter = self._build_zone_counter()
            sample_period = float(self.zone_sample_period_seconds)
            fps_effective, fps_source = self._resolve_capture_fps(
                capture=capture,
                source=source,
                source_kind=source_kind,
                stop_event=stop_event,
            )
            frame_stride = self._resolve_sample_stride(
                fps_effective=fps_effective,
                sample_period_seconds=sample_period,
            )
            clock = SampleClock(source_kind=source_kind, fps_effective=fps_effective)
            self._publish_stream_profile(
                run_id=run_id,
                fps_effective=fps_effective,
                fps_source=fps_source,
                frame_stride=frame_stride,
                sample_period_seconds=sample_period,
            )

            sampled_frame_index = 0
            stream_frame_index = 0
            recent_presence: list[int] = []
            consecutive_read_failures = 0
            previous_sample_t: float | None = None
            overrun_since: float | None = None
            headroom_since: float | None = None
            fps_window_started = time.monotonic()
            fps_window_frames = 0
            previous_signature = None
            motion_history: deque[tuple[float, float]] = deque()
            stalled = False

            while not stop_event.is_set():
                is_sample = stream_frame_index % frame_stride == 0
                if is_sample:
                    ok, frame = capture.read()
                else:
                    ok = capture.grab()
                    frame = None

                if not ok or (is_sample and frame is None):
                    consecutive_read_failures += 1
                    if stop_event.is_set():
                        return
                    if consecutive_read_failures >= self.live_max_read_failures:
                        if source_kind != "rtsp":
                            # End of a finite file.  Reopening it would replay
                            # the same bags from position 0 into the same
                            # counter and grow bag_count without bound for a
                            # fixed amount of real content.
                            counter.finish()
                            with self.lock:
                                if run_id != self.run_id:
                                    return
                                self.state = "done"
                                self.message = (
                                    f"Источник обработан: учтено {int(counter.count)} мешк(ов), "
                                    "опущенных в бункер."
                                )
                                self.bag_count = int(counter.count)
                                self.max_bag_count = int(counter.count)
                                self.episode_state = counter.state
                                self.completed_at = datetime.now(timezone.utc)
                                self.last_result_at = datetime.now(timezone.utc)
                                self.last_error = None
                                self.thread = None
                                self.stop_event = None
                            return
                        capture.release()
                        capture = self._wait_for_capture(
                            run_id=run_id,
                            source=source,
                            source_kind=source_kind,
                            stop_event=stop_event,
                        )
                        if capture is None:
                            return
                        fps_effective, fps_source = self._resolve_capture_fps(
                            capture=capture,
                            source=source,
                            source_kind=source_kind,
                            stop_event=stop_event,
                        )
                        frame_stride = self._resolve_sample_stride(
                            fps_effective=fps_effective,
                            sample_period_seconds=sample_period,
                        )
                        clock.update_fps(fps_effective)
                        self._publish_stream_profile(
                            run_id=run_id,
                            fps_effective=fps_effective,
                            fps_source=fps_source,
                            frame_stride=frame_stride,
                            sample_period_seconds=sample_period,
                        )
                        stream_frame_index = 0
                        fps_window_started = time.monotonic()
                        fps_window_frames = 0
                        consecutive_read_failures = 0
                    time.sleep(self.live_read_retry_seconds)
                    continue

                consecutive_read_failures = 0
                sequential_index = stream_frame_index
                stream_frame_index += 1
                fps_window_frames += 1
                if not is_sample:
                    continue

                sampled_frame_index += 1
                timestamp = clock.stamp(capture=capture, sequential_index=sequential_index)
                observation = counter.observe(
                    frame=frame,
                    frame_position=sequential_index,
                    timestamp=timestamp,
                )

                recent_presence.append(1 if observation.present else 0)
                if len(recent_presence) > self.live_history_size:
                    recent_presence.pop(0)

                if observation.committed:
                    self._emit_detection_capture(
                        frame=frame,
                        observation=observation,
                        bag_index=int(counter.count),
                    )

                sample_counts = recent_presence.copy()
                support_frames = sum(sample_counts)

                signature = _frame_signature(frame)
                if previous_signature is not None:
                    motion_history.append(
                        (timestamp, float(np.abs(signature - previous_signature).mean()))
                    )
                    while motion_history and timestamp - motion_history[0][0] > self.stall_seconds:
                        motion_history.popleft()
                previous_signature = signature
                stalled = (
                    self.stall_seconds > 0.0
                    and len(motion_history) >= STALL_MIN_SAMPLES
                    and timestamp - motion_history[0][0] >= self.stall_seconds * 0.8
                    and float(np.median([value for _, value in motion_history]))
                    < STALL_DIFF_THRESHOLD
                )

                with self.lock:
                    if run_id != self.run_id:
                        return

                    self.state = "stalled" if stalled else "running"
                    self.message = self._build_stall_message(
                        seconds=(timestamp - motion_history[0][0]) if motion_history else 0.0,
                        final_count=int(counter.count),
                    ) if stalled else self._build_live_message(
                        final_count=int(counter.count),
                        support_frames=support_frames,
                        sampled_frames=len(sample_counts),
                        episode_state=counter.state,
                        pending_hits=int(counter.pending_hits),
                        degraded=bool(counter.degraded),
                    )
                    self.stream_stalled = bool(stalled)
                    self.bag_count = int(counter.count)
                    self.max_bag_count = int(counter.count)
                    self.frames_processed = sampled_frame_index
                    self.sample_counts = sample_counts
                    self.last_zone_fill = float(observation.fill)
                    self.last_zone_baseline = float(observation.baseline)
                    self.episode_state = counter.state
                    self.completed_at = None
                    self.last_result_at = datetime.now(timezone.utc)
                    self.last_error = None
                    self.class_names = [str(name) for _, name in sorted(names_map.items())]

                if source_kind != "rtsp":
                    previous_sample_t = timestamp
                    continue

                # "Behind" means the loop fell out of ITS OWN steady state, not
                # that it missed a nominal period it never had a chance of
                # hitting.  Measuring against the nominal period alone made a
                # four-times-too-large stride look like an overload and ratchet
                # the period up, which makes the real problem worse.
                expected_sample_seconds = max(
                    sample_period,
                    counter.effective_sample_period_seconds,
                )
                behind = (
                    previous_sample_t is not None
                    and timestamp - previous_sample_t
                    > ZONE_PENDING_GAP_FACTOR * expected_sample_seconds
                )
                previous_sample_t = timestamp
                if behind:
                    headroom_since = None
                    if overrun_since is None:
                        overrun_since = timestamp
                else:
                    overrun_since = None
                    if headroom_since is None:
                        headroom_since = timestamp

                updated_period = sample_period
                if (
                    overrun_since is not None
                    and timestamp - overrun_since >= self.overrun_escalate_seconds
                ):
                    updated_period = min(sample_period + 1.0, self.zone_max_sample_period_seconds)
                    overrun_since = timestamp
                elif (
                    headroom_since is not None
                    and timestamp - headroom_since >= self.overrun_relax_seconds
                ):
                    updated_period = max(sample_period - 1.0, self.zone_sample_period_seconds)
                    headroom_since = timestamp

                fps_window_elapsed = time.monotonic() - fps_window_started
                updated_fps = fps_effective
                if fps_window_elapsed >= self.fps_reestimate_seconds and fps_window_frames >= 2:
                    observed_fps = fps_window_frames / fps_window_elapsed
                    if (
                        self.fps_min_plausible <= observed_fps <= self.fps_max_plausible
                        and abs(observed_fps - fps_effective)
                        > self.fps_reestimate_drift * fps_effective
                    ):
                        updated_fps = observed_fps
                        fps_source = "measured"
                    fps_window_started = time.monotonic()
                    fps_window_frames = 0

                if updated_period != sample_period or updated_fps != fps_effective:
                    sample_period = updated_period
                    fps_effective = updated_fps
                    counter.set_sample_period(sample_period)
                    clock.update_fps(fps_effective)
                    frame_stride = self._resolve_sample_stride(
                        fps_effective=fps_effective,
                        sample_period_seconds=sample_period,
                    )
                    stream_frame_index = 0
                    self._publish_stream_profile(
                        run_id=run_id,
                        fps_effective=fps_effective,
                        fps_source=fps_source,
                        frame_stride=frame_stride,
                        sample_period_seconds=sample_period,
                    )

        except Exception as exc:
            with self.lock:
                if run_id != self.run_id:
                    return

                self.state = "error"
                self.message = "Ошибка аналитики мешков, опускаемых в бункер."
                self.last_error = str(exc)
                self.completed_at = datetime.now(timezone.utc)
                self.last_result_at = None
                self.thread = None
                self.stop_event = None

        finally:
            if capture is not None:
                capture.release()

    def set_detection_sink(self, sink: Callable[[DetectionCapture], None] | None) -> None:
        with self.lock:
            self.detection_sink = sink

    def _emit_detection_capture(
        self,
        frame: Any,
        observation: ZoneObservation,
        bag_index: int,
    ) -> None:
        with self.lock:
            sink = self.detection_sink
        if sink is None:
            return

        try:
            image = self._annotate_detection_frame(
                frame=frame,
                candidate=observation.candidate,
                bag_index=bag_index,
                fill=float(observation.fill),
                frame_position=int(observation.frame_position),
            )
            if image is None:
                return

            ok, buffer = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.detection_jpeg_quality],
            )
            if not ok:
                return

            sink(
                DetectionCapture(
                    bag_index=int(bag_index),
                    frame_position=int(observation.frame_position),
                    fill=float(observation.fill),
                    captured_at=datetime.now(timezone.utc),
                    image_jpeg=buffer.tobytes(),
                    image_width=int(image.shape[1]),
                    image_height=int(image.shape[0]),
                )
            )
        except Exception:
            return

    def _annotate_detection_frame(
        self,
        frame: Any,
        candidate: BagCandidate | None,
        bag_index: int,
        fill: float,
        frame_position: int,
    ) -> Any:
        if frame is None or not hasattr(frame, "shape"):
            return None

        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return None

        if candidate is not None:
            x1 = max(int(candidate.x1), 0)
            y1 = max(int(candidate.y1), 0)
            x2 = min(int(candidate.x2), frame_width - 1)
            y2 = min(int(candidate.y2), frame_height - 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 128, 255), 3)

        cv2.putText(
            annotated,
            f"bag {bag_index} | fill {fill:.3f} | frame {frame_position}",
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (32, 128, 255),
            3,
            cv2.LINE_AA,
        )

        if frame_width > self.detection_capture_width:
            scale = self.detection_capture_width / float(frame_width)
            annotated = cv2.resize(
                annotated,
                (self.detection_capture_width, max(int(round(frame_height * scale)), 1)),
                interpolation=cv2.INTER_AREA,
            )

        return annotated

    def _get_model(self) -> YOLO:
        with self.lock:
            if self.model is not None:
                return self.model

        model = YOLO(str(self.model_path))

        with self.lock:
            if self.model is None:
                self.model = model
                self.class_names = [
                    str(name)
                    for _, name in sorted(self._normalize_names(getattr(model, "names", {})).items())
                ]
            return self.model

    def _load_class_names(self) -> dict[int, str]:
        try:
            model = self._get_model()
        except Exception:
            return {}
        return self._normalize_names(getattr(model, "names", {}))

    def _build_zone_counter(self) -> HopperZoneEpisodeCounter:
        return HopperZoneEpisodeCounter(
            zone=self.zone_above_hopper,
            control_zone=self.zone_control,
            gray_level=self.zone_gray_level,
            saturation_max=self.zone_saturation_max,
            fill_enter=self.zone_fill_enter,
            fill_stay=self.zone_fill_stay,
            control_ratio=self.zone_control_ratio,
            baseline_window=self.zone_baseline_window,
            baseline_percentile=self.zone_baseline_percentile,
            baseline_min_samples=self.zone_baseline_min_samples,
            baseline_margin=self.zone_baseline_margin,
            min_present_hits=self.zone_min_present_hits,
            min_present_seconds=self.zone_min_present_seconds,
            absence_seconds=self.zone_absence_seconds,
            sample_period_seconds=self.zone_sample_period_seconds,
            max_commits_per_hour=self.zone_max_commits_per_hour,
            max_episode_seconds=self.zone_max_episode_seconds,
            stuck_clear_seconds=self.zone_stuck_clear_seconds,
            max_tracked_episodes=self.zone_max_tracked_episodes,
        )

    def _resolve_capture_fps(
        self,
        capture: cv2.VideoCapture,
        source: str,
        source_kind: str,
        stop_event: threading.Event | None = None,
    ) -> tuple[float, str]:
        declared_fps = 0.0
        try:
            declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            declared_fps = 0.0

        if source_kind == "rtsp":
            measured_fps = self._measure_live_fps(capture=capture, stop_event=stop_event)
        else:
            measured_fps = self._measure_file_fps(source=source)

        return self._select_fps(measured_fps=measured_fps, declared_fps=declared_fps)

    def _measure_live_fps(
        self,
        capture: cv2.VideoCapture,
        stop_event: threading.Event | None,
    ) -> float:
        # Warm up FIRST and start the clock only once a picture has actually
        # arrived.  Charging the connect handshake and the wait for the first
        # keyframe to elapsed time can only ever UNDER-read the rate, and an
        # under-read shrinks the stride, which is the unsafe direction.
        warmup_deadline = time.monotonic() + self.fps_warmup_seconds
        while True:
            if stop_event is not None and stop_event.is_set():
                return 0.0
            if capture.grab():
                break
            if time.monotonic() >= warmup_deadline:
                return 0.0

        started = time.monotonic()
        decoded = 0

        while decoded < self.fps_probe_frames:
            if stop_event is not None and stop_event.is_set():
                break
            if time.monotonic() - started >= self.fps_probe_seconds:
                break
            if not capture.grab():
                break
            decoded += 1

        elapsed = time.monotonic() - started
        if decoded >= 2 and elapsed >= 0.5:
            return decoded / elapsed
        return 0.0

    def _measure_file_fps(self, source: str) -> float:
        capture = cv2.VideoCapture(source)
        try:
            if not capture.isOpened():
                return 0.0

            start_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            decoded = 0
            for _ in range(self.fps_file_probe_frames):
                if not capture.grab():
                    break
                decoded += 1

            end_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if decoded >= 2 and end_msec > start_msec:
                return decoded * 1000.0 / (end_msec - start_msec)
        except Exception:
            return 0.0
        finally:
            capture.release()

        return 0.0

    def _select_fps(self, measured_fps: float, declared_fps: float) -> tuple[float, str]:
        for value, origin in ((measured_fps, "measured"), (declared_fps, "declared")):
            if value and self.fps_min_plausible <= float(value) <= self.fps_max_plausible:
                return float(value), origin
        return float(self.fps_fallback), "fallback"

    @staticmethod
    def _decimate_series(series: list[int], limit: int) -> list[int]:
        total = len(series)
        budget = max(int(limit), 2)
        if total <= budget:
            return list(series)
        step = total / float(budget)
        return [series[min(int(index * step), total - 1)] for index in range(budget)]

    def _resolve_sample_stride(self, fps_effective: float, sample_period_seconds: float) -> int:
        stride = int(round(float(fps_effective) * float(sample_period_seconds)))
        return max(min(stride, self.max_sample_stride), 1)

    def _publish_stream_profile(
        self,
        run_id: int,
        fps_effective: float,
        fps_source: str,
        frame_stride: int,
        sample_period_seconds: float,
    ) -> None:
        with self.lock:
            if run_id != self.run_id:
                return

            self.fps_effective = float(fps_effective)
            self.fps_source = str(fps_source)
            self.active_frame_stride = int(frame_stride)
            self.active_sample_period_seconds = float(sample_period_seconds)

    def _resolve_source_kind(self, source: str) -> str:
        normalized = source.lower()
        if normalized.startswith("rtsp://") or normalized.startswith("rtsps://"):
            return "rtsp"
        return "file"

    def _wait_for_capture(
        self,
        run_id: int,
        source: str,
        source_kind: str,
        stop_event: threading.Event,
    ) -> cv2.VideoCapture | None:
        while not stop_event.is_set():
            try:
                return self._open_capture(
                    source=source,
                    source_kind=source_kind,
                    stop_event=stop_event,
                )
            except RuntimeError:
                if stop_event.is_set():
                    return None
                if source_kind != "rtsp":
                    raise
                self._set_waiting_state(run_id=run_id, source_kind=source_kind)
                time.sleep(self.live_read_retry_seconds)

        return None

    def _set_waiting_state(self, run_id: int, source_kind: str) -> None:
        with self.lock:
            if run_id != self.run_id:
                return

            self.state = "waiting"
            if self.bag_count is None:
                self.message = "Ожидание RTSP-источника. Подсчет мешков, опускаемых в бункер, продолжится, когда поток снова станет доступен."
            else:
                self.message = (
                    f"RTSP-источник временно недоступен. Сохраняю счет мешков, опущенных в бункер: {self.bag_count}, "
                    "пока жду переподключения. Счет и эпизод не сбрасываются."
                )
            self.completed_at = None
            self.last_error = None
            self.source_kind = source_kind

    def _open_capture(
        self,
        source: str,
        source_kind: str,
        stop_event: threading.Event,
    ) -> cv2.VideoCapture:
        deadline = time.monotonic() + self.live_open_timeout_seconds

        while not stop_event.is_set():
            if source_kind == "rtsp":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            capture = cv2.VideoCapture(source)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return capture

            capture.release()
            if time.monotonic() >= deadline:
                break
            time.sleep(self.live_read_retry_seconds)

        if stop_event.is_set():
            raise RuntimeError("Аналитика остановлена до открытия источника видео.")
        raise RuntimeError(f"Не удалось открыть источник {source_kind.upper()} для аналитики: {source}")

    def _predict_frame_detection_from_frame(
        self,
        model: YOLO,
        frame: Any,
        frame_position: int,
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
    ) -> FrameDetection:
        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )
        result = results[0] if results else None
        count, confidence_sum, top_confidence = self._extract_metrics(result, bag_class_ids)
        return FrameDetection(
            frame_position=frame_position,
            count=count,
            confidence_sum=confidence_sum,
            top_confidence=top_confidence,
        )

    def _predict_tiled_detections_from_frame(
        self,
        model: YOLO,
        frame: Any,
        frame_position: int,
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
    ) -> list[SpatialDetection]:
        frame_height, frame_width = frame.shape[:2]
        grid = 4
        tile_width = max(frame_width // grid, 1)
        tile_height = max(frame_height // grid, 1)
        step_x = max(tile_width // 2, 1)
        step_y = max(tile_height // 2, 1)

        detections: list[SpatialDetection] = []
        for y in range(0, max(frame_height - tile_height, 0) + 1, step_y):
            for x in range(0, max(frame_width - tile_width, 0) + 1, step_x):
                tile = frame[y:y + tile_height, x:x + tile_width]
                results = model.predict(
                    source=tile,
                    conf=confidence,
                    imgsz=imgsz,
                    device="cpu",
                    verbose=False,
                )
                result = results[0] if results else None
                detections.extend(
                    self._extract_spatial_detections(
                        result=result,
                        bag_class_ids=bag_class_ids,
                        frame_position=frame_position,
                        offset_x=x,
                        offset_y=y,
                    )
                )

        return detections

    def _summarize_counts(
        self,
        evidence: list[FrameDetection],
        detections: list[SpatialDetection],
        frame_width: int,
        frame_height: int,
        frame_positions: list[int],
    ) -> tuple[int, int, list[int], int]:
        sorted_evidence = sorted(evidence, key=lambda item: item.frame_position)
        evidence_count, evidence_support = self._resolve_count(sorted_evidence)
        evidence_peak = max((item.count for item in sorted_evidence), default=0)
        tiled_clusters = self._resolve_tiled_clusters(
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        if tiled_clusters:
            sample_counts = self._build_frame_cluster_counts(tiled_clusters, frame_positions)
            peak_sample_count = max(sample_counts, default=0)
            final_count = max(evidence_count, peak_sample_count)
            support_frames = sum(
                1 for count in sample_counts if count > 0 and abs(count - final_count) <= 1
            )
            max_count = max(peak_sample_count, evidence_peak)
        else:
            final_count = evidence_count
            support_frames = evidence_support
            max_count = evidence_peak
            sample_counts = [int(item.count) for item in sorted_evidence]

        return int(final_count), int(max_count), sample_counts, int(support_frames)

    def _build_stall_message(self, seconds: float, final_count: int) -> str:
        return (
            f"Картинка в RTSP-потоке не меняется уже {int(seconds)} с - похоже, источник "
            f"завис или отдаёт застывший кадр. Подсчёт приостановлен, учтено {final_count} "
            "мешк(ов). Проверьте камеру и сеть."
        )

    def _build_live_message(
        self,
        final_count: int,
        support_frames: int,
        sampled_frames: int,
        episode_state: str,
        pending_hits: int,
        degraded: bool = False,
    ) -> str:
        if degraded:
            return (
                "Аналитика RTSP-потока работает в безопасном режиме: превышен лимит "
                f"{self.zone_max_commits_per_hour} выгрузок в час, счет остановлен на {final_count}. "
                "Проверьте кадр и зону над бункером."
            )
        if final_count <= 0 and pending_hits <= 0:
            return (
                "Аналитика RTSP-потока работает. Мешок над бункером пока не обнаружен."
                if sampled_frames == 0
                else (
                    f"Аналитика RTSP-потока работает. На последних {sampled_frames} пробах зоны над бункером "
                    "мешок не обнаружен."
                )
            )
        if final_count <= 0:
            return (
                "Аналитика RTSP-потока работает. Мешок вошел в зону над бункером, "
                "жду подтверждение эпизода выгрузки."
            )
        if episode_state == "active":
            return (
                f"Аналитика RTSP-потока работает. Учтено {final_count} мешк(ов), опущенных в бункер; "
                f"текущая выгрузка продолжается. На последних {sampled_frames} пробах зоны "
                f"мешок виден на {support_frames}."
            )
        return (
            f"Аналитика RTSP-потока работает. Учтено {final_count} мешк(ов), опущенных в бункер; "
            f"на последних {sampled_frames} пробах зоны мешок виден на {support_frames}. "
            "Жду следующий мешок."
        )

    def _scan_video_zone(
        self,
        capture: cv2.VideoCapture,
        frame_stride: int,
        fps_effective: float,
        debug_output_dir: Path | None,
        debug_frame_limit: int,
        stop_event: threading.Event,
    ) -> tuple[HopperZoneEpisodeCounter, list[int], list[int], int, list[dict[str, Any]]]:
        counter = self._build_zone_counter()
        clock = SampleClock(source_kind="file", fps_effective=fps_effective)
        summary_positions: list[int] = []
        sample_counts: list[int] = []
        debug_frames: list[dict[str, Any]] = []
        pending_debug: dict[str, Any] | None = None
        stream_frame_index = 0
        debug_limit = max(int(debug_frame_limit), 0) if debug_output_dir is not None else 0

        while not stop_event.is_set():
            is_sample = stream_frame_index % frame_stride == 0
            if is_sample:
                ok, frame = capture.read()
            else:
                ok = capture.grab()
                frame = None

            if not ok or (is_sample and frame is None):
                break

            sequential_index = stream_frame_index
            stream_frame_index += 1
            if not is_sample:
                continue

            timestamp = clock.stamp(capture=capture, sequential_index=sequential_index)
            observation = counter.observe(
                frame=frame,
                frame_position=sequential_index,
                timestamp=timestamp,
            )
            summary_positions.append(int(sequential_index))
            sample_counts.append(1 if observation.present else 0)

            if debug_limit <= 0 or counter.state != "active" or observation.candidate is None:
                continue

            episode_index = int(counter.count)
            if episode_index <= 0 or episode_index > debug_limit:
                continue
            if pending_debug is None or int(pending_debug["episode_index"]) != episode_index:
                self._flush_zone_debug_frame(
                    pending=pending_debug,
                    output_dir=debug_output_dir,
                    debug_frames=debug_frames,
                )
                pending_debug = {
                    "episode_index": episode_index,
                    "frame_position": int(sequential_index),
                    "fill": float(observation.fill),
                    "frame": frame.copy(),
                    "candidate": observation.candidate,
                }
            elif float(observation.fill) > float(pending_debug["fill"]):
                pending_debug = {
                    "episode_index": episode_index,
                    "frame_position": int(sequential_index),
                    "fill": float(observation.fill),
                    "frame": frame.copy(),
                    "candidate": observation.candidate,
                }

        counter.finish()
        self._flush_zone_debug_frame(
            pending=pending_debug,
            output_dir=debug_output_dir,
            debug_frames=debug_frames,
        )
        return counter, summary_positions, sample_counts, stream_frame_index, debug_frames

    def _flush_zone_debug_frame(
        self,
        pending: dict[str, Any] | None,
        output_dir: Path | None,
        debug_frames: list[dict[str, Any]],
    ) -> None:
        if pending is None or output_dir is None:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        candidate: BagCandidate = pending["candidate"]
        frame_position = int(pending["frame_position"])
        episode_index = int(pending["episode_index"])
        annotated = pending["frame"]

        x1 = max(int(candidate.x1), 0)
        y1 = max(int(candidate.y1), 0)
        x2 = min(int(candidate.x2), annotated.shape[1] - 1)
        y2 = min(int(candidate.y2), annotated.shape[0] - 1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 128, 255), 3)
        label = f"hopper zone episode {episode_index} | fill {float(pending['fill']):.3f}"
        label_origin_y = y1 - 10 if y1 > 28 else y1 + 24
        cv2.putText(
            annotated,
            label,
            (x1, label_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (32, 128, 255),
            2,
            cv2.LINE_AA,
        )

        summary_label = f"frame {frame_position} | bag {episode_index}"
        cv2.putText(
            annotated,
            summary_label,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (24, 24, 24),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            summary_label,
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        filename = f"debug-{episode_index:02d}-frame-{frame_position}.jpg"
        if not cv2.imwrite(str(output_dir / filename), annotated):
            return

        debug_frames.append(
            {
                "filename": filename,
                "frame_position": frame_position,
                "bag_count": 1,
                "cluster_confidences": [round(float(pending["fill"]), 4)],
                "episode_index": episode_index,
                "source": "hopper-zone",
            }
        )

    def analyze_video_file(
        self,
        source: str,
        debug_output_dir: Path | None = None,
        debug_frame_limit: int = 4,
    ) -> dict[str, Any]:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Видеофайл не найден: {source}")

        started_at = datetime.now(timezone.utc)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Не удалось открыть видеофайл для аналитики: {source}")

        stop_event = threading.Event()

        try:
            names_map = self._load_class_names()
            bag_class_ids = self._resolve_bag_class_ids(names_map)

            declared_frame_count = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0)
            declared_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_width = max(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), 1)
            frame_height = max(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), 1)

            fps_effective, fps_source = self._resolve_capture_fps(
                capture=capture,
                source=str(video_path),
                source_kind="file",
            )
            frame_stride = self._resolve_sample_stride(
                fps_effective=fps_effective,
                sample_period_seconds=self.zone_sample_period_seconds,
            )

            counter, summary_positions, sample_counts, frame_count, debug_frames = self._scan_video_zone(
                capture=capture,
                frame_stride=frame_stride,
                fps_effective=fps_effective,
                debug_output_dir=debug_output_dir,
                debug_frame_limit=debug_frame_limit,
                stop_event=stop_event,
            )

            episodes = counter.episodes
            support_frames = sum(sample_counts)
            sampled_frames = len(summary_positions)
            final_count = int(counter.count)
            max_count = int(counter.count)
            candidate_positions = [int(episode["start_frame"]) for episode in episodes]
            # The payload is re-serialised on every status poll, so the
            # per-sample series are decimated to a fixed budget instead of
            # growing linearly with the length of the uploaded recording.
            summary_positions = self._decimate_series(summary_positions, self.video_summary_points)
            sample_counts = self._decimate_series(sample_counts, self.video_summary_points)
            structured_sample_counts = sample_counts.copy()
            episode_payload = [
                {
                    "index": int(episode["index"]),
                    "start_frame": int(episode["start_frame"]),
                    "end_frame": int(episode["end_frame"]),
                    "start_seconds": round(float(episode["start_timestamp"]), 2),
                    "end_seconds": round(float(episode["end_timestamp"]), 2),
                    "hit_count": int(episode["hit_count"]),
                    "peak_score": round(float(episode["peak_fill"]), 4),
                    "representative_frame": int(episode["representative_frame"]),
                }
                for episode in episodes
            ]

            completed_at = datetime.now(timezone.utc)
            duration_seconds = round((completed_at - started_at).total_seconds(), 2)

            return {
                "enabled": self.model_path.exists(),
                "state": "done",
                "message": (
                    f"Обнаружено {final_count} мешк(ов), опущенных краном в бункер, "
                    "по эпизодам заполнения зоны над бункером; "
                    f"подтверждено на {support_frames} пробах из {sampled_frames}."
                    if final_count > 0
                    else "Мешки, опускаемые в бункер, в загруженном видео не обнаружены."
                ),
                "bag_count": int(final_count),
                "max_bag_count": int(max_count),
                "frames_processed": sampled_frames,
                "sample_counts": sample_counts,
                "class_names": [str(name) for _, name in sorted(names_map.items())],
                "source_kind": "file",
                "count_mode": "hopper_zone_episodes",
                "target_zone": "above_hopper",
                "fps": round(fps_effective, 3),
                "fps_source": fps_source,
                "declared_fps": round(declared_fps, 3),
                "frame_stride": int(frame_stride),
                "sample_period_seconds": round(float(self.zone_sample_period_seconds), 3),
                "frame_count": int(frame_count),
                "declared_frame_count": int(declared_frame_count),
                "frame_width": int(frame_width),
                "frame_height": int(frame_height),
                "candidate_positions": candidate_positions,
                "summary_positions": summary_positions,
                "structured_sample_counts": structured_sample_counts,
                "episodes": episode_payload,
                "debug_frames": debug_frames,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "last_error": None,
            }

            coarse_positions = self._build_positions(
                frame_count=frame_count,
                start_ratio=0.02,
                end_ratio=0.98,
                samples=self.coarse_samples,
            )
            coarse_results = self._scan_positions(
                model=model,
                capture=capture,
                positions=coarse_positions,
                confidence=self.coarse_confidence,
                imgsz=self.coarse_imgsz,
                bag_class_ids=bag_class_ids,
                stop_event=stop_event,
            )

            ranked_coarse_results = self._rank_frame_detections(coarse_results)
            positive_coarse_positions = self._deduplicate_positions(
                positions=[item.frame_position for item in ranked_coarse_results],
                max_position=frame_count - 1,
                limit=max(self.candidate_windows * 2, 8),
            )

            candidate_positions = self._select_candidate_positions(
                results=coarse_results,
                frame_count=frame_count,
            )

            fine_positions: list[int] = []
            if candidate_positions:
                window_radius = max(int(frame_count * 0.03), 900)
                for candidate in candidate_positions:
                    fine_positions.extend(
                        self._build_window_positions(
                            start_frame=max(candidate - window_radius, 0),
                            end_frame=min(candidate + window_radius, frame_count - 1),
                            samples=self.fine_samples,
                        )
                    )
                fine_positions.extend(positive_coarse_positions)
            else:
                fine_positions.extend(positive_coarse_positions[: self.fine_samples])

            if not fine_positions:
                fine_positions = coarse_positions[: self.fine_samples]

            fine_positions = self._deduplicate_positions(
                positions=fine_positions,
                max_position=frame_count - 1,
                limit=max(self.candidate_windows * self.fine_samples, 12),
            )
            fine_results = self._scan_positions(
                model=model,
                capture=capture,
                positions=fine_positions,
                confidence=self.fine_confidence,
                imgsz=self.fine_imgsz,
                bag_class_ids=bag_class_ids,
                stop_event=stop_event,
            )

            ranked_fine_results = self._rank_frame_detections(fine_results)
            fallback_anchor_positions = self._deduplicate_positions(
                positions=[
                    *[item.frame_position for item in ranked_fine_results],
                    *positive_coarse_positions,
                ],
                max_position=frame_count - 1,
                limit=max(self.candidate_windows * 2, 8),
            )
            fallback_positions = self._deduplicate_positions(
                positions=[
                    *fallback_anchor_positions,
                    *self._expand_anchor_positions(
                        anchor_positions=fallback_anchor_positions,
                        frame_count=frame_count,
                        radius=max(int(frame_count * 0.008), 300),
                        samples_per_anchor=3,
                    ),
                ],
                max_position=frame_count - 1,
                limit=max(self.candidate_windows * 3, 12),
            )
            fallback_results = self._scan_positions(
                model=model,
                capture=capture,
                positions=fallback_positions,
                confidence=self.fallback_confidence,
                imgsz=max(self.fine_imgsz, 1280),
                bag_class_ids=bag_class_ids,
                stop_event=stop_event,
            )

            evidence = self._merge_frame_detections(fine_results, fallback_results) or coarse_results
            evidence_positions = [item.frame_position for item in sorted(evidence, key=lambda item: item.frame_position)]
            ranked_evidence_results = self._rank_frame_detections(evidence)

            tiled_positions = self._deduplicate_positions(
                positions=[
                    *[
                        item.frame_position
                        for item in ranked_evidence_results
                    ],
                    *candidate_positions,
                    *fallback_anchor_positions,
                    *positive_coarse_positions,
                ],
                max_position=frame_count - 1,
                limit=max(self.candidate_windows * 2, 8),
            )

            if not tiled_positions:
                tiled_positions = evidence_positions[: min(len(evidence_positions), 6)]

            tiled_detections = self._scan_tiled_positions(
                model=model,
                capture=capture,
                positions=tiled_positions,
                confidence=self.tile_confidence,
                imgsz=self.fine_imgsz,
                bag_class_ids=bag_class_ids,
                stop_event=stop_event,
            )

            summary_positions = sorted({*evidence_positions, *tiled_positions})
            if not summary_positions:
                summary_positions = evidence_positions

            final_count, max_count, sample_counts, support_frames = self._summarize_counts(
                evidence=evidence,
                detections=tiled_detections,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_positions=summary_positions,
            )

            structured_candidates = self._detect_structured_bag_candidates(
                capture=capture,
                positions=summary_positions,
            )
            structured_results = [
                FrameDetection(
                    frame_position=frame_position,
                    count=len(candidates),
                    confidence_sum=float(sum(candidate.score for candidate in candidates)),
                    top_confidence=float(max((candidate.score for candidate in candidates), default=0.0)),
                )
                for frame_position, candidates in sorted(structured_candidates.items())
            ]
            structured_count, structured_support = self._resolve_count(structured_results)
            structured_sample_counts = [
                len(structured_candidates.get(frame_position, []))
                for frame_position in summary_positions
            ]

            if structured_count >= final_count and structured_support > 0:
                final_count = int(structured_count)
                max_count = max(int(max_count), int(structured_count))
                sample_counts = structured_sample_counts
                support_frames = int(structured_support)

            debug_frames: list[dict[str, Any]] = []
            if debug_output_dir is not None:
                if structured_candidates and structured_count > 0:
                    debug_frames = self._save_candidate_debug_frames(
                        capture=capture,
                        frame_candidates=structured_candidates,
                        output_dir=debug_output_dir,
                        max_frames=debug_frame_limit,
                    )
                else:
                    debug_frames = self._save_debug_frames(
                        capture=capture,
                        detections=tiled_detections,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        output_dir=debug_output_dir,
                        max_frames=debug_frame_limit,
                    )

            completed_at = datetime.now(timezone.utc)
            duration_seconds = round((completed_at - started_at).total_seconds(), 2)

            return {
                "enabled": self.model_path.exists(),
                "state": "done",
                "message": (
                    f"Обнаружено {final_count} мешк(ов) в загруженном видео, "
                    f"подтверждено на {support_frames} обработанных кадрах."
                    if final_count > 0
                    else "Мешки в загруженном видео не обнаружены."
                ),
                "bag_count": int(final_count),
                "max_bag_count": int(max_count),
                "frames_processed": len(summary_positions),
                "sample_counts": sample_counts,
                "class_names": [str(name) for _, name in sorted(names_map.items())],
                "source_kind": "file",
                "frame_count": int(frame_count),
                "frame_width": int(frame_width),
                "frame_height": int(frame_height),
                "candidate_positions": candidate_positions,
                "summary_positions": summary_positions,
                "structured_sample_counts": structured_sample_counts,
                "debug_frames": debug_frames,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "duration_seconds": duration_seconds,
                "last_error": None,
            }
        finally:
            capture.release()

    def _build_positions(
        self,
        frame_count: int,
        start_ratio: float,
        end_ratio: float,
        samples: int,
    ) -> list[int]:
        start_frame = int(frame_count * start_ratio)
        end_frame = int(frame_count * end_ratio)
        return self._build_window_positions(start_frame, end_frame, samples)

    def _build_window_positions(
        self,
        start_frame: int,
        end_frame: int,
        samples: int,
    ) -> list[int]:
        if end_frame <= start_frame:
            return [max(start_frame, 0)]

        step = max((end_frame - start_frame) // max(samples - 1, 1), 1)
        positions = list(range(start_frame, end_frame + 1, step))
        if positions[-1] != end_frame:
            positions.append(end_frame)

        deduplicated: list[int] = []
        seen: set[int] = set()
        for position in positions:
            if position not in seen:
                deduplicated.append(position)
                seen.add(position)

        return deduplicated[:samples] if len(deduplicated) > samples else deduplicated

    def _build_video_event_positions(
        self,
        frame_count: int,
        fps: float,
    ) -> list[int]:
        max_position = max(frame_count - 1, 0)
        sample_step = self._resolve_video_event_sample_step(fps=fps)
        positions = list(range(0, frame_count, sample_step))
        if not positions:
            positions = [0]
        if positions[-1] != max_position:
            positions.append(max_position)
        return self._deduplicate_positions(
            positions=positions,
            max_position=max_position,
        )

    def _resolve_video_event_sample_step(self, fps: float) -> int:
        normalized_fps = fps if fps and fps > 0 else 25.0
        return max(int(round(normalized_fps * self.video_event_sample_seconds)), 1)

    def _resolve_video_event_merge_gap_frames(self, fps: float) -> int:
        normalized_fps = fps if fps and fps > 0 else 25.0
        return max(
            int(round(normalized_fps * self.video_event_merge_gap_seconds)),
            self._resolve_video_event_sample_step(fps=fps),
        )

    def _resolve_live_event_gap_samples(self, stream_fps: float) -> int:
        if stream_fps and stream_fps > 0:
            sampled_fps = max(stream_fps / max(self.live_frame_stride, 1), 1e-6)
            return max(
                int(round(sampled_fps * self.video_event_merge_gap_seconds)),
                self.live_event_gap_samples,
            )
        return self.live_event_gap_samples

    def _deduplicate_positions(
        self,
        positions: list[int],
        max_position: int,
        limit: int | None = None,
    ) -> list[int]:
        deduplicated: list[int] = []
        seen: set[int] = set()

        for position in positions:
            normalized = min(max(int(position), 0), max_position)
            if normalized in seen:
                continue
            deduplicated.append(normalized)
            seen.add(normalized)
            if limit is not None and len(deduplicated) >= limit:
                break

        return deduplicated

    def _rank_frame_detections(self, results: list[FrameDetection]) -> list[FrameDetection]:
        return sorted(
            [item for item in results if item.count > 0 or item.top_confidence > 0],
            key=lambda item: (item.count, item.confidence_sum, item.top_confidence),
            reverse=True,
        )

    def _merge_frame_detections(
        self,
        primary: list[FrameDetection],
        secondary: list[FrameDetection],
    ) -> list[FrameDetection]:
        merged: dict[int, FrameDetection] = {}

        for item in [*primary, *secondary]:
            existing = merged.get(item.frame_position)
            if existing is None or (
                item.count,
                item.confidence_sum,
                item.top_confidence,
            ) > (
                existing.count,
                existing.confidence_sum,
                existing.top_confidence,
            ):
                merged[item.frame_position] = item

        return [merged[position] for position in sorted(merged)]

    def _expand_anchor_positions(
        self,
        anchor_positions: list[int],
        frame_count: int,
        radius: int,
        samples_per_anchor: int,
    ) -> list[int]:
        if not anchor_positions:
            return []

        expanded: list[int] = []
        max_position = max(frame_count - 1, 0)

        for anchor in anchor_positions:
            expanded.extend(
                self._build_window_positions(
                    start_frame=max(anchor - radius, 0),
                    end_frame=min(anchor + radius, max_position),
                    samples=max(samples_per_anchor, 2),
                )
            )

        return self._deduplicate_positions(
            positions=expanded,
            max_position=max_position,
        )

    def _build_frame_cluster_map(
        self,
        detections: list[SpatialDetection],
        frame_width: int,
        frame_height: int,
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[SpatialDetection]] = {}
        for detection in detections:
            grouped.setdefault(detection.frame_position, []).append(detection)

        frame_clusters: dict[int, list[dict[str, Any]]] = {}
        for frame_position, frame_detections in grouped.items():
            clusters = self._resolve_tiled_clusters(
                detections=frame_detections,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if clusters:
                frame_clusters[frame_position] = clusters

        return frame_clusters

    def _detect_top_bag_candidates(
        self,
        capture: cv2.VideoCapture,
        positions: list[int],
    ) -> dict[int, BagCandidate]:
        candidates_by_frame: dict[int, BagCandidate] = {}
        for frame_position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            candidate = self._detect_top_bag_candidate_from_frame(
                frame=frame,
                frame_position=frame_position,
            )
            if candidate is not None:
                candidates_by_frame[frame_position] = candidate

        return candidates_by_frame

    def _detect_top_bag_candidate_from_frame(
        self,
        frame: Any,
        frame_position: int,
    ) -> BagCandidate | None:
        fill = _zone_fill(
            frame,
            self.zone_above_hopper,
            self.zone_gray_level,
            self.zone_saturation_max,
        )
        if fill < self.zone_fill_enter:
            return None

        if self.zone_control_ratio > 0.0:
            control_fill = _zone_fill(
                frame,
                self.zone_control,
                self.zone_gray_level,
                self.zone_saturation_max,
            )
            if fill < self.zone_control_ratio * max(control_fill, 1e-6):
                return None

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = _zone_bounds(frame_width, frame_height, self.zone_above_hopper)
        return BagCandidate(
            frame_position=int(frame_position),
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            score=float(fill),
            band="top",
            source="hopper_zone",
        )

    def _build_top_bag_episodes(
        self,
        candidates: list[BagCandidate],
        merge_gap_frames: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        episodes: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for candidate in sorted(candidates, key=lambda item: item.frame_position):
            if current is None or candidate.frame_position - current["last_frame"] > merge_gap_frames:
                if current is not None:
                    episodes.append(current)
                current = {
                    "start_frame": candidate.frame_position,
                    "end_frame": candidate.frame_position,
                    "last_frame": candidate.frame_position,
                    "hit_count": 1,
                    "peak_score": float(candidate.score),
                    "representative": candidate,
                    "samples": [candidate],
                }
                continue

            current["end_frame"] = candidate.frame_position
            current["last_frame"] = candidate.frame_position
            current["hit_count"] += 1
            current["samples"].append(candidate)
            if float(candidate.score) >= float(current["peak_score"]):
                current["peak_score"] = float(candidate.score)
                current["representative"] = candidate

        if current is not None:
            episodes.append(current)

        filtered: list[dict[str, Any]] = []
        for episode in episodes:
            if episode["hit_count"] >= self.video_event_min_hits:
                filtered.append(episode)
                continue
            if float(episode["peak_score"]) >= self.video_event_min_single_score:
                filtered.append(episode)

        return filtered

    def _save_top_episode_debug_frames(
        self,
        capture: cv2.VideoCapture,
        episodes: list[dict[str, Any]],
        output_dir: Path,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        if not episodes:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_frames: list[dict[str, Any]] = []

        for index, episode in enumerate(episodes[: max(max_frames, 1)], start=1):
            candidate = episode["representative"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            annotated = frame.copy()
            x1 = max(int(candidate.x1), 0)
            y1 = max(int(candidate.y1), 0)
            x2 = min(int(candidate.x2), annotated.shape[1] - 1)
            y2 = min(int(candidate.y2), annotated.shape[0] - 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 128, 255), 3)
            label = f"top bag episode {index}"
            label_origin_y = y1 - 10 if y1 > 28 else y1 + 24
            cv2.putText(
                annotated,
                label,
                (x1, label_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (32, 128, 255),
                2,
                cv2.LINE_AA,
            )

            summary_label = f"frame {candidate.frame_position} | top bag {index}"
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (24, 24, 24),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            filename = f"debug-{index:02d}-frame-{candidate.frame_position}.jpg"
            frame_path = output_dir / filename
            if not cv2.imwrite(str(frame_path), annotated):
                continue

            saved_frames.append(
                {
                    "filename": filename,
                    "frame_position": int(candidate.frame_position),
                    "bag_count": 1,
                    "cluster_confidences": [round(float(candidate.score), 1)],
                    "episode_index": int(index),
                    "episode_start_frame": int(episode["start_frame"]),
                    "episode_end_frame": int(episode["end_frame"]),
                    "episode_hits": int(episode["hit_count"]),
                    "source": "top-episode",
                }
            )

        return saved_frames

    def _detect_structured_bag_candidates(
        self,
        capture: cv2.VideoCapture,
        positions: list[int],
    ) -> dict[int, list[BagCandidate]]:
        candidates_by_frame: dict[int, list[BagCandidate]] = {}
        for frame_position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            candidates = self._detect_structured_bag_candidates_from_frame(
                frame=frame,
                frame_position=frame_position,
            )
            if candidates:
                candidates_by_frame[frame_position] = candidates

        return candidates_by_frame

    def _detect_structured_bag_candidates_from_frame(
        self,
        frame: Any,
        frame_position: int,
    ) -> list[BagCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = gray.shape[:2]

        threshold_value = max(170, min(190, int(np.percentile(gray, 86))))
        _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        raw_candidates: list[dict[str, Any]] = []

        for label in range(1, num_labels):
            x, y, width, height, area = [int(value) for value in stats[label]]
            center_x, center_y = centroids[label]

            band: str | None = None
            if center_y < frame_height * 0.30:
                if area >= 12000 and height >= 180 and width >= 50:
                    band = "top"
            elif center_y > frame_height * 0.68:
                if area >= 35000 and height >= 180 and width >= 180:
                    band = "bottom"

            if band is None:
                continue

            raw_candidates.append(
                {
                    "band": band,
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x + width),
                    "y2": float(y + height),
                    "score": float(area),
                }
            )

        merged_candidates = self._merge_structured_candidates(raw_candidates)
        return [
            BagCandidate(
                frame_position=frame_position,
                x1=candidate["x1"],
                y1=candidate["y1"],
                x2=candidate["x2"],
                y2=candidate["y2"],
                score=float(candidate["score"]),
                band=str(candidate["band"]),
                source="structured",
            )
            for candidate in merged_candidates
        ]

    def _merge_structured_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        merged: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: (item["band"], item["x1"], item["y1"])):
            absorbed = False
            for target in merged:
                if target["band"] != candidate["band"]:
                    continue

                horizontal_gap = max(
                    0.0,
                    max(candidate["x1"], target["x1"]) - min(candidate["x2"], target["x2"]),
                )
                vertical_gap = max(
                    0.0,
                    max(candidate["y1"], target["y1"]) - min(candidate["y2"], target["y2"]),
                )
                vertical_overlap = max(
                    0.0,
                    min(candidate["y2"], target["y2"]) - max(candidate["y1"], target["y1"]),
                )
                min_height = max(
                    min(candidate["y2"] - candidate["y1"], target["y2"] - target["y1"]),
                    1.0,
                )

                if horizontal_gap <= 80.0 and (vertical_gap <= 80.0 or vertical_overlap / min_height >= 0.25):
                    target["x1"] = min(target["x1"], candidate["x1"])
                    target["y1"] = min(target["y1"], candidate["y1"])
                    target["x2"] = max(target["x2"], candidate["x2"])
                    target["y2"] = max(target["y2"], candidate["y2"])
                    target["score"] += candidate["score"]
                    absorbed = True
                    break

            if not absorbed:
                merged.append(dict(candidate))

        return merged

    def _save_candidate_debug_frames(
        self,
        capture: cv2.VideoCapture,
        frame_candidates: dict[int, list[BagCandidate]],
        output_dir: Path,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        if not frame_candidates:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        ranked_frames = sorted(
            frame_candidates.items(),
            key=lambda item: (
                len(item[1]),
                sum(candidate.score for candidate in item[1]),
                max(candidate.score for candidate in item[1]),
            ),
            reverse=True,
        )[: max(max_frames, 1)]

        saved_frames: list[dict[str, Any]] = []
        for index, (frame_position, candidates) in enumerate(ranked_frames, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            annotated = frame.copy()
            for candidate_index, candidate in enumerate(
                sorted(candidates, key=lambda item: item.score, reverse=True),
                start=1,
            ):
                x1 = max(int(candidate.x1), 0)
                y1 = max(int(candidate.y1), 0)
                x2 = min(int(candidate.x2), annotated.shape[1] - 1)
                y2 = min(int(candidate.y2), annotated.shape[0] - 1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 128, 255), 3)
                label = f"bag {candidate_index}"
                label_origin_y = y1 - 10 if y1 > 28 else y1 + 24
                cv2.putText(
                    annotated,
                    label,
                    (x1, label_origin_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (32, 128, 255),
                    2,
                    cv2.LINE_AA,
                )

            summary_label = f"frame {frame_position} | bags {len(candidates)}"
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (24, 24, 24),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            filename = f"debug-{index:02d}-frame-{frame_position}.jpg"
            frame_path = output_dir / filename
            if not cv2.imwrite(str(frame_path), annotated):
                continue

            saved_frames.append(
                {
                    "filename": filename,
                    "frame_position": int(frame_position),
                    "bag_count": int(len(candidates)),
                    "cluster_confidences": [
                        round(float(candidate.score), 1)
                        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)
                    ],
                    "source": "structured",
                }
            )

        return saved_frames

    def _save_debug_frames(
        self,
        capture: cv2.VideoCapture,
        detections: list[SpatialDetection],
        frame_width: int,
        frame_height: int,
        output_dir: Path,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        frame_clusters = self._build_frame_cluster_map(
            detections=detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if not frame_clusters:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        ranked_frames = sorted(
            frame_clusters.items(),
            key=lambda item: (
                len(item[1]),
                sum(cluster["conf_sum"] for cluster in item[1]),
                max(cluster["max_conf"] for cluster in item[1]),
            ),
            reverse=True,
        )[: max(max_frames, 1)]

        saved_frames: list[dict[str, Any]] = []
        for index, (frame_position, clusters) in enumerate(ranked_frames, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            annotated = frame.copy()
            for cluster_index, cluster in enumerate(
                sorted(clusters, key=lambda item: item["conf_sum"], reverse=True),
                start=1,
            ):
                x1 = max(int(cluster["x1"]), 0)
                y1 = max(int(cluster["y1"]), 0)
                x2 = min(int(cluster["x2"]), annotated.shape[1] - 1)
                y2 = min(int(cluster["y2"]), annotated.shape[0] - 1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (32, 128, 255), 3)
                label = f"bag {cluster_index} | conf {cluster['max_conf']:.2f}"
                label_origin_y = y1 - 10 if y1 > 28 else y1 + 24
                cv2.putText(
                    annotated,
                    label,
                    (x1, label_origin_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (32, 128, 255),
                    2,
                    cv2.LINE_AA,
                )

            summary_label = f"frame {frame_position} | bags {len(clusters)}"
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (24, 24, 24),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                summary_label,
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            filename = f"debug-{index:02d}-frame-{frame_position}.jpg"
            frame_path = output_dir / filename
            if not cv2.imwrite(str(frame_path), annotated):
                continue

            saved_frames.append(
                {
                    "filename": filename,
                    "frame_position": int(frame_position),
                    "bag_count": int(len(clusters)),
                    "cluster_confidences": [
                        round(float(cluster["max_conf"]), 3)
                        for cluster in sorted(clusters, key=lambda item: item["max_conf"], reverse=True)
                    ],
                }
            )

        return saved_frames

    def _scan_positions(
        self,
        model: YOLO,
        capture: cv2.VideoCapture,
        positions: list[int],
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
        stop_event: threading.Event,
    ) -> list[FrameDetection]:
        results: list[FrameDetection] = []
        for position in positions:
            if stop_event.is_set():
                return results

            detection = self._predict_frame_detection(
                model=model,
                capture=capture,
                frame_position=position,
                confidence=confidence,
                imgsz=imgsz,
                bag_class_ids=bag_class_ids,
            )
            if detection is not None:
                results.append(detection)
        return results

    def _scan_tiled_positions(
        self,
        model: YOLO,
        capture: cv2.VideoCapture,
        positions: list[int],
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
        stop_event: threading.Event,
    ) -> list[SpatialDetection]:
        detections: list[SpatialDetection] = []
        for position in positions:
            if stop_event.is_set():
                return detections

            detections.extend(
                self._predict_tiled_detections(
                    model=model,
                    capture=capture,
                    frame_position=position,
                    confidence=confidence,
                    imgsz=imgsz,
                    bag_class_ids=bag_class_ids,
                )
            )
        return detections

    def _predict_frame_detection(
        self,
        model: YOLO,
        capture: cv2.VideoCapture,
        frame_position: int,
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
    ) -> FrameDetection | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ok, frame = capture.read()
        if not ok:
            return None

        results = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            device="cpu",
            verbose=False,
        )
        result = results[0] if results else None
        count, confidence_sum, top_confidence = self._extract_metrics(result, bag_class_ids)
        return FrameDetection(
            frame_position=frame_position,
            count=count,
            confidence_sum=confidence_sum,
            top_confidence=top_confidence,
        )

    def _predict_tiled_detections(
        self,
        model: YOLO,
        capture: cv2.VideoCapture,
        frame_position: int,
        confidence: float,
        imgsz: int,
        bag_class_ids: set[int],
    ) -> list[SpatialDetection]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ok, frame = capture.read()
        if not ok:
            return []

        frame_height, frame_width = frame.shape[:2]
        grid = 4
        tile_width = max(frame_width // grid, 1)
        tile_height = max(frame_height // grid, 1)
        step_x = max(tile_width // 2, 1)
        step_y = max(tile_height // 2, 1)

        detections: list[SpatialDetection] = []
        for y in range(0, max(frame_height - tile_height, 0) + 1, step_y):
            for x in range(0, max(frame_width - tile_width, 0) + 1, step_x):
                tile = frame[y:y + tile_height, x:x + tile_width]
                results = model.predict(
                    source=tile,
                    conf=confidence,
                    imgsz=imgsz,
                    device="cpu",
                    verbose=False,
                )
                result = results[0] if results else None
                detections.extend(
                    self._extract_spatial_detections(
                        result=result,
                        bag_class_ids=bag_class_ids,
                        frame_position=frame_position,
                        offset_x=x,
                        offset_y=y,
                    )
                )

        return detections

    def _extract_metrics(self, result: Any, bag_class_ids: set[int]) -> tuple[int, float, float]:
        if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return 0, 0.0, 0.0

        boxes = result.boxes
        classes = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        confidences = boxes.conf.tolist() if getattr(boxes, "conf", None) is not None else []
        if not confidences:
            return 0, 0.0, 0.0

        if not classes or not bag_class_ids:
            matched_confidences = [float(confidence) for confidence in confidences]
        else:
            matched_confidences = [
                float(confidence)
                for class_id, confidence in zip(classes, confidences)
                if int(class_id) in bag_class_ids
            ]

        if not matched_confidences:
            return 0, 0.0, 0.0

        return (
            len(matched_confidences),
            float(sum(matched_confidences)),
            float(max(matched_confidences)),
        )

    def _extract_spatial_detections(
        self,
        result: Any,
        bag_class_ids: set[int],
        frame_position: int,
        offset_x: int,
        offset_y: int,
    ) -> list[SpatialDetection]:
        if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes
        if getattr(boxes, "xyxy", None) is None or getattr(boxes, "conf", None) is None:
            return []

        raw_boxes = boxes.xyxy.tolist()
        classes = boxes.cls.tolist() if getattr(boxes, "cls", None) is not None else []
        confidences = boxes.conf.tolist()
        detections: list[SpatialDetection] = []

        for index, (box, confidence) in enumerate(zip(raw_boxes, confidences)):
            class_id = int(classes[index]) if index < len(classes) else 0
            if bag_class_ids and class_id not in bag_class_ids:
                continue

            x1, y1, x2, y2 = (
                float(box[0]) + offset_x,
                float(box[1]) + offset_y,
                float(box[2]) + offset_x,
                float(box[3]) + offset_y,
            )
            detections.append(
                SpatialDetection(
                    frame_position=frame_position,
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections

    def _normalize_names(self, names: Any) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        if isinstance(names, list):
            return {index: str(value) for index, value in enumerate(names)}
        return {}

    def _resolve_bag_class_ids(self, names_map: dict[int, str]) -> set[int]:
        if len(names_map) == 1:
            return set(names_map.keys())

        return {
            class_id
            for class_id, name in names_map.items()
            if name.strip().lower() in BAG_LABEL_HINTS
        }

    def _resolve_tiled_clusters(
        self,
        detections: list[SpatialDetection],
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        if not detections:
            return []

        distance_threshold = max(frame_width, frame_height) * 0.12
        clusters: list[dict[str, Any]] = []

        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            assigned = False
            for cluster in clusters:
                dx = detection.center_x - cluster["center_x"]
                dy = detection.center_y - cluster["center_y"]
                if (dx * dx + dy * dy) ** 0.5 <= distance_threshold:
                    cluster["sum_x"] += detection.center_x
                    cluster["sum_y"] += detection.center_y
                    cluster["count"] += 1
                    cluster["conf_sum"] += detection.confidence
                    cluster["max_conf"] = max(cluster["max_conf"], detection.confidence)
                    cluster["center_x"] = cluster["sum_x"] / cluster["count"]
                    cluster["center_y"] = cluster["sum_y"] / cluster["count"]
                    cluster["x1"] = min(cluster["x1"], detection.x1)
                    cluster["y1"] = min(cluster["y1"], detection.y1)
                    cluster["x2"] = max(cluster["x2"], detection.x2)
                    cluster["y2"] = max(cluster["y2"], detection.y2)
                    cluster["frames"].add(detection.frame_position)
                    assigned = True
                    break

            if not assigned:
                clusters.append(
                    {
                        "center_x": detection.center_x,
                        "center_y": detection.center_y,
                        "sum_x": detection.center_x,
                        "sum_y": detection.center_y,
                        "count": 1,
                        "conf_sum": detection.confidence,
                        "max_conf": detection.confidence,
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                        "frames": {detection.frame_position},
                    }
                )

        clusters = [
            cluster
            for cluster in clusters
            if cluster["center_y"] > frame_height * 0.15
        ]
        clusters = [
            cluster
            for cluster in clusters
            if len(cluster["frames"]) >= 2 or cluster["count"] >= 2 or cluster["max_conf"] >= 0.18
        ]

        x_threshold = frame_width * 0.06
        y_threshold = frame_height * 0.20
        merged: list[dict[str, Any]] = []
        for cluster in sorted(clusters, key=lambda item: item["conf_sum"], reverse=True):
            absorbed = False
            for target in merged:
                dx = abs(cluster["center_x"] - target["center_x"])
                dy = abs(cluster["center_y"] - target["center_y"])
                if (
                    dx < x_threshold
                    and dy < y_threshold
                    and cluster["conf_sum"] < target["conf_sum"] * 0.5
                ):
                    target["count"] += cluster["count"]
                    target["conf_sum"] += cluster["conf_sum"]
                    target["max_conf"] = max(target["max_conf"], cluster["max_conf"])
                    target["sum_x"] += cluster["sum_x"]
                    target["sum_y"] += cluster["sum_y"]
                    target["center_x"] = target["sum_x"] / target["count"]
                    target["center_y"] = target["sum_y"] / target["count"]
                    target["x1"] = min(target["x1"], cluster["x1"])
                    target["y1"] = min(target["y1"], cluster["y1"])
                    target["x2"] = max(target["x2"], cluster["x2"])
                    target["y2"] = max(target["y2"], cluster["y2"])
                    target["frames"].update(cluster["frames"])
                    absorbed = True
                    break
            if not absorbed:
                merged.append(cluster)

        return merged

    def _build_frame_cluster_counts(
        self,
        clusters: list[dict[str, Any]],
        frame_positions: list[int],
    ) -> list[int]:
        frame_counts = {frame_position: 0 for frame_position in frame_positions}
        for cluster in clusters:
            for frame_position in cluster["frames"]:
                if frame_position in frame_counts:
                    frame_counts[frame_position] += 1
        return [int(frame_counts[frame_position]) for frame_position in frame_positions]

    def _select_candidate_positions(
        self,
        results: list[FrameDetection],
        frame_count: int,
    ) -> list[int]:
        ranked = [item for item in results if item.count > 0]
        if not ranked:
            return []

        ranked.sort(
            key=lambda item: (item.count, item.confidence_sum, item.top_confidence),
            reverse=True,
        )

        min_gap = max(int(frame_count * 0.04), 1200)
        selected: list[int] = []
        for item in ranked:
            if all(abs(item.frame_position - chosen) >= min_gap for chosen in selected):
                selected.append(item.frame_position)
            if len(selected) >= self.candidate_windows:
                break

        return selected

    def _resolve_count(self, results: list[FrameDetection]) -> tuple[int, int]:
        positive_counts = [item.count for item in results if item.count > 0]
        if not positive_counts:
            return 0, 0

        frequencies = Counter(positive_counts)
        supported_counts = [count for count, occurrences in frequencies.items() if occurrences >= 2]
        if supported_counts:
            selected_count = max(supported_counts)
        else:
            selected_count = max(positive_counts)

        return selected_count, frequencies[selected_count]

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "enabled": self.model_path.exists(),
            "model_path": str(self.model_path),
            "model_loaded": self.model is not None,
            "state": self.state,
            "message": self.message,
            "bag_count": self.bag_count,
            "max_bag_count": self.max_bag_count,
            "frames_processed": self.frames_processed,
            "sample_counts": self.sample_counts,
            "class_names": self.class_names,
            "source_kind": self.source_kind,
            "count_mode": (
                "hopper_zone_episodes"
                if self.source_kind == "file"
                else "hopper_zone_episodes_live"
                if self.source_kind == "rtsp"
                else None
            ),
            "target_zone": "above_hopper" if self.source_kind in {"file", "rtsp"} else None,
            "frame_stride": self.active_frame_stride,
            "sample_period_seconds": round(float(self.active_sample_period_seconds), 3),
            "fps_effective": (
                round(float(self.fps_effective), 3) if self.fps_effective is not None else None
            ),
            "fps_source": self.fps_source,
            "episode_state": self.episode_state,
            "stream_stalled": self.stream_stalled,
            "zone_fill": (
                round(float(self.last_zone_fill), 4) if self.last_zone_fill is not None else None
            ),
            "zone_baseline": (
                round(float(self.last_zone_baseline), 4)
                if self.last_zone_baseline is not None
                else None
            ),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_result_at": self.last_result_at.isoformat() if self.last_result_at else None,
            "last_error": self.last_error,
        }
