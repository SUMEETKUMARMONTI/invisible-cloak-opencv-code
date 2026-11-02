"""
====================================================================
                      INVISIBLE CLOAK PRO  v2.0
====================================================================
Real-time invisibility cloak using Python, OpenCV and NumPy.

Developer : Sumeet Kumar Monti
GitHub    : https://github.com/SUMEETKUMARMONTI
LinkedIn  : https://www.linkedin.com/in/sumeetkumarmonti932003/

Install   : pip install opencv-python numpy
Run       : python Invisible_Cloak.py
Help      : python Invisible_Cloak.py --help

Highlights
  * HSV segmentation with hue wrap-around (red cloth works)
  * Click on the cloth to auto-calibrate the colour range
  * Soft feathered edges, median-of-N-frames background
  * 4 effects: invisible, ghost, portal, neon
  * Recording, screenshots, one-key original/mask/result panel
  * Camera-drift warning, threaded low-latency camera reader
  * Headless mode for video files:  --video in.mp4 --headless -o out.mp4

Keys: click=sample colour  1-5/C=preset  E=effect  B=background  M=mask
      T=sliders  R=record  S=screenshot  P=panel  K=save settings
      SPACE=pause  I=HUD  H=help  F=fullscreen  Q/ESC=quit
====================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

__version__ = "2.0.0"
__author__ = "Sumeet Kumar Monti"

log = logging.getLogger("invisible_cloak")


# ==================================================================
#  CONFIGURATION, COLOUR PRESETS AND SETTINGS
# ==================================================================

HUE_MAX = 179
SV_MAX = 255

Bounds = Tuple[np.ndarray, np.ndarray]


def _clamp(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


@dataclass
class HSVRange:
    """An inclusive HSV colour range (OpenCV scale: H 0-179, S/V 0-255)."""

    h_lo: int = 50
    s_lo: int = 80
    v_lo: int = 50
    h_hi: int = 90
    s_hi: int = 255
    v_hi: int = 255

    def __post_init__(self) -> None:
        self.clamp()

    def clamp(self) -> None:
        """Force every channel into its legal OpenCV range."""
        self.h_lo = _clamp(self.h_lo, 0, HUE_MAX)
        self.h_hi = _clamp(self.h_hi, 0, HUE_MAX)
        for name in ("s_lo", "s_hi", "v_lo", "v_hi"):
            setattr(self, name, _clamp(getattr(self, name), 0, SV_MAX))

    @property
    def wraps(self) -> bool:
        """True when the hue interval crosses the 179 -> 0 boundary."""
        return self.h_lo > self.h_hi

    def cv_bounds(self) -> List[Bounds]:
        """Return one or two ``(lower, upper)`` pairs usable by ``cv2.inRange``."""
        if not self.wraps:
            return [
                (
                    np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8),
                    np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8),
                )
            ]
        return [
            (
                np.array([self.h_lo, self.s_lo, self.v_lo], dtype=np.uint8),
                np.array([HUE_MAX, self.s_hi, self.v_hi], dtype=np.uint8),
            ),
            (
                np.array([0, self.s_lo, self.v_lo], dtype=np.uint8),
                np.array([self.h_hi, self.s_hi, self.v_hi], dtype=np.uint8),
            ),
        ]

    @classmethod
    def from_sample(
        cls,
        h: float,
        s: float,
        v: float,
        hue_tol: int = 12,
        sat_tol: int = 70,
        val_tol: int = 80,
    ) -> "HSVRange":
        """Build a tolerant range around a sampled pixel colour."""
        return cls(
            h_lo=int(round(h - hue_tol)) % 180,
            h_hi=int(round(h + hue_tol)) % 180,
            s_lo=max(40, int(s) - sat_tol),
            s_hi=SV_MAX,
            v_lo=max(40, int(v) - val_tol),
            v_hi=SV_MAX,
        )

    def copy(self) -> "HSVRange":
        return HSVRange(**asdict(self))


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
_PRESETS: Dict[str, HSVRange] = {
    "green": HSVRange(50, 80, 50, 90, 255, 255),
    "blue": HSVRange(95, 100, 50, 130, 255, 255),
    "red": HSVRange(170, 120, 70, 10, 255, 255),  # wraps around hue 0
    "yellow": HSVRange(20, 100, 100, 35, 255, 255),
    "purple": HSVRange(130, 60, 50, 160, 255, 255),
}

PRESET_ORDER: Tuple[str, ...] = tuple(_PRESETS)


def get_preset(name: str) -> HSVRange:
    """Return a *copy* of a named preset."""
    try:
        return _PRESETS[name].copy()
    except KeyError:
        raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(PRESET_ORDER)}") from None


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
_PERSISTED = (
    "mirror",
    "preset",
    "effect",
    "min_area",
    "feather",
    "ghost_strength",
    "open_kernel",
    "close_kernel",
    "dilate_kernel",
)


@dataclass
class Settings:
    """Runtime settings.  A subset can be saved to / loaded from JSON."""

    # Input
    camera_index: int = 0
    source: Optional[str] = None  # path to a video file (overrides the camera)
    width: int = 1280
    height: int = 720
    mirror: bool = True
    threaded: bool = True

    # Colour model
    preset: str = "green"
    hsv: HSVRange = field(default_factory=lambda: get_preset("green"))

    # Effect
    effect: str = "invisible"
    ghost_strength: float = 0.65  # 0 = invisible effect off, 1 = fully hidden

    # Background capture
    countdown_seconds: int = 3
    background_frames: int = 15  # frames combined by a per-pixel median

    # Mask refinement
    min_area: int = 1500  # ignore blobs smaller than this many pixels
    open_kernel: int = 5
    close_kernel: int = 7
    dilate_kernel: int = 5
    feather: int = 11  # edge softening (odd Gaussian kernel, 1 disables)

    # Diagnostics
    shift_threshold: float = 12.0  # median grey-level drift that triggers a warning

    # Output
    output_dir: str = "output"
    record_fps: float = 30.0
    show_hud: bool = True
    branding: bool = True

    # ---- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        data = {key: getattr(self, key) for key in _PERSISTED}
        data["hsv"] = asdict(self.hsv)
        return data

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("Settings saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Settings":
        settings = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in _PERSISTED:
            if key in data:
                setattr(settings, key, data[key])
        hsv_fields = {f.name for f in fields(HSVRange)}
        if isinstance(data.get("hsv"), dict):
            settings.hsv = HSVRange(**{k: v for k, v in data["hsv"].items() if k in hsv_fields})
        elif settings.preset in PRESET_ORDER:
            settings.hsv = get_preset(settings.preset)
        log.info("Settings loaded from %s", path)
        return settings


def circular_mean_hue(hues: np.ndarray) -> int:
    """Mean of OpenCV hue values on the colour wheel (handles red's wrap-around)."""
    angles = hues.astype(np.float64) * (2.0 * math.pi / 180.0)
    mean = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    return int(round(mean * 180.0 / (2.0 * math.pi))) % 180


# ==================================================================
#  VIDEO INPUT (WEBCAM OR VIDEO FILE)
# ==================================================================

Source = Union[int, str]


class CameraError(RuntimeError):
    """Raised when a video source cannot be opened or stops delivering frames."""


class VideoSource:
    """A webcam index or a video file path, optionally read on a worker thread."""

    def __init__(self, source: Source, width: int = 1280, height: int = 720, threaded: bool = True):
        self.source = source
        self.is_file = isinstance(source, str)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            kind = "video file" if self.is_file else "webcam"
            raise CameraError(
                f"Unable to open {kind} '{source}'. "
                "Check the path / camera index, other apps using the camera, and OS camera permissions."
            )

        if not self.is_file:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.threaded = bool(threaded and not self.is_file)

        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_seq = 0
        self._alive = True
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.threaded:
            self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
            self._thread.start()
        log.info("Opened %s (%dx%d, threaded=%s)", source, *self.size, self.threaded)

    # ------------------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return not self.is_file

    @property
    def size(self) -> Tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    # ------------------------------------------------------------------
    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._alive = False
                    self._cond.notify_all()
                    return
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Return ``(ok, frame)``.  Threaded mode yields only *new* frames."""
        if not self.threaded:
            return self._cap.read()

        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._last_seq or not self._alive, timeout)
            if self._seq == self._last_seq:
                return False, None
            self._last_seq = self._seq
            return True, self._frame

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ==================================================================
#  COLOUR SEGMENTATION
# ==================================================================

@dataclass
class SegmentationResult:
    mask: np.ndarray  # uint8, 0 or 255 - hard mask
    alpha: np.ndarray  # float32, 0..1 - feathered mask used for blending
    contours: List[np.ndarray]  # outlines of the surviving regions
    coverage: float  # fraction of the frame covered by the cloak


@lru_cache(maxsize=32)
def _ellipse(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


class ColorSegmenter:
    """Detects the cloak in a BGR frame.

    Pipeline: Gaussian denoise -> BGR->HSV -> ``inRange`` (OR-ed for wrapping
    hues) -> morphological close / open / dilate -> drop blobs below
    ``min_area`` -> optional edge feathering for natural-looking blends.

    The tuning parameters are read from ``params`` on every call, so changes
    made at runtime (sliders, key presses) take effect immediately.
    """

    def __init__(self, params: Any):
        self.params = params

    def segment(self, frame: np.ndarray, hsv_range: HSVRange) -> SegmentationResult:
        p = self.params

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = None
        for lower, upper in hsv_range.cv_bounds():
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(p.close_kernel))  # fill pin-holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(p.open_kernel))  # drop speckle
        mask = cv2.dilate(mask, _ellipse(p.dilate_kernel))  # cover the cloak's fringe

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kept = [c for c in contours if cv2.contourArea(c) >= p.min_area]
        clean = np.zeros_like(mask)
        if kept:
            cv2.drawContours(clean, kept, -1, 255, thickness=cv2.FILLED)

        if p.feather and p.feather > 1:
            k = int(p.feather) | 1  # force odd
            alpha = cv2.GaussianBlur(clean, (k, k), 0).astype(np.float32) / 255.0
        else:
            alpha = clean.astype(np.float32) / 255.0

        coverage = cv2.countNonZero(clean) / float(clean.size)
        return SegmentationResult(mask=clean, alpha=alpha, contours=kept, coverage=coverage)


def sample_hsv(frame: np.ndarray, x: int, y: int, radius: int = 5) -> Tuple[int, int, int]:
    """Average HSV colour of the patch around ``(x, y)`` (hue averaged circularly)."""
    h, w = frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    patch = frame[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return (
        circular_mean_hue(hsv[:, 0]),
        int(np.median(hsv[:, 1])),
        int(np.median(hsv[:, 2])),
    )


# ==================================================================
#  BACKGROUND MODEL
# ==================================================================

_SMALL = (160, 90)


class BackgroundModel:
    """Holds the 'empty scene' image that replaces the cloak."""

    def __init__(self) -> None:
        self._image: Optional[np.ndarray] = None
        self._small_gray: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self._image is not None

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    def clear(self) -> None:
        self._image = None
        self._small_gray = None

    def build_from_frames(self, frames: Sequence[np.ndarray]) -> None:
        """Per-pixel median of several frames: removes sensor noise and flicker."""
        if not frames:
            raise ValueError("Cannot build a background from zero frames.")
        self.set_image(np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8))
        log.info("Background built from %d frames", len(frames))

    def set_image(self, image: np.ndarray) -> None:
        self._image = image
        self._small_gray = None

    def ensure_size(self, width: int, height: int) -> None:
        """Resize the stored background if the camera resolution changed."""
        if self._image is not None and self._image.shape[:2] != (height, width):
            self.set_image(cv2.resize(self._image, (width, height), interpolation=cv2.INTER_AREA))

    # ------------------------------------------------------------------
    def drift_score(self, frame: np.ndarray, cloak_mask: np.ndarray) -> float:
        """Median grey-level difference between the live frame and the background.

        Only pixels *outside* the cloak are compared.  The median (rather than
        the mean) keeps the score robust against the person standing in the
        scene.  A high value means the camera moved or the lighting changed.
        """
        if self._image is None:
            return 0.0
        if self._small_gray is None:
            self._small_gray = cv2.cvtColor(cv2.resize(self._image, _SMALL), cv2.COLOR_BGR2GRAY)

        live = cv2.cvtColor(cv2.resize(frame, _SMALL), cv2.COLOR_BGR2GRAY)
        outside = cv2.resize(cloak_mask, _SMALL, interpolation=cv2.INTER_NEAREST) == 0
        if int(outside.sum()) < 100:
            return 0.0
        diff = cv2.absdiff(live, self._small_gray)
        return float(np.median(diff[outside]))


# ==================================================================
#  COMPOSITING EFFECTS
# ==================================================================

class Effect(str, Enum):
    INVISIBLE = "invisible"
    GHOST = "ghost"
    PORTAL = "portal"
    NEON = "neon"

    @property
    def label(self) -> str:
        return self.value.title()

    def next(self) -> "Effect":
        members = list(Effect)
        return members[(members.index(self) + 1) % len(members)]


def blend(under: np.ndarray, over: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``alpha * under + (1 - alpha) * over`` using OpenCV's fast linear blend."""
    return cv2.blendLinear(under, over, alpha, 1.0 - alpha)


class EffectEngine:
    """Applies the selected :class:`Effect` to a frame."""

    _GRID = (72, 128)  # resolution of the procedural 'portal' texture

    def __init__(self) -> None:
        gh, gw = self._GRID
        self._ys, self._xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
        self._radius = np.hypot(self._xs - gw / 2.0, self._ys - gh / 2.0)
        self._portal_image: Optional[np.ndarray] = None
        self._portal_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def set_portal_image(self, image: Optional[np.ndarray]) -> None:
        """Use a custom picture inside the 'portal' effect instead of the plasma."""
        self._portal_image = image
        self._portal_cache = None

    def apply(
        self,
        effect: Effect,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        seg: SegmentationResult,
        tick: float = 0.0,
        ghost_strength: float = 0.65,
    ) -> np.ndarray:
        if background is None:
            return frame

        if effect is Effect.INVISIBLE:
            return blend(background, frame, seg.alpha)

        if effect is Effect.GHOST:
            return blend(background, frame, seg.alpha * float(np.clip(ghost_strength, 0.0, 1.0)))

        if effect is Effect.PORTAL:
            h, w = frame.shape[:2]
            return blend(self._portal_frame(h, w, tick), frame, seg.alpha)

        if effect is Effect.NEON:
            return self._neon(frame, background, seg, tick)

        raise ValueError(f"Unsupported effect: {effect!r}")

    # ------------------------------------------------------------------
    def _neon(self, frame: np.ndarray, background: np.ndarray, seg: SegmentationResult, tick: float) -> np.ndarray:
        out = blend(background, frame, seg.alpha)
        if not seg.contours:
            return out
        glow = np.zeros_like(out)
        cv2.drawContours(glow, seg.contours, -1, (255, 220, 0), thickness=4, lineType=cv2.LINE_AA)
        # Blur at quarter resolution: visually identical glow, ~8x cheaper.
        h, w = glow.shape[:2]
        small = cv2.resize(glow, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        pulse = 1.4 + 0.6 * math.sin(tick * 4.0)
        out = cv2.add(out, cv2.convertScaleAbs(glow, alpha=pulse))
        cv2.drawContours(out, seg.contours, -1, (255, 255, 160), thickness=2, lineType=cv2.LINE_AA)
        return out

    def _portal_frame(self, h: int, w: int, tick: float) -> np.ndarray:
        if self._portal_image is not None:
            if self._portal_cache is None or self._portal_cache.shape[:2] != (h, w):
                self._portal_cache = cv2.resize(self._portal_image, (w, h), interpolation=cv2.INTER_AREA)
            return self._portal_cache

        xs, ys = self._xs, self._ys
        field = (
            np.sin(xs * 0.11 + tick)
            + np.sin(ys * 0.15 + tick * 1.3)
            + np.sin((xs + ys) * 0.07 + tick * 0.7)
            + np.sin(self._radius * 0.20 - tick * 1.5)
        )  # range -4 .. 4
        hue = (((field + 4.0) / 8.0) * 179.0 + tick * 12.0) % 180.0
        hsv = np.dstack(
            (
                hue.astype(np.uint8),
                np.full(hue.shape, 220, np.uint8),
                np.full(hue.shape, 255, np.uint8),
            )
        )
        small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ==================================================================
#  ON-SCREEN GRAPHICS (HUD, TOASTS, PANELS)
# ==================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX
CYAN = (255, 230, 0)  # BGR
WHITE = (240, 240, 240)
AMBER = (0, 180, 255)
RED = (60, 60, 255)
GREEN = (120, 255, 120)

HELP_LINES: List[Tuple[str, str]] = [
    ("Click", "Sample cloak colour"),
    ("1-5", "Colour preset"),
    ("C", "Next colour preset"),
    ("E", "Next effect"),
    ("B", "Recapture background"),
    ("M", "Toggle mask view"),
    ("T", "HSV sliders window"),
    ("R", "Start / stop recording"),
    ("S", "Save screenshot"),
    ("P", "Save pipeline panel"),
    ("K", "Save settings"),
    ("SPACE", "Pause / resume"),
    ("I", "Toggle HUD"),
    ("F", "Fullscreen"),
    ("H", "Toggle this help"),
    ("Q / ESC", "Quit"),
]


def _text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def put_text(img, text, org, scale=0.55, color=WHITE, thickness=1, shadow=True) -> None:
    """Anti-aliased text with a soft drop shadow so it stays readable anywhere."""
    if shadow:
        cv2.putText(img, text, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def shade(img, x0, y0, x1, y1, color=(18, 14, 10), alpha=0.6) -> None:
    """Draw a translucent rectangle."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0)


class Hud:
    """Heads-up display drawn on top of the output frame."""

    def __init__(self, title: str = "INVISIBLE CLOAK PRO", author: str = "Sumeet Kumar Monti", branding: bool = True):
        self.title = title
        self.author = author
        self.branding = branding
        self._toasts: List[Tuple[str, float]] = []

    def toast(self, message: str, seconds: float = 2.5) -> None:
        self._toasts.append((message, time.monotonic() + seconds))
        self._toasts = self._toasts[-3:]

    # ------------------------------------------------------------------
    def draw(self, img: np.ndarray, info: Dict) -> None:
        h, w = img.shape[:2]
        s = max(0.55, min(1.5, w / 1280.0))
        now = time.monotonic()

        # --- top bar ------------------------------------------------------
        bar_h = int(64 * s)
        shade(img, 0, 0, w, bar_h, alpha=0.68)
        cv2.line(img, (0, bar_h), (w, bar_h), CYAN, max(1, int(2 * s)), cv2.LINE_AA)
        put_text(img, self.title, (int(16 * s), int(28 * s)), 0.78 * s, CYAN, 2)

        fps_text = f"{info.get('fps', 0.0):4.1f} FPS"
        tw, _ = _text_size(fps_text, 0.62 * s, 2)
        put_text(img, fps_text, (w - tw - int(16 * s), int(28 * s)), 0.62 * s, GREEN, 2)

        parts = [str(info.get("state", "")), f"Effect: {info.get('effect', '')}", f"Colour: {info.get('preset', '')}"]
        if info.get("coverage") is not None:
            parts.append(f"Cloak: {info['coverage']:.0%}")
        if info.get("view"):
            parts.append(f"View: {info['view']}")
        if info.get("paused"):
            parts.append("PAUSED")
        put_text(img, "   |   ".join(parts), (int(16 * s), int(53 * s)), 0.5 * s, WHITE, 1)

        # --- recording badge (in the top bar, left of the FPS counter) ------
        if info.get("recording"):
            rx = w - tw - int(100 * s)
            if int(now * 2) % 2 == 0:
                cv2.circle(img, (rx, int(21 * s)), int(7 * s), RED, -1, cv2.LINE_AA)
            put_text(img, "REC", (rx + int(14 * s), int(28 * s)), 0.6 * s, RED, 2)

        # --- warning banner ----------------------------------------------
        if info.get("warning"):
            y0 = bar_h + int(8 * s)
            shade(img, 0, y0, w, y0 + int(34 * s), color=(0, 90, 160), alpha=0.75)
            put_text(img, "! " + info["warning"], (int(16 * s), y0 + int(23 * s)), 0.55 * s, (255, 255, 255), 1)

        # --- centre message ----------------------------------------------
        if info.get("center"):
            text = info["center"]
            scale = 1.1 * s
            tw, th = _text_size(text, scale, 2)
            org = ((w - tw) // 2, (h + th) // 2)
            shade(img, org[0] - 24, org[1] - th - 22, org[0] + tw + 24, org[1] + 24, alpha=0.6)
            put_text(img, text, org, scale, CYAN, 2)

        # --- toasts --------------------------------------------------------
        self._toasts = [(m, t) for m, t in self._toasts if t > now]
        y = h - int(18 * s)
        for message, _ in reversed(self._toasts):
            tw, th = _text_size(message, 0.55 * s, 1)
            shade(img, int(10 * s), y - th - int(10 * s), int(28 * s) + tw, y + int(8 * s), alpha=0.7)
            put_text(img, message, (int(19 * s), y), 0.55 * s, WHITE, 1)
            y -= th + int(22 * s)

        # --- help panel ----------------------------------------------------
        if info.get("show_help"):
            line_h = int(24 * s)
            pw, ph = int(300 * s), line_h * (len(HELP_LINES) + 2)
            x0, y0 = w - pw - int(12 * s), bar_h + int(50 * s)
            shade(img, x0, y0, x0 + pw, y0 + ph, alpha=0.78)
            put_text(img, "CONTROLS", (x0 + int(14 * s), y0 + int(22 * s)), 0.6 * s, CYAN, 2)
            for i, (key, desc) in enumerate(HELP_LINES, start=1):
                yy = y0 + int(22 * s) + i * line_h
                put_text(img, key, (x0 + int(14 * s), yy), 0.5 * s, AMBER, 1)
                put_text(img, desc, (x0 + int(100 * s), yy), 0.5 * s, WHITE, 1)

        # --- branding ------------------------------------------------------
        if self.branding and not info.get("show_help"):
            tag = f"{self.author}  |  github.com/SUMEETKUMARMONTI"
            tw, _ = _text_size(tag, 0.42 * s, 1)
            put_text(img, tag, (w - tw - int(12 * s), h - int(12 * s)), 0.42 * s, (200, 200, 200), 1)


def pipeline_panel(original: np.ndarray, mask: Optional[np.ndarray], result: np.ndarray, height: int = 360) -> np.ndarray:
    """Side-by-side 'original | mask | result' image - ideal for READMEs and posts."""

    def prep(img: np.ndarray, label: str) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        shade(img, 0, 0, img.shape[1], 34, alpha=0.7)
        put_text(img, label, (12, 24), 0.6, CYAN, 2)
        return img

    if mask is None:
        mask = np.zeros(original.shape[:2], np.uint8)
    panels = [prep(original, "1  ORIGINAL"), prep(mask, "2  HSV MASK"), prep(result, "3  RESULT")]
    gap = np.full((height, 10, 3), (30, 24, 18), np.uint8)
    row: List[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            row.append(gap)
        row.append(p)
    return np.hstack(row)


# ==================================================================
#  VIDEO RECORDER
# ==================================================================

class VideoRecorder:
    """Thin wrapper around ``cv2.VideoWriter`` with sane defaults and fallbacks."""

    def __init__(self, output_dir: str = "output", fps: float = 30.0):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Tuple[int, int] = (0, 0)
        self.path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, size: Tuple[int, int], path: Optional[Path] = None) -> Path:
        """Begin recording.  ``size`` is ``(width, height)``."""
        if self.active:
            self.stop()
        if path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"invisible_cloak_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        for fourcc, suffix in (("mp4v", path.suffix or ".mp4"), ("MJPG", ".avi")):
            candidate = path.with_suffix(suffix)
            writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*fourcc), self.fps, size)
            if writer.isOpened():
                self._writer, self._size, self.path = writer, size, candidate
                log.info("Recording to %s (%s)", candidate, fourcc)
                return candidate
            writer.release()
        raise RuntimeError("Could not open a video writer (no usable codec found).")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        self._writer.write(frame)

    def stop(self) -> Optional[Path]:
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        log.info("Recording saved: %s", self.path)
        return self.path


# ==================================================================
#  APPLICATION (STATE MACHINE, KEYBOARD, MOUSE, SLIDERS)
# ==================================================================

WINDOW = "Invisible Cloak Pro"
CONTROLS = "HSV Controls"
_TRACKBARS = (
    ("Lower Hue", "h_lo", 179),
    ("Lower Sat", "s_lo", 255),
    ("Lower Val", "v_lo", 255),
    ("Upper Hue", "h_hi", 179),
    ("Upper Sat", "s_hi", 255),
    ("Upper Val", "v_hi", 255),
)
_STATE_TEXT = {"COUNTDOWN": "Get ready", "CAPTURING": "Capturing", "RUNNING": "Live"}


class State(Enum):
    COUNTDOWN = 1  # waiting for the person to leave the frame
    CAPTURING = 2  # collecting background frames
    RUNNING = 3  # cloak effect active


class CloakApp:
    """Ties camera, segmentation, effects and UI together."""

    def __init__(
        self,
        settings: Settings,
        *,
        headless: bool = False,
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_panel: Optional[str] = None,
        portal_image: Optional[str] = None,
        background_image: Optional[str] = None,
        settings_path: str = "settings.json",
    ):
        self.settings = settings
        self.headless = headless
        self.output_video = output_video
        self.max_frames = max_frames
        self.save_panel = save_panel
        self.settings_path = settings_path

        source = settings.source if settings.source else settings.camera_index
        self.source = VideoSource(source, settings.width, settings.height, settings.threaded)

        self.segmenter = ColorSegmenter(settings)
        self.background = BackgroundModel()
        self.effects = EffectEngine()
        self.recorder = VideoRecorder(settings.output_dir, settings.record_fps)
        self.hud = Hud(branding=settings.branding)
        try:
            self.effect = Effect(settings.effect)
        except ValueError:
            self.effect = Effect.INVISIBLE

        self._static_background: Optional[np.ndarray] = self._read_image(background_image)
        self.effects.set_portal_image(self._read_image(portal_image))

        # runtime state
        self.state = State.COUNTDOWN
        self.running = True
        self.paused = False
        self.show_mask = False
        self.show_help = False
        self.fullscreen = False
        self._trackbars_open = False
        self._syncing = False

        self._deadline = 0.0
        self._bg_frames: List[np.ndarray] = []
        self._last_frame: Optional[np.ndarray] = None
        self._clean: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._coverage: Optional[float] = None
        self._center_text: Optional[str] = None
        self._drift_warning = False
        self._read_failures = 0
        self._frames = 0
        self._rec_started = False
        self._best: Optional[tuple] = None  # (coverage, original, mask, result)

        self.fps = 0.0
        self._t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_image(path: Optional[str]) -> Optional[np.ndarray]:
        if not path:
            return None
        image = cv2.imread(path)
        if image is None:
            raise CameraError(f"Could not read image: {path}")
        return image

    def _out_path(self, prefix: str) -> Path:
        out = Path(self.settings.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.png"

    def _toast(self, message: str) -> None:
        log.info(message)
        self.hud.toast(message)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        if not self.headless:
            self._init_windows()
        self._begin_capture()
        if not self.headless:
            log.info("Running. Press H for help, Q to quit.")
        try:
            while self.running:
                self._step()
        finally:
            self._shutdown()
        return 0

    def _step(self) -> None:
        t0 = time.perf_counter()

        if not self.paused or self._last_frame is None:
            ok, frame = self.source.read()
            if not ok or frame is None:
                if self.source.is_file:
                    log.info("End of video reached.")
                    self.running = False
                    return
                self._read_failures += 1
                if self._read_failures > 30:
                    raise CameraError("Camera stopped delivering frames.")
                return
            self._read_failures = 0

            if self.settings.mirror:
                frame = cv2.flip(frame, 1)
            self._last_frame = frame
            self._clean, self._mask = self._process(frame)
            self._frames += 1

            if self.recorder.active:
                self.recorder.write(self._clean)

        if not self.headless:
            view = self._compose_display()
            if view is not None:
                cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._on_key(key)
            if self._frames > 5 and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                self.running = False

        if self.max_frames is not None and self._frames >= self.max_frames:
            self.running = False

        dt = time.perf_counter() - t0
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst

    # ------------------------------------------------------------------
    # frame processing
    # ------------------------------------------------------------------
    def _begin_capture(self) -> None:
        """(Re)start background acquisition."""
        self._bg_frames = []
        self._drift_warning = False
        if self._static_background is not None:
            self.background.set_image(self._static_background)
            self.state = State.RUNNING
            self._toast("Using static background image")
            return
        self.background.clear()
        seconds = self.settings.countdown_seconds if self.source.is_live else 0
        self._deadline = time.monotonic() + seconds
        self.state = State.COUNTDOWN

    def _process(self, frame: np.ndarray):
        self._center_text = None

        if self.state is State.COUNTDOWN:
            remaining = math.ceil(self._deadline - time.monotonic())
            if remaining <= 0:
                self.state = State.CAPTURING
            else:
                self._center_text = f"Step out of the frame...  {remaining}"
                return frame, None

        if self.state is State.CAPTURING:
            self._bg_frames.append(frame)
            need = max(1, self.settings.background_frames)
            self._center_text = f"Capturing background  {len(self._bg_frames)}/{need}"
            if len(self._bg_frames) >= need:
                self.background.build_from_frames(self._bg_frames)
                self._bg_frames = []
                self.state = State.RUNNING
                self._center_text = None
                self._toast("Background captured - put the cloak on!")
            return frame, None

        # ---- RUNNING ------------------------------------------------------
        h, w = frame.shape[:2]
        self.background.ensure_size(w, h)
        seg = self.segmenter.segment(frame, self.settings.hsv)
        tick = time.perf_counter() - self._t_start
        out = self.effects.apply(
            self.effect, frame, self.background.image, seg, tick, self.settings.ghost_strength
        )
        self._coverage = seg.coverage

        if self._frames % 10 == 0:
            drift = self.background.drift_score(frame, seg.mask)
            self._drift_warning = drift > self.settings.shift_threshold

        if self._best is None or seg.coverage > self._best[0]:
            self._best = (seg.coverage, frame, seg.mask, out)

        if self.output_video and not self._rec_started:
            self.recorder.start((w, h), Path(self.output_video))
            self._rec_started = True
        return out, seg.mask

    def _compose_display(self) -> Optional[np.ndarray]:
        if self._clean is None:
            return None
        if self.show_mask and self._mask is not None:
            view = cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR)
        else:
            view = self._clean.copy()
        if self.settings.show_hud:
            self.hud.draw(view, self._hud_info())
        return view

    def _hud_info(self) -> dict:
        return {
            "fps": self.fps,
            "state": _STATE_TEXT[self.state.name],
            "effect": self.effect.label,
            "preset": self.settings.preset,
            "coverage": self._coverage if self.state is State.RUNNING else None,
            "recording": self.recorder.active,
            "paused": self.paused,
            "view": "Mask" if self.show_mask else None,
            "show_help": self.show_help,
            "center": self._center_text,
            "warning": "Camera moved or lighting changed - press B to recapture"
            if self._drift_warning
            else None,
        }

    # ------------------------------------------------------------------
    # windows, sliders, mouse
    # ------------------------------------------------------------------
    def _init_windows(self) -> None:
        w, h = self.source.size
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(w or 1280, 1280), min(h or 720, 720))
        cv2.setMouseCallback(WINDOW, self._on_mouse)

    def _open_trackbars(self) -> None:
        cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CONTROLS, 480, 280)
        for label, field_name, maximum in _TRACKBARS:
            cv2.createTrackbar(
                label, CONTROLS, getattr(self.settings.hsv, field_name), maximum,
                partial(self._on_trackbar, field_name),
            )
        self._trackbars_open = True

    def _close_trackbars(self) -> None:
        cv2.destroyWindow(CONTROLS)
        self._trackbars_open = False

    def _sync_trackbars(self) -> None:
        if not self._trackbars_open:
            return
        self._syncing = True
        try:
            for label, field_name, _ in _TRACKBARS:
                cv2.setTrackbarPos(label, CONTROLS, getattr(self.settings.hsv, field_name))
        finally:
            self._syncing = False

    def _on_trackbar(self, field_name: str, value: int) -> None:
        if self._syncing:
            return
        setattr(self.settings.hsv, field_name, value)
        self.settings.preset = "custom"

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._last_frame is None:
            return
        h, s, v = sample_hsv(self._last_frame, x, y)
        self.settings.hsv = HSVRange.from_sample(h, s, v)
        self.settings.preset = "custom"
        self._sync_trackbars()
        self._toast(f"Sampled colour  H={h}  S={s}  V={v}")

    # ------------------------------------------------------------------
    # keyboard
    # ------------------------------------------------------------------
    def _on_key(self, key: int) -> None:
        if key == 27:
            self.running = False
            return
        if not 32 <= key < 127:
            return
        ch = chr(key).lower()

        if ch == "q":
            self.running = False
        elif ch == "b":
            self._begin_capture()
        elif ch == "s":
            self._save_screenshot()
        elif ch == "p":
            self._save_panel()
        elif ch == "r":
            self._toggle_recording()
        elif ch == "m":
            self.show_mask = not self.show_mask
        elif ch == "e":
            self.effect = self.effect.next()
            self.settings.effect = self.effect.value
            self._toast(f"Effect: {self.effect.label}")
        elif ch == "c":
            names = list(PRESET_ORDER)
            idx = names.index(self.settings.preset) + 1 if self.settings.preset in names else 0
            self._apply_preset(names[idx % len(names)])
        elif ch.isdigit() and 1 <= int(ch) <= len(PRESET_ORDER):
            self._apply_preset(PRESET_ORDER[int(ch) - 1])
        elif ch == " ":
            self.paused = not self.paused
        elif ch == "h":
            self.show_help = not self.show_help
        elif ch == "i":
            self.settings.show_hud = not self.settings.show_hud
        elif ch == "t":
            if self._trackbars_open:
                self._close_trackbars()
            else:
                self._open_trackbars()
        elif ch == "k":
            self.settings.save(self.settings_path)
            self._toast(f"Settings saved to {self.settings_path}")
        elif ch == "f":
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )

    def _apply_preset(self, name: str) -> None:
        self.settings.preset = name
        self.settings.hsv = get_preset(name)
        self._sync_trackbars()
        self._toast(f"Colour preset: {name}")

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _save_screenshot(self) -> None:
        if self._clean is None:
            return
        path = self._out_path("invisible_cloak")
        cv2.imwrite(str(path), self._clean)
        self._toast(f"Screenshot saved: {path.name}")

    def _save_panel(self) -> None:
        if self._clean is None or self._last_frame is None:
            return
        path = self._out_path("pipeline")
        cv2.imwrite(str(path), pipeline_panel(self._last_frame, self._mask, self._clean))
        self._toast(f"Pipeline panel saved: {path.name}")

    def _toggle_recording(self) -> None:
        if self.recorder.active:
            path = self.recorder.stop()
            self._toast(f"Recording saved: {path.name if path else ''}")
        elif self._clean is not None:
            h, w = self._clean.shape[:2]
            path = self.recorder.start((w, h))
            self._toast(f"Recording to {path.name}")

    def _shutdown(self) -> None:
        self.recorder.stop()
        if self.save_panel and self._best is not None:
            _, original, mask, result = self._best
            path = Path(self.save_panel)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), pipeline_panel(original, mask, result))
            log.info("Pipeline panel written to %s", path)
        self.source.release()
        if not self.headless:
            cv2.destroyAllWindows()
        log.info("Processed %d frames. Goodbye!", self._frames)


# ==================================================================
#  COMMAND LINE INTERFACE
# ==================================================================

EPILOG = """\
examples:
  python main.py                              # webcam, green cloak
  python main.py --preset red --effect neon   # red cloak with a neon outline
  python main.py --background-image room.jpg  # no need to leave the frame
  python main.py --video clip.mp4 --no-mirror --headless -o out.mp4 --save-panel panel.png
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="invisible-cloak",
        description="Real-time invisibility cloak using HSV colour segmentation (OpenCV).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_argument_group("input")
    src.add_argument("--camera", type=int, default=None, help="webcam index (default: 0)")
    src.add_argument("--video", metavar="FILE", help="process a video file instead of a webcam")
    src.add_argument("--width", type=int, default=None, help="requested capture width")
    src.add_argument("--height", type=int, default=None, help="requested capture height")
    src.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    src.add_argument("--no-thread", action="store_true", help="disable the threaded camera reader")

    look = p.add_argument_group("appearance")
    look.add_argument("--preset", choices=PRESET_ORDER, help="cloak colour preset")
    look.add_argument("--effect", choices=[e.value for e in Effect], help="compositing effect")
    look.add_argument("--background-image", metavar="IMG", help="use this image as the background (skips capture)")
    look.add_argument("--portal-image", metavar="IMG", help="picture shown by the 'portal' effect")
    look.add_argument("--no-hud", action="store_true", help="hide the on-screen display")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--countdown", type=int, default=None, help="seconds before background capture")
    tune.add_argument("--bg-frames", type=int, default=None, help="frames used to build the background")
    tune.add_argument("--min-area", type=int, default=None, help="ignore cloak blobs smaller than N pixels")
    tune.add_argument("--feather", type=int, default=None, help="edge softness (odd number, 1 = hard edge)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", metavar="FILE", help="record the result to this video file")
    out.add_argument("--output-dir", default=None, help="folder for screenshots and recordings")
    out.add_argument("--save-panel", metavar="FILE", help="on exit, save an original|mask|result panel")
    out.add_argument("--headless", action="store_true", help="no windows; process and exit (great with --video)")
    out.add_argument("--max-frames", type=int, default=None, help="stop after N frames")

    misc = p.add_argument_group("misc")
    misc.add_argument("--config", default="settings.json", help="settings file to load / save (default: settings.json)")
    misc.add_argument("--list-presets", action="store_true", help="print colour presets and exit")
    misc.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    misc.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> None:
    if args.camera is not None:
        settings.camera_index = args.camera
    if args.video:
        settings.source = args.video
    if args.width:
        settings.width = args.width
    if args.height:
        settings.height = args.height
    if args.no_mirror:
        settings.mirror = False
    if args.no_thread:
        settings.threaded = False
    if args.preset:
        settings.preset = args.preset
        settings.hsv = get_preset(args.preset)
    if args.effect:
        settings.effect = args.effect
    if args.no_hud:
        settings.show_hud = False
    if args.countdown is not None:
        settings.countdown_seconds = max(0, args.countdown)
    if args.bg_frames is not None:
        settings.background_frames = max(1, args.bg_frames)
    if args.min_area is not None:
        settings.min_area = max(0, args.min_area)
    if args.feather is not None:
        settings.feather = max(1, args.feather)
    if args.output_dir:
        settings.output_dir = args.output_dir


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_presets:
        for name in PRESET_ORDER:
            r = get_preset(name)
            note = "  (hue wraps around 0)" if r.wraps else ""
            print(f"{name:8s} H {r.h_lo:3d}-{r.h_hi:3d}  S {r.s_lo:3d}-{r.s_hi:3d}  V {r.v_lo:3d}-{r.v_hi:3d}{note}")
        return 0

    config_path = Path(args.config)
    try:
        settings = Settings.load(config_path) if config_path.exists() else Settings()
    except (ValueError, KeyError, OSError) as exc:
        log.warning("Ignoring unreadable settings file %s (%s)", config_path, exc)
        settings = Settings()
    _apply_overrides(settings, args)

    try:
        app = CloakApp(
            settings,
            headless=args.headless,
            output_video=args.output,
            max_frames=args.max_frames,
            save_panel=args.save_panel,
            portal_image=args.portal_image,
            background_image=args.background_image,
            settings_path=str(config_path),
        )
        return app.run()
    except CameraError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())