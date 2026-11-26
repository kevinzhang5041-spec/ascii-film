#!/usr/bin/env python3
"""
ascii_film.py - render real footage (video or photos) as ASCII-art video, keeping the original audio.

  python ascii_film.py clip.mov                       # -> clip_ascii.mp4
  python ascii_film.py clip.mov out.mp4 --cols "0:20,180:160" --color "0:mono,180:color"
  python ascii_film.py clip.mov --preview 12.5        # one test frame at 12.5 s -> PNG

Every style option takes either one value or "time:value" keyframes separated by commas.
Times are seconds ("90:") or a percentage of the clip ("50%:").
Numbers and colours interpolate between keyframes; --charset and --invert switch at them.

  --cols      characters across the frame                 "120"   or "0:20,180:160"
  --color     mono -> color blend (mono | color | 0..1)    "mono"  or "0%:mono,100%:color"
  --charset   named set (see CHARSETS) or literal string   "standard" or "0:blocks,60:detailed"
  --tint      glyph colour in mono                         "#ffffff"
  --bg        background colour                            "#000000"
  --contrast  luminance contrast around mid-grey           "1.0"
  --invert    dense glyphs for dark areas (for light bg)   flag, or "0:false,90:true"
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".3gp"}
IMAGE_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

# Ordered sparse -> dense.
CHARSETS = {
    "standard": " .:-=+*#%@",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "blocks": " ░▒▓█",
    "dots": " ·•●",
    "minimal": " .:",
    "binary": " 01",
    "lines": " .-=≡",
}
NAMED_COLORS = {
    "white": "#ffffff", "black": "#000000", "green": "#33ff66", "amber": "#ffb000",
    "paper": "#f2ede3", "ink": "#1b1b1b", "moss": "#9cc47a", "night": "#0b0f0c",
}
DEFAULT_STYLE = {
    "cols": "120", "color": "mono", "charset": "standard", "tint": "#ffffff",
    "bg": "#000000", "contrast": "1.0", "invert": "false",
}
STYLE_KINDS = {
    "cols": "int", "color": "mix", "charset": "charset", "tint": "rgb",
    "bg": "rgb", "contrast": "num", "invert": "bool",
}
CHAR_ASPECT = 2.0          # cell height / cell width
LUMA = np.array([0.299, 0.587, 0.114], np.float32) / 255.0
FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/Library/Fonts/Courier New.ttf",
    "DejaVuSansMono.ttf",
]


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- schedules

_PAIR = re.compile(r"^\s*(\d+(?:\.\d+)?)(%?)\s*:(.*)$", re.S)
_SPLIT = re.compile(r",(?=\s*\d+(?:\.\d+)?%?\s*:)")


def _spec_str(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_color(s: str) -> tuple:
    s = s.strip().lower()
    s = NAMED_COLORS.get(s, s)
    if re.fullmatch(r"#?[0-9a-f]{3}", s):
        s = "#" + "".join(c * 2 for c in s.lstrip("#"))
    if not re.fullmatch(r"#?[0-9a-f]{6}", s):
        raise ValueError(f"not a colour: {s!r} (use #rrggbb or one of {', '.join(NAMED_COLORS)})")
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _parse_value(raw: str, kind: str, name: str):
    s = raw.strip()
    try:
        if kind in ("int", "num"):
            v = float(s)
            if kind == "int" and v < 2:
                raise ValueError("must be at least 2")
            return v
        if kind == "mix":
            low = s.lower()
            if low in ("mono", "bw", "grey", "gray", "off", "false"):
                return 0.0
            if low in ("color", "colour", "full", "on", "true"):
                return 1.0
            return min(1.0, max(0.0, float(s)))
        if kind == "rgb":
            return parse_color(s)
        if kind == "bool":
            low = s.lower()
            if low in ("1", "true", "yes", "on", "invert"):
                return True
            if low in ("0", "false", "no", "off", ""):
                return False
            raise ValueError("use true/false")
        if kind == "charset":
            if s.lower() in CHARSETS:
                return CHARSETS[s.lower()]
            lit = raw.rstrip("\n")
            if len(set(lit)) < 2:
                raise ValueError(f"needs at least 2 characters or a name: {', '.join(CHARSETS)}")
            return lit
    except ValueError as e:
        raise ValueError(f"--{name}: bad value {raw!r}: {e}") from None
    raise ValueError(kind)


class Schedule:
    """A constant or a list of (time, value) keyframes. Time may be seconds or % of the clip."""

    def __init__(self, spec, kind: str, name: str = ""):
        self.kind, self.name = kind, name
        spec = _spec_str(spec)
        keys = []
        if _PAIR.match(spec):
            for part in _SPLIT.split(spec):
                m = _PAIR.match(part)
                if not m:
                    raise ValueError(f"--{name}: can't read keyframe {part!r} (expected seconds:value)")
                keys.append((float(m.group(1)), m.group(2) == "%", _parse_value(m.group(3), kind, name)))
        else:
            keys.append((0.0, False, _parse_value(spec, kind, name)))
        self.keys = keys
        self._resolved = {}

    def values(self):
        return [k[2] for k in self.keys]

    def _at_duration(self, duration):
        r = self._resolved.get(duration)
        if r is None:
            r = sorted(((t / 100.0 * duration if pct else t), v) for t, pct, v in self.keys)
            self._resolved[duration] = r
        return r

    def at(self, t: float, duration: float):
        ks = self._at_duration(duration)
        if t <= ks[0][0] or len(ks) == 1:
            return ks[0][1]
        if t >= ks[-1][0]:
            return ks[-1][1]
        for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
            if t0 <= t < t1:
                if self.kind in ("charset", "bool") or t1 == t0:
                    return v0
                f = (t - t0) / (t1 - t0)
                if self.kind == "rgb":
                    return tuple(int(round(a + (b - a) * f)) for a, b in zip(v0, v1))
                return v0 + (v1 - v0) * f
        return ks[-1][1]


@dataclass(frozen=True)
class FrameStyle:
    cols: int
    mix: float
    chars: str
    tint: tuple
    bg: tuple
    contrast: float
    invert: bool


class Style:
    def __init__(self, spec: dict | None = None):
        merged = dict(DEFAULT_STYLE)
        for k, v in (spec or {}).items():
            if k in STYLE_KINDS and v is not None and _spec_str(v).strip() != "":
                merged[k] = _spec_str(v)
        self.spec = merged
        self.sched = {k: Schedule(merged[k], kind, k) for k, kind in STYLE_KINDS.items()}

    def at(self, t: float, duration: float) -> FrameStyle:
        s = self.sched
        return FrameStyle(
            cols=int(round(s["cols"].at(t, duration))),
            mix=round(float(s["color"].at(t, duration)), 3),
            chars=s["charset"].at(t, duration),
            tint=s["tint"].at(t, duration),
            bg=s["bg"].at(t, duration),
            contrast=round(float(s["contrast"].at(t, duration)), 3),
            invert=bool(s["invert"].at(t, duration)),
        )

    def max_cols(self) -> int:
        return int(round(max(self.sched["cols"].values())))

    def is_static(self) -> bool:
        return all(len(set(map(repr, sc.values()))) == 1 for sc in self.sched.values())


# --------------------------------------------------------------------------- glyph rendering

def find_font(path: str | None = None) -> str:
    for cand in ([path] if path else []) + FONT_CANDIDATES:
        try:
            ImageFont.truetype(cand, 12)
            return cand
        except OSError:
            continue
    raise SystemExit("No monospace font found; pass --font /path/to/font.ttf")


@lru_cache(maxsize=128)
def _font(path: str, size: int):
    return ImageFont.truetype(path, size)


_ATLAS: dict = {}


def glyph_atlas(chars: str, cw: int, ch: int, font_path: str) -> np.ndarray:
    """(len(chars), ch, cw) uint8 coverage masks, one per character."""
    key = (chars, cw, ch, font_path)
    a = _ATLAS.get(key)
    if a is not None:
        return a
    font = _font(font_path, max(4, int(min(cw / 0.6, ch / 1.18))))
    a = np.zeros((len(chars), ch, cw), np.uint8)
    block_box = None
    for i, c in enumerate(chars):
        if c == " ":
            continue
        if 0x2580 <= ord(c) <= 0x259F:
            # block/shade elements: stretch to fill the whole cell so rows and columns join up
            big = _font(font_path, 96)
            if block_box is None:
                block_box = big.getbbox("█", anchor="lt")
            im = Image.new("L", (block_box[2] - block_box[0], block_box[3] - block_box[1]), 0)
            ImageDraw.Draw(im).text((-block_box[0], -block_box[1]), c, fill=255, font=big, anchor="lt")
            a[i] = cv2.resize(np.asarray(im), (cw, ch), interpolation=cv2.INTER_AREA)
            continue
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((cw / 2, ch / 2), c, fill=255, font=font, anchor="mm")
        a[i] = np.asarray(im)
    if len(_ATLAS) > 400:
        _ATLAS.clear()
    _ATLAS[key] = a
    return a


def grid(W: int, H: int, cols: int):
    cols = max(2, min(int(cols), W // 3))
    cw = math.ceil(W / cols)
    rows = max(1, round(H / (cw * CHAR_ASPECT)))
    ch = math.ceil(H / rows)
    return cols, rows, cw, ch


def render_frame(src: np.ndarray, fs: FrameStyle, W: int, H: int, font_path: str) -> np.ndarray:
    """src: RGB uint8 frame (any size, ideally small). Returns an (H, W, 3) RGB uint8 ASCII frame."""
    cols, rows, _, _ = grid(W, H, fs.cols)
    cells = cv2.resize(src, (cols, rows), interpolation=cv2.INTER_AREA)
    return draw_glyphs(cell_glyphs(cells, fs), cells, fs, W, H, font_path)


def cell_glyphs(cells: np.ndarray, fs: FrameStyle) -> np.ndarray:
    """(rows, cols, 3) RGB uint8 cell colours -> (rows, cols) int32 index into fs.chars (sparse -> dense)."""
    lum = cells.astype(np.float32) @ LUMA
    if fs.contrast != 1.0:
        lum = (lum - 0.5) * fs.contrast + 0.5
    if fs.invert:
        lum = 1.0 - lum
    n = len(fs.chars)
    return np.clip((lum * n).astype(np.int32), 0, n - 1)


def draw_glyphs(idx: np.ndarray, cells: np.ndarray, fs: FrameStyle, W: int, H: int, font_path: str) -> np.ndarray:
    """Paint glyph indices (rows, cols) into an (H, W, 3) RGB uint8 frame, coloured by `cells` (rows, cols, 3).

    Split from render_frame so callers can edit the indices in between (e.g. swap random glyphs)."""
    cols, rows, cw, ch = grid(W, H, fs.cols)
    if idx.shape != (rows, cols) or cells.shape[:2] != (rows, cols):
        raise ValueError(f"cell grid {idx.shape} doesn't match {rows}x{cols} for {W}x{H} at {fs.cols} cols")
    atlas = glyph_atlas(fs.chars, cw, ch, font_path)
    mask = atlas[idx].transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)
    y0, x0 = (rows * ch - H) // 2, (cols * cw - W) // 2
    mask = mask[y0:y0 + H, x0:x0 + W]
    bg = np.array(fs.bg, np.float32)
    tint = np.array(fs.tint, np.float32)
    mask3 = cv2.merge([mask, mask, mask])
    ramp = np.arange(256, dtype=np.float32)[:, None] / 255.0

    if fs.mix <= 0.0:  # mono: a single 256-entry lookup per channel
        lut = np.clip(bg + ramp * (tint - bg) + 0.5, 0, 255).astype(np.uint8).reshape(1, 256, 3)
        return cv2.LUT(mask3, lut)

    # colour: out = fg * a + bg * (1 - a), with a = glyph coverage. fg is per-cell, so build it small
    # and blow it up with nearest-neighbour; the bg term is a lookup on the mask.
    lifted = np.power(cells.astype(np.float32) / 255.0, 0.7) * 255.0  # glyphs cover part of a cell: lift colour
    fg_cells = np.clip((1.0 - fs.mix) * tint + fs.mix * lifted + 0.5, 0, 255).astype(np.uint8)
    fg = cv2.resize(fg_cells, (cols * cw, rows * ch), interpolation=cv2.INTER_NEAREST)[y0:y0 + H, x0:x0 + W]
    out = cv2.multiply(fg, mask3, scale=1.0 / 255.0)
    if fs.bg != (0, 0, 0):
        bg_lut = np.clip(bg * (1.0 - ramp) + 0.5, 0, 255).astype(np.uint8).reshape(1, 256, 3)
        out = cv2.add(out, cv2.LUT(mask3, bg_lut))
    return out


# --------------------------------------------------------------------------- media I/O

def tool(name: str) -> str:
    p = shutil.which(name) or next((c for c in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}")
                                    if os.path.exists(c)), None)
    if not p:
        raise SystemExit(f"{name} not found. Run ./ascii setup (or: brew install ffmpeg)")
    return p


@lru_cache(maxsize=None)
def has_filter(name: str) -> bool:
    out = subprocess.run([tool("ffmpeg"), "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", out, re.M) is not None


@dataclass
class MediaInfo:
    path: str
    kind: str                 # "video" | "image"
    width: int                # display size (rotation applied)
    height: int
    duration: float
    fps: float
    hdr: bool = False
    audio_index: int | None = None
    image_path: str | None = None
    extra: dict = field(default_factory=dict)


def default_cache_dir() -> Path:
    return HERE / ".work"


def readable_image(path: Path, cache_dir: Path | None = None) -> Path:
    """HEIC/HEIF -> cached JPEG via macOS `sips` (input file is never touched)."""
    if path.suffix.lower() not in (".heic", ".heif"):
        return path
    cache = Path(cache_dir or default_cache_dir()) / "heic"
    cache.mkdir(parents=True, exist_ok=True)
    st = path.stat()
    out = cache / f"{path.stem}-{st.st_size}-{int(st.st_mtime)}.jpg"
    if not out.exists():
        try:
            import pillow_heif  # optional fallback, not required on macOS
            pillow_heif.register_heif_opener()
            with Image.open(path) as im:
                im.save(out, quality=95, exif=im.getexif())
        except ImportError:
            r = subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "95", str(path),
                                "--out", str(out)], capture_output=True, text=True)
            if r.returncode != 0 or not out.exists():
                raise RuntimeError(f"HEIC conversion failed for {path.name}: {r.stderr.strip() or r.stdout.strip()}")
    return out


def probe(path, cache_dir=None) -> MediaInfo:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        img = readable_image(p, cache_dir)
        with Image.open(img) as im:
            w, h = im.size
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                w, h = h, w
        return MediaInfo(str(p), "image", w, h, 0.0, 0.0, image_path=str(img))
    r = subprocess.run([tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams",
                        "-show_format", str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe couldn't read {p.name}: {r.stderr.strip()}")
    data = json.loads(r.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise RuntimeError(f"{p.name} has no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    dur = float(v.get("duration") or data.get("format", {}).get("duration") or 0)
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    aidx = next((i for i, s in enumerate(audio) if s.get("codec_name") not in (None, "", "none", "apac")), None)
    hdr = v.get("color_transfer") in ("smpte2084", "arib-std-b67")
    return MediaInfo(str(p), "video", w, h, dur, fps, hdr, aidx,
                     extra={"codec": v.get("codec_name"), "rotation": rot, "pix_fmt": v.get("pix_fmt"),
                            "start_time": float(v.get("start_time") or 0.0)})


def even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


def out_size(w: int, h: int, size: int):
    s = size / max(w, h)
    return even(w * s), even(h * s)


def analysis_size(w: int, h: int, max_cols: int):
    aw = int(min(w, max(96, max_cols * 4)))
    return aw, max(2, int(round(aw * h / w)))


@lru_cache(maxsize=None)
def hwdec_args() -> tuple:
    """Apple VideoToolbox hardware decoding: identical pixels, ~3.5x less CPU on iPhone HEVC.
    Disable with ASCII_FILM_HWDEC=0. ffmpeg falls back to software by itself if a stream isn't supported."""
    if sys.platform != "darwin" or os.environ.get("ASCII_FILM_HWDEC", "1") == "0":
        return ()
    out = subprocess.run([tool("ffmpeg"), "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    return ("-hwaccel", "videotoolbox") if "videotoolbox" in out else ()


@lru_cache(maxsize=None)
def scale_has_color_mgmt() -> bool:
    out = subprocess.run([tool("ffmpeg"), "-hide_banner", "-h", "filter=scale"], capture_output=True, text=True).stdout
    return "out_transfer" in out


def _vf(info: MediaInfo, aw: int, ah: int, fps: float | None) -> str:
    """Decode filter chain: constant fps, small analysis size, HDR (iPhone HLG / PQ) -> SDR BT.709."""
    parts = [f"fps={fps}"] if fps else []
    if info.hdr and scale_has_color_mgmt():
        # ffmpeg >= 8 swscale maps HDR -> SDR itself. Downscale first: converting at 4K is ~3x slower.
        parts.append(f"scale={aw}:{ah}:flags=area")
        parts.append("scale=out_transfer=bt709:out_primaries=bt709:out_color_matrix=bt709")
    elif info.hdr and has_filter("zscale") and has_filter("tonemap"):
        parts.append(f"scale={aw}:{ah}:flags=area")
        parts.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                     "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    else:
        parts.append(f"scale={aw}:{ah}:flags=area")
    parts.append("format=rgb24")
    return ",".join(parts)


def decode_frames(info: MediaInfo, aw: int, ah: int, fps: float, start_frame: int = 0, frames: int | None = None,
                  threads: int = 0):
    """Yield raw RGB24 frames (bytes) at a constant `fps`, downscaled to aw x ah, starting at output frame
    `start_frame` of the clip.

    -copyts keeps the source's own timestamps, so the fps filter snaps frames to the same absolute grid whether
    we decode from the top or from the middle; the segment is then cut at an exact grid line with select. (Seeking
    without it re-bases time at the seek point, and the fps filter then drops one frame and doubles the next
    near the segment start.) Verified frame-exact against a frame-numbered HEVC clip with 1 s GOPs."""
    cmd = [tool("ffmpeg"), "-v", "error", "-nostdin"]
    if threads:
        cmd += ["-threads", str(threads), "-filter_threads", "1"]
    vf = _vf(info, aw, ah, fps)
    if start_frame > 0:
        cmd += ["-ss", f"{max(0.0, start_frame / fps - 2.0):.4f}"]      # land a little early, trim exactly
        tick0 = round(info.extra.get("start_time", 0.0) * fps)
        # select on pts, not trim: trim keeps a frame whose *duration* overlaps the start, and durations aren't
        # reliable after a seek, so it sometimes lets the previous frame through.
        vf = vf.replace(f"fps={fps},", f"fps={fps},select=gte(t\\,{(tick0 + start_frame - 0.5) / fps:.6f}),", 1)
    cmd += ["-noaccurate_seek"] if start_frame > 0 else []
    cmd += ["-copyts", *hwdec_args(), "-i", info.path, "-an", "-sn", "-dn", "-vf", vf]
    if frames:
        cmd += ["-frames:v", str(frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
                            bufsize=aw * ah * 3 * 4)
    n = aw * ah * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield buf
        if proc.wait() != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg decode failed: {err.read().decode(errors='replace')[-800:]}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        err.close()


def load_still(info: MediaInfo, aw: int, ah: int) -> np.ndarray:
    with Image.open(info.image_path or info.path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(im)
    return cv2.resize(arr, (aw, ah), interpolation=cv2.INTER_AREA)


def grab_frame(info: MediaInfo, t: float, aw: int, ah: int) -> np.ndarray:
    """One RGB frame at time t (seconds), downscaled to aw x ah."""
    if info.kind == "image":
        return load_still(info, aw, ah)
    t = max(0.0, min(t, max(0.0, info.duration - 0.1)))
    for tt in (t, max(0.0, t - 0.5), 0.0):
        cmd = [tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{tt:.3f}", *hwdec_args(), "-i", info.path,
               "-an", "-sn",
               "-dn", "-vf", _vf(info, aw, ah, None), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "pipe:1"]
        data = subprocess.run(cmd, capture_output=True).stdout
        if len(data) >= aw * ah * 3:
            return np.frombuffer(data[:aw * ah * 3], np.uint8).reshape(ah, aw, 3)
    raise RuntimeError(f"couldn't grab a frame at {t:.2f}s from {Path(info.path).name}")


ENCODERS = {
    "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"],
    "videotoolbox": ["-c:v", "h264_videotoolbox", "-q:v", "70", "-allow_sw", "1"],
}


class Encoder:
    """Raw RGB frames in on stdin -> H.264 BT.709 MP4 (video only).

    encoder: a name from ENCODERS, or a list of ffmpeg codec arguments such as ["-c:v", "libx264", ...]."""

    def __init__(self, dst: Path, W: int, H: int, fps: float, encoder="x264", threads: int = 0):
        self.err = tempfile.TemporaryFile()
        codec = ENCODERS[encoder] if isinstance(encoder, str) else list(encoder)
        cmd = [tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"] + codec
        if threads:
            cmd += ["-threads", str(threads)]
        cmd += ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-f", "mp4", str(dst)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def _errtext(self):
        self.err.seek(0)
        return self.err.read().decode(errors="replace")[-1500:]

    def write(self, data: bytes):
        try:
            self.proc.stdin.write(data)
        except BrokenPipeError:
            self.proc.wait()
            raise RuntimeError(f"ffmpeg encode failed: {self._errtext()}") from None

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg encode failed: {self._errtext()}")
        self.err.close()

    def abort(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.err.close()


def mux(parts: list, dst: Path, audio_src: str | None, audio_index: int | None):
    """Concatenate video-only segments (stream copy) and add the source audio, or silence."""
    lst = dst.with_suffix(".txt")
    lst.write_text("".join("file '{}'\n".format(str(p).replace("'", "'\\''")) for p in parts))
    cmd = [tool("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if audio_src is not None and audio_index is not None:
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", f"1:a:{audio_index}", "-af", "apad"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest",
            "-movflags", "+faststart", "-f", "mp4", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr[-1500:]}")


# --------------------------------------------------------------------------- parallel render
#
# Profiling showed the per-frame ASCII render is cheap (~2-6 ms) and a single process tops out around
# 150 fps because every 1080p frame (6 MB of RGB) must squeeze through pipes. So instead of farming
# frames out to a pool, the clip is cut into time segments and each worker runs its own complete
# decode -> render -> encode pipeline. The segments are then joined by stream copy and the audio added.

MIN_SEGMENT_FRAMES = 150          # ~5 s at 30 fps; shorter clips render in a single process
_SEG: dict = {}


def _seg_init(counter, cancel_event):
    cv2.setNumThreads(1)
    _SEG.update(counter=counter, cancel=cancel_event)


def _render_segment(job: dict) -> int:
    """Render frames [start_frame, start_frame + n_frames) to job['out'] (video only). Returns frame count."""
    info, style, fps, dur = job["info"], Style(job["style"]), job["fps"], job["dur"]
    W, H, aw, ah, font_path = job["W"], job["H"], job["aw"], job["ah"], job["font"]
    counter, cancel, report = _SEG.get("counter"), _SEG.get("cancel"), job.get("report")
    start, n = job["start_frame"], job["n_frames"]
    enc = Encoder(Path(job["out"]), W, H, fps, job["encoder"], job["threads"])
    if info.kind == "video":
        frames = decode_frames(info, aw, ah, fps, start, n, job["threads"])
    else:
        still = load_still(info, aw, ah)
        frames = (None for _ in range(n))
    done, last_fs, last_out, pending = 0, None, None, 0
    try:
        for buf in frames:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            t = (start + done) / fps
            fs = style.at(t, dur)
            if buf is None and fs == last_fs:
                out = last_out                              # identical still frame: reuse
            else:
                src = still if buf is None else np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
                out = render_frame(src, fs, W, H, font_path).tobytes()
            enc.write(out)
            last_fs, last_out = fs, out
            done += 1
            pending += 1
            if pending >= 5:
                if counter is not None:
                    with counter.get_lock():
                        counter.value += pending
                if report:
                    report(pending)
                pending = 0
        enc.close()
    except BaseException:
        enc.abort()
        raise
    finally:
        if hasattr(frames, "close"):
            frames.close()
    if pending:
        if counter is not None:
            with counter.get_lock():
                counter.value += pending
        if report:
            report(pending)
    return done


def stop_pool(pool: ProcessPoolExecutor, ev=None, graceful: bool = True) -> None:
    """Shut a worker pool down; on error/cancel, ask workers to stop, then kill any that don't."""
    if graceful:
        pool.shutdown(wait=True)
        return
    procs = list((getattr(pool, "_processes", None) or {}).values())
    if ev is not None:
        ev.set()                                   # workers check this every frame and abort their encoders
    pool.shutdown(wait=False, cancel_futures=True)
    deadline = time.time() + 3
    for p in procs:
        p.join(max(0.0, deadline - time.time()))
    for p in procs:
        if p.is_alive():
            p.kill()


def default_jobs() -> int:
    # Benchmarked on an M5 Pro (18 cores) with 60 s iPhone-style 1080p30 HEVC: 6 segments beat 4 for both
    # SDR (356 vs 322 fps) and 10-bit HLG (332 vs 309 fps); 8 was slower again.
    return max(1, min(6, (os.cpu_count() or 4) // 3))


def render(src, dst, style=None, *, fps: float = 30, size: int = 1920, photo_seconds: float = 5.0,
           jobs: int | None = None, encoder: str = "x264", font: str | None = None,
           progress=None, cancel=None, cache_dir=None, segments: int | None = None) -> dict:
    """Render one video or photo to an ASCII-art MP4 (H.264 + AAC). Returns stats.

    progress(done_frames, total_frames) is called from this thread; cancel is a threading.Event.
    """
    t_start = time.time()
    info = probe(src, cache_dir)
    style = style if isinstance(style, Style) else Style(style)
    font_path = find_font(font)
    W, H = out_size(info.width, info.height, size)
    dur = info.duration if info.kind == "video" else float(photo_seconds)
    total = max(1, int(round(dur * fps)))
    aw, ah = analysis_size(info.width, info.height, style.max_cols())
    jobs = max(1, int(jobs or default_jobs()))
    nseg = segments or max(1, min(jobs, total // MIN_SEGMENT_FRAMES))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = dst.parent / f".{dst.stem}.parts"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    tmp = dst.with_name(f".{dst.stem}.partial.mp4")
    cpu = os.cpu_count() or 4
    base = dict(info=info, style=style.spec, fps=fps, dur=dur, W=W, H=H, aw=aw, ah=ah, font=font_path,
                encoder=encoder, threads=max(2, cpu // nseg) if nseg > 1 else 0)
    bounds = [round(k * total / nseg) for k in range(nseg + 1)]
    seg_jobs = []
    for k in range(nseg):
        last = k == nseg - 1
        n = bounds[k + 1] - bounds[k]
        if info.kind == "video" and last:
            n = None                                        # let the last segment run to the true end
        seg_jobs.append(dict(base, start_frame=bounds[k], n_frames=n, out=str(work / f"seg{k:03d}.mp4")))

    done = 0
    last_report = [0.0]

    def tick(k):
        nonlocal done
        done += k
        now = time.time()
        if progress and now - last_report[0] > 0.25:
            last_report[0] = now
            progress(done, max(total, done))

    pool = ev = None
    ok = False
    try:
        if nseg == 1:
            _SEG.clear()
            if cancel is not None:
                _SEG["cancel"] = cancel
            frames = _render_segment(dict(seg_jobs[0], report=tick))
        else:
            # ProcessPoolExecutor, not multiprocessing.Pool: if a worker dies, Pool silently respawns it and
            # the job waits forever; the executor fails the futures with BrokenProcessPool instead.
            ctx = mp.get_context("spawn")
            counter, ev = ctx.Value("i", 0), ctx.Event()
            pool = ProcessPoolExecutor(nseg, mp_context=ctx, initializer=_seg_init, initargs=(counter, ev))
            futs = [pool.submit(_render_segment, j) for j in seg_jobs]
            while True:
                finished, running = wait(futs, timeout=0.2, return_when=FIRST_EXCEPTION)
                failed = next((f for f in finished if f.exception() is not None), None)
                if failed is not None:
                    raise failed.exception()
                if not running:
                    break
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if progress:
                    progress(counter.value, max(total, counter.value))
            frames = sum(f.result() for f in futs)
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        mux([Path(j["out"]) for j in seg_jobs], tmp, info.path if info.kind == "video" else None,
            info.audio_index if info.kind == "video" else None)
        os.replace(tmp, dst)
        ok = True
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            stop_pool(pool, ev, graceful=ok)
        shutil.rmtree(work, ignore_errors=True)
    secs = time.time() - t_start
    if progress:
        progress(frames, frames)
    return {"frames": frames, "seconds": round(secs, 2), "fps": round(frames / secs, 1) if secs else 0,
            "width": W, "height": H, "duration": round(frames / fps, 3), "kind": info.kind, "segments": nseg}


def preview(src, t: float, out_png, style=None, *, size: int = 1920, photo_seconds: float = 5.0,
            font: str | None = None, cache_dir=None) -> Path:
    """Render the single frame at time t to a PNG."""
    info = probe(src, cache_dir)
    style = style if isinstance(style, Style) else Style(style)
    W, H = out_size(info.width, info.height, size)
    dur = info.duration if info.kind == "video" else float(photo_seconds)
    fs = style.at(t, dur)
    aw, ah = analysis_size(info.width, info.height, fs.cols)
    img = render_frame(grab_frame(info, t, aw, ah), fs, W, H, find_font(font))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_png, compress_level=1)
    return out_png


# --------------------------------------------------------------------------- CLI

def add_style_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("style (single value or 'seconds:value,...' keyframes; '50%:' = half-way)")
    g.add_argument("--cols", help="characters across, e.g. 120 or '0:20,180:160'")
    g.add_argument("--color", help="mono | color | 0..1 blend, e.g. '0%%:mono,100%%:color'")
    g.add_argument("--charset", help=f"{' | '.join(CHARSETS)} or a literal string (sparse -> dense)")
    g.add_argument("--tint", help="glyph colour in mono, e.g. '#ffffff' or green/amber/moss")
    g.add_argument("--bg", help="background colour, e.g. '#000000' or paper/night")
    g.add_argument("--contrast", help="luminance contrast, 1.0 = unchanged")
    g.add_argument("--invert", nargs="?", const="true", default=None,
                   help="dense glyphs in dark areas (use with a light --bg)")


def style_from_args(a) -> dict:
    return {k: getattr(a, k) for k in STYLE_KINDS if getattr(a, k, None) is not None}


def cli_progress(prefix=""):
    t0 = time.time()

    def cb(done, total):
        el = time.time() - t0
        rate = done / el if el else 0
        bar = "#" * int(30 * done / max(1, total))
        sys.stderr.write(f"\r{prefix}[{bar:<30}] {done}/{total} frames  {rate:5.1f} fps ")
        if done >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return cb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    add_style_args(ap)
    ap.add_argument("--preview", type=float, metavar="SECONDS", help="render one test frame to PNG")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--size", type=int, default=1920, help="long edge of the output, px (default 1920)")
    ap.add_argument("--photo-seconds", type=float, default=5.0, help="length of the clip made from a photo")
    ap.add_argument("--jobs", type=int, default=None, help=f"parallel segments (default {default_jobs()})")
    ap.add_argument("--encoder", choices=list(ENCODERS), default="x264")
    ap.add_argument("--font", help="path to a monospace .ttf/.ttc (default Menlo)")
    a = ap.parse_args(argv)
    try:
        style = Style(style_from_args(a))
    except ValueError as e:
        ap.error(str(e))
    src = Path(a.input)
    if a.preview is not None:
        out = Path(a.output) if a.output and a.output.lower().endswith(".png") else \
            src.with_name(f"{src.stem}_preview_{a.preview:g}s.png")
        print(preview(src, a.preview, out, style, size=a.size, photo_seconds=a.photo_seconds, font=a.font))
        return
    out = Path(a.output) if a.output else src.with_name(f"{src.stem}_ascii.mp4")
    stats = render(src, out, style, fps=a.fps, size=a.size, photo_seconds=a.photo_seconds, jobs=a.jobs,
                   encoder=a.encoder, font=a.font, progress=cli_progress())
    print(f"{out}  ({stats['frames']} frames {stats['width']}x{stats['height']} in {stats['seconds']}s, "
          f"{stats['fps']} fps)")


if __name__ == "__main__":
    main()
