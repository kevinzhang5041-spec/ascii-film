"""A video instead of black: where the track is silent (or wherever the film would be black), a clip plays.

The clip lives at paths.background (background/video.mov by default; .mov .mp4 .m4v and the other formats
./ascii reads). Its options are under `background:` in config.yaml. With no clip, or background.enabled
false, the film is exactly what it was before this existed.

  clock: film   the video runs under the whole film from `start`; each silence is a window onto it
  clock: gaps   the video only moves while it shows, so each silence carries on where the last one stopped

Either way it loops at its end. The score gives each frame that shows it a frame number of the video (the
EDL's `background` column) and a brightness. Frames are read at the film's fps with ascii_film.decode_frames,
which is frame-exact from any starting point, so a frame comes out the same whichever process draws it.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

import ascii_film as af

VIDEO_EXTS = tuple(sorted(af.VIDEO_EXTS))
_PROBES: dict = {}


def _base(root: Path, cfg: dict) -> Path:
    p = Path(cfg["paths"]["background"])
    return p if p.is_absolute() else Path(root) / p


def files(root: Path, cfg: dict) -> list[Path]:
    """Every file that counts as the background video (background/video.mov, .mp4, ...)."""
    want = _base(root, cfg)
    if want.suffix.lower() in VIDEO_EXTS:
        return [want] if want.is_file() else []
    if not want.parent.is_dir():
        return []
    names = {(want.name + ext).lower() for ext in VIDEO_EXTS}
    # the real directory listing: the Mac's disk ignores case, so video.mov and video.MOV are one file
    return sorted(p for p in want.parent.iterdir() if p.name.lower() in names and p.is_file())


def find(root: Path, cfg: dict) -> Path | None:
    fs = files(root, cfg)
    return fs[0] if fs else None


def dest(root: Path, cfg: dict, ext: str) -> Path:
    """Where a new background video with this extension goes."""
    want = _base(root, cfg)
    return want if want.suffix.lower() in VIDEO_EXTS else want.with_name(want.name + ext.lower())


def probe(path: Path) -> af.MediaInfo:
    st = Path(path).stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _PROBES:
        _PROBES[key] = af.probe(path)
    return _PROBES[key]


def plan(root: Path, cfg: dict) -> dict | None:
    """What the score and the renderer need to know about the video, or None when there's none to use."""
    if not cfg["background"]["enabled"]:
        return None
    p = find(root, cfg)
    if p is None:
        return None
    info = probe(p)
    if info.kind != "video":
        raise SystemExit(f"{p.name} is not a video")
    fps = cfg["video"]["fps"]
    # whole frames at the film's fps, a little short of the end so the last one always decodes
    frames = int(math.floor(max(0.0, info.duration - 0.05) * fps))
    if frames < 1:
        raise SystemExit(f"{p.name} is too short to use as the background video")
    start = min(frames - 1, max(0, int(round(float(cfg["background"]["start"]) * fps))))
    lv = levels(root, p, info) if cfg["ascii"].get("auto_level", True) else None
    from .title import frames as title_frames
    # lead: frames of the video the title card shows before the film, so the film carries on from there
    return dict(path=str(p), name=p.name, info=info, fps=fps, frames=frames, start=start, levels=lv,
                lead=title_frames(cfg))


def levels(root: Path, p: Path, info: af.MediaInfo) -> list[float] | None:
    """Like the photos' auto-level, but one stretch for the whole clip (per frame it would flicker): the
    0.5 / 99.5 % luminance over its keyframes. Cached in .work/background_levels.json."""
    st = p.stat()
    key = f"{p.name}|{st.st_size}|{st.st_mtime_ns}"
    cache = Path(root) / ".work" / "background_levels.json"
    try:
        known = json.loads(cache.read_text())
    except (OSError, json.JSONDecodeError):
        known = {}
    if key not in known:
        aw = 96
        ah = af.even(aw * info.height / info.width)
        cmd = [af.tool("ffmpeg"), "-v", "error", "-nostdin", "-skip_frame", "nokey", "-i", str(p), "-an", "-sn",
               "-dn", "-vf", af._vf(info, aw, ah, None), "-fps_mode", "passthrough", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "pipe:1"]
        raw = subprocess.run(cmd, capture_output=True).stdout
        n = len(raw) // (aw * ah * 3)
        if n:
            lum = np.frombuffer(raw[:n * aw * ah * 3], np.uint8).reshape(-1, 3).astype(np.float32) @ af.LUMA * 255
            lo, hi = np.percentile(lum, [0.5, 99.5])
            known[key] = [round(float(lo), 2), round(float(hi), 2)]
        else:
            known[key] = None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1))
    lv = known[key]
    return lv if lv and lv[1] - lv[0] > 8 else None


def assign(rows: list[dict], cfg: dict, bg: dict | None) -> None:
    """Give each frame its background: `bg` = the video's frame number there (-1: none), `bg_gain`."""
    b = cfg["background"]
    show = ("silence",) if b["show_in"] == "silence" else ("silence", "black")
    level, fade = float(np.clip(b["level"], 0, 1)), max(0, int(b["fade_frames"]))
    loop = bg["frames"] - bg["start"] if bg else 0
    lead = bg.get("lead", 0) if bg else 0          # the title card played the first `lead` frames of it
    shown = run = lead
    for r in rows:
        if bg and r["kind"] in show:
            i = r["frame"] + lead if b["clock"] == "film" else shown
            run += 1
            shown += 1
            r["bg"] = bg["start"] + i % loop
            r["bg_gain"] = round(level * min(1.0, run / (fade + 1)), 4)
            r["reason"] += f", background video {r['bg']}"
        else:
            r["bg"], r["bg_gain"] = -1, 0.0
            run = 0


class Reader:
    """The video's frames at the film's fps, centre-cropped to the frame's shape, `width` pixels across (or
    less if the video is smaller). Read in order; an earlier frame, or one far ahead, restarts the decoder there."""

    def __init__(self, bg: dict, aspect: float, width: int):
        info = self.info = bg["info"]
        self.fps = bg["fps"]
        w, h = info.width, info.height
        crop_w, crop_h = (h * aspect, h) if w / h > aspect else (w, w / aspect)
        s = min(1.0, width / crop_w)
        self.aw, self.ah = af.even(w * s), af.even(h * s)
        self.cw, self.ch = min(self.aw, max(2, round(crop_w * s))), min(self.ah, max(2, round(crop_h * s)))
        self.x0, self.y0 = (self.aw - self.cw) // 2, (self.ah - self.ch) // 2
        self.jump = int(self.fps * 8)                 # further ahead than this: seek rather than decode through
        self.gen, self.at, self.last = None, 0, None

    def frame(self, i: int) -> np.ndarray:
        if self.last is not None and self.last[0] == i:
            return self.last[1]
        if self.gen is None or i < self.at or i - self.at > self.jump:
            self.close()
            self.gen, self.at = af.decode_frames(self.info, self.aw, self.ah, self.fps, start_frame=i), i
        buf = None
        while self.at <= i:
            buf = next(self.gen, None)
            if buf is None:
                break
            self.at += 1
        if buf is None:                               # past the end: black (the same wherever it's read from)
            self.close()
            img = np.zeros((self.ch, self.cw, 3), np.uint8)
        else:
            full = np.frombuffer(buf, np.uint8).reshape(self.ah, self.aw, 3)
            img = np.ascontiguousarray(full[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw])
        self.last = (i, img)
        return img

    def close(self) -> None:
        if self.gen is not None:
            self.gen.close()                          # stops ffmpeg
        self.gen = None
