"""Video clips in the image pools: a cut to a clip plays it.

Put clips (.mov .mp4 .m4v and the other formats ./ascii reads) in images/bright, texture or dark beside the
photos, or in the inbox to sort them like photos. The edit treats a clip like any photo: the sound picks the
pool, the cut picks the clip, and the crop/zoom, colour shift, layering, hand crop, sets and sections all
apply. The difference: while the shot is on screen (cut, hold, decay) the clip plays, one clip frame per film
frame at the film's fps (times clips.speed). Its sound is never used.

  clock: shots   a clip only moves while it's on screen, so each cut to it carries on where the last stopped
  clock: film    every clip runs under the whole film; each cut is a window onto it
  clock: random  each cut starts somewhere new in the clip (seeded per cut, so it's the same every time)

Every clip loops at its end, after skipping clips.start seconds. The score gives each frame the clip's frame
number (the EDL's `clip` and `layer_clip` columns). Before drawing, a render decodes every clip frame it needs
once, in one pass per clip (ffmpeg only seeks across long gaps), into its scratch folder next to the parts;
the segments read them from there. Frame numbers are exact from any starting point
(ascii_film.decode_frames), so a frame comes out the same whichever process draws it, and the dashboard's
live tiles, which decode the odd frame on demand, match the render.
"""
from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

import ascii_film as af

from . import background

VIDEO_EXTS = background.VIDEO_EXTS
GAP_S = 3.0                     # needed frames further apart than this: seek (~0.25 s) instead of decoding through


def is_clip(path) -> bool:
    return Path(str(path)).suffix.lower() in af.VIDEO_EXTS


def info(path) -> af.MediaInfo:
    return background.probe(Path(path))


def length(path, fps: float) -> int:
    """Whole frames at the film's fps, a little short of the end so the last one always decodes."""
    i = info(path)
    if i.kind != "video":
        raise SystemExit(f"{Path(path).name} is not a video")
    return int(math.floor(max(0.0, i.duration - 0.05) * fps))


def lengths(pools: dict, fps: float) -> dict[str, int]:
    """{readable path: frames} for every clip in the pools."""
    out = {}
    for items in pools.values():
        for name, path in items:
            if is_clip(path) and path not in out:
                try:
                    out[path] = length(path, fps)
                except (RuntimeError, OSError) as e:
                    raise SystemExit(f"images/{name}: can't read it as a video ({e})")
                if out[path] < 1:
                    raise SystemExit(f"images/{name} is too short to use")
    return out


def still(path: Path, cache_dir: Path) -> Path:
    """A frame from the middle of the clip as a cached JPEG (thumbnails, the sorter's scores). Images pass
    through ascii_film.readable_image (HEIC -> JPEG)."""
    path = Path(path)
    if not is_clip(path):
        return af.readable_image(path, cache_dir)
    st = path.stat()
    out = Path(cache_dir) / "stills" / f"{path.stem}-{path.suffix[1:].lower()}-{st.st_size}-{int(st.st_mtime)}.jpg"
    if not out.exists():
        i = info(path)
        s = min(1.0, 1920 / max(i.width, i.height))
        rgb = af.grab_frame(i, i.duration / 2, af.even(i.width * s), af.even(i.height * s))
        out.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"couldn't make a still of {path.name}")
        tmp = out.with_name(f".{out.name}.{threading.get_ident()}")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(out)
    return out


# --------------------------------------------------------------------------- the score

def assign(rows: list[dict], cfg: dict, lens: dict[str, int], seed: int) -> None:
    """Give each frame showing a clip the clip's frame number: row['clip'] (the shot's clip) and
    row['layer_clip'] (a clip layered in), -1 where it's a photo or nothing."""
    c, fps = cfg["clips"], cfg["video"]["fps"]
    clock, speed = c["clock"], float(c["speed"])
    played: dict[str, float] = {}               # clock shots: how far each clip has got
    for r in rows:
        r["clip"] = r["layer_clip"] = -1
        if r["shot"] < 0:
            continue
        for key, src, part in (("clip", "image", 0), ("layer_clip", "layer", 1)):
            n = lens.get(r[src]) if r[src] else None
            if not n:
                continue
            s0 = min(n - 1, max(0, int(round(float(c["start"]) * fps))))
            loop = n - s0
            if clock == "film":
                t = r["frame"] * speed
            elif clock == "random":             # its own generator per cut: other shots don't move it
                t = np.random.default_rng([seed, 31, r["shot"], part]).random() * loop + r["since"] * speed
            else:
                t = played.get(r[src], 0.0)
                played[r[src]] = t + speed
            r[key] = s0 + int(math.floor(t + 1e-9)) % loop


def needed(rows: list[dict]) -> dict[str, list[int]]:
    """{clip path: the clip frames these rows show, sorted}."""
    out: dict[str, set] = {}
    for r in rows:
        if r.get("clip", -1) >= 0:
            out.setdefault(r["image"], set()).add(r["clip"])
        if r.get("layer_clip", -1) >= 0:
            out.setdefault(r["layer"], set()).add(r["layer_clip"])
    return {p: sorted(s) for p, s in out.items()}


# --------------------------------------------------------------------------- frames

def levels_for(root: Path, cfg: dict, paths) -> dict[str, dict]:
    """{clip: {'levels': [lo, hi] or None}}: like the photos' auto-level, but one stretch per clip (per frame
    it would flicker), from its keyframes. Cached in .work/background_levels.json."""
    auto = bool(cfg["ascii"].get("auto_level", True))
    return {p: {"levels": background.levels(root, Path(p), info(p)) if auto else None} for p in paths}


class Prep:
    """Turns the clip's decoded frames into what FrameRenderer.image gives for a photo: your rotation and
    crop, the cover crop to the frame's shape, `work_w` pixels across, auto-levelled."""

    def __init__(self, inf: af.MediaInfo, view: dict | None, aspect: float, work_w: int, levels):
        self.view, self.aspect, self.work_w, self.levels = view, aspect, work_w, levels
        w, h = inf.width, inf.height
        rot = math.radians(float((view or {}).get("rot", 0)))
        rw, rh = w * abs(math.cos(rot)) + h * abs(math.sin(rot)), w * abs(math.sin(rot)) + h * abs(math.cos(rot))
        vw, vh = (rw * view["w"], rh * view["h"]) if view and "w" in view else (rw, rh)
        shown = min(vw, vh * aspect)            # source pixels across the part the film shows
        s = min(1.0, 2 * work_w / max(shown, 1.0))
        self.dw, self.dh = af.even(w * s), af.even(h * s)

    def __call__(self, buf) -> np.ndarray:
        from .frames import apply_view, fit
        arr = np.frombuffer(buf, np.uint8).reshape(self.dh, self.dw, 3)
        img = fit(apply_view(arr, self.view), self.aspect, self.work_w)
        if self.levels:
            lo, hi = self.levels
            img = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        return img

    def blank(self) -> np.ndarray:              # past the end of the clip: black, wherever it's read from
        return np.zeros_like(self(bytes(self.dw * self.dh * 3)))


def runs(idx: list[int], gap: int) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for i in idx:
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def decode(path: str, prep: Prep, fps: float, idx: list[int]):
    """Yield (clip frame, work image) for the sorted frame numbers idx."""
    inf, want = info(path), set(idx)
    for a, b in runs(idx, int(GAP_S * fps)):
        i = a
        for buf in af.decode_frames(inf, prep.dw, prep.dh, fps, start_frame=a, frames=b - a + 1):
            if i in want:
                yield i, prep(buf)
            i += 1
        for j in range(i, b + 1):
            if j in want:
                yield j, prep.blank()


def extract(job: dict) -> dict:
    """Decode the frames a render needs from one clip into job['out'] (.npy, in frame order). Returns the
    manifest entry: {path, file, index: {clip frame: row in the file}}."""
    prep = Prep(info(job["path"]), job["view"], job["aspect"], job["work_w"], job["levels"])
    pos = {i: k for k, i in enumerate(job["idx"])}
    arr = None
    for i, img in decode(job["path"], prep, job["fps"], job["idx"]):
        if arr is None:
            arr = np.lib.format.open_memmap(job["out"], mode="w+", dtype=np.uint8, shape=(len(pos),) + img.shape)
        arr[pos[i]] = img
    if arr is not None:
        arr.flush()
        del arr
    return {"path": job["path"], "file": job["out"], "index": pos}


_NOW: OrderedDict = OrderedDict()
_NOW_LOCK = threading.Lock()


def frame_now(path: str, i: int, view, aspect: float, work_w: int, levels, fps: float) -> np.ndarray:
    """One work image decoded on demand (the dashboard's live tiles), the same as extract() gives."""
    st = Path(path).stat()
    key = (path, st.st_size, st.st_mtime_ns, i, json.dumps(view, sort_keys=True), aspect, work_w,
           tuple(levels) if levels else None, fps)
    with _NOW_LOCK:
        if key in _NOW:
            _NOW.move_to_end(key)
            return _NOW[key]
    prep = Prep(info(path), view, aspect, work_w, levels)
    img = next(decode(path, prep, fps, [i]))[1]
    with _NOW_LOCK:
        _NOW[key] = img
        while len(_NOW) > 48:
            _NOW.popitem(last=False)
    return img
