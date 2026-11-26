"""Step 3a: draw one frame of the score.

  image -> crop/zoom -> (layer) -> colour shift -> black point -> exposure x decay   (at cell resolution)
        -> glyph indices (ascii_film.cell_glyphs) -> character jitter
        -> glyphs (ascii_film.draw_glyphs) -> grain -> gate weave -> (debug overlay)

A frame with no photo is black, or the background video's frame (background.py) if the score gave it one:
  video frame -> centre crop -> cells -> black point -> x level -> glyphs (or the picture itself, dimmed)
        -> grain -> gate weave

Per-frame randomness comes from generators seeded by (seed, frame), so frames can be drawn in any order,
in any process, and always come out the same.
"""
from __future__ import annotations

import dataclasses
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageOps

import ascii_film as af

from .background import Reader


def glyph_ramp(chars: str, font_path: str, sort: bool = True, keep_black: bool = True) -> str:
    """De-duplicate a ramp, optionally sort it sparse -> dense by measured ink, and start it with a space."""
    seen = "".join(dict.fromkeys(chars))
    if keep_black and " " not in seen:
        seen = " " + seen
    if sort:
        atlas = af.glyph_atlas(seen, 16, 32, font_path)
        ink = atlas.reshape(len(seen), -1).sum(1)
        seen = "".join(c for _, _, c in sorted(zip(ink, range(len(seen)), seen)))
    return seen


def hue_matrix(degrees: float) -> np.ndarray:
    """3x3 RGB -> RGB matrix rotating hue by `degrees` in YIQ space (keeps luminance)."""
    t = np.deg2rad(degrees)
    to_yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return (np.linalg.inv(to_yiq) @ rot @ to_yiq).astype(np.float32)


def shift(img: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """Translate by (dx, dy) pixels, filling the uncovered edge with `fill`."""
    if not dx and not dy:
        return img
    H, W = img.shape[:2]
    out = np.empty_like(img)
    out[:] = fill
    w, h = W - abs(dx), H - abs(dy)
    if w <= 0 or h <= 0:
        return out
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    out[yd:yd + h, xd:xd + w] = img[ys:ys + h, xs:xs + w]
    return out


def rotate(arr: np.ndarray, deg: float) -> np.ndarray:
    """Rotate clockwise by deg. Quarter turns are exact; other angles grow the canvas and fill the corners
    with black (empty space in the film)."""
    deg = float(deg) % 360
    q = round(deg / 90)
    if abs(deg - q * 90) < 1e-6:
        return np.ascontiguousarray(np.rot90(arr, -q % 4))
    h, w = arr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)     # OpenCV turns counter-clockwise for +
    c, s_ = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * s_ + w * c)), int(round(h * c + w * s_))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(arr, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def apply_view(arr: np.ndarray, v: dict | None) -> np.ndarray:
    """A hand view from images/crops.json: {rot} (degrees clockwise), then {x, y, w, h} (fractions of the
    rotated photo). Without one, the photo is unchanged (the caller then takes the centre 16:9)."""
    if not v:
        return arr
    if float(v.get("rot", 0)) % 360:
        arr = rotate(arr, v["rot"])
    if "w" in v:
        h, w = arr.shape[:2]
        x0, y0 = int(round(v["x"] * w)), int(round(v["y"] * h))
        x1, y1 = max(x0 + 2, int(round((v["x"] + v["w"]) * w))), max(y0 + 2, int(round((v["y"] + v["h"]) * h)))
        arr = arr[y0:min(h, y1), x0:min(w, x1)]
    return arr


class FrameRenderer:
    """Turns score rows into (H, W, 3) RGB uint8 frames."""

    def __init__(self, cfg: dict, bands: list[dict], W: int, H: int, font_path: str, seed: int, overlay=None,
                 views: dict | None = None, background: dict | None = None):
        self.cfg, self.W, self.H, self.font, self.seed, self.overlay = cfg, W, H, font_path, seed, overlay
        self.views = views or {}                     # {photo path: hand crop {x, y, w, h}} (images/crops.json)
        a, cs = cfg["ascii"], cfg["ascii"]["charset"]
        self.ramps = [glyph_ramp(b["chars"], font_path, cs["sort_by_coverage"], cs["keep_black"]) for b in bands]
        self.bgv, self._reader = background, None    # background.plan(): the video instead of black, if any
        bc = str(cfg["background"]["chars"])
        self.bg_ramp = glyph_ramp(bc, font_path, cs["sort_by_coverage"], cs["keep_black"]) if bc.strip() else None
        self.tint, self.bg = af.parse_color(a["tint"]), af.parse_color(a["background"])
        self.contrast = float(a["contrast"])
        self.auto_level = bool(a.get("auto_level", True))
        self.black_point = float(a.get("black_point", 0.0))
        self.saturation = float(a["color"].get("saturation", 1.0))
        fl = cfg["editing"]["flash"]
        self.ink, self.paper = af.parse_color(fl["ink"]), af.parse_color(fl["paper"])
        tr = cfg["treatments"]
        self.include_blank = bool(tr["char_jitter"].get("include_blank", False))
        self.blend = tr["layering"].get("blend", "screen")
        # source images are kept at a "work" size: enough for the finest grid at the deepest zoom
        zmax = 1 + 2 * tr["crop_zoom"]["intensity"] if tr["crop_zoom"]["enabled"] else 1
        self.work_w = int(np.clip(a["resolution"]["cols_loud"] * zmax * 1.5, 480, 1920))
        self._images: OrderedDict = OrderedDict()
        self._grain = None

    # ---- sources -------------------------------------------------------------------------------------
    def image(self, path: str) -> np.ndarray:
        img = self._images.get(path)
        if img is not None:
            self._images.move_to_end(path)
            return img
        with Image.open(path) as im:
            im.draft("RGB", (self.work_w * 2, self.work_w * 2))    # JPEG: decode big scans at reduced size
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        arr = apply_view(arr, self.views.get(path))  # your rotation and crop, if you made them
        h, w = arr.shape[:2]
        target = self.W / self.H                     # cover-crop to the frame's shape
        if w / h > target:
            cw = int(round(h * target))
            arr = arr[:, (w - cw) // 2:(w - cw) // 2 + cw]
        else:
            ch = int(round(w / target))
            arr = arr[(h - ch) // 2:(h - ch) // 2 + ch]
        ww = min(self.work_w, arr.shape[1])
        img = cv2.resize(arr, (ww, max(2, int(round(ww / target)))), interpolation=cv2.INTER_AREA)
        if self.auto_level:                          # stretch luminance 0.5-99.5 % to full range, keep hue
            lum = img.astype(np.float32) @ af.LUMA * 255
            lo, hi = np.percentile(lum, [0.5, 99.5])
            if hi - lo > 8:
                img = np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        self._images[path] = img
        if len(self._images) > 96:
            self._images.popitem(last=False)
        return img

    def cells(self, path: str, crop, cols: int, rows: int) -> np.ndarray:
        """The (cropped) image averaged down to one colour per character cell, float32 0-1."""
        img = self.image(path)
        h, w = img.shape[:2]
        cx, cy, z = crop or (0.5, 0.5, 1.0)
        cw, ch = max(2, int(round(w / z))), max(2, int(round(h / z)))
        x0 = int(np.clip(round(cx * w - cw / 2), 0, w - cw))
        y0 = int(np.clip(round(cy * h - ch / 2), 0, h - ch))
        view = img[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(view, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255)

    # ---- per-frame treatments ------------------------------------------------------------------------
    def _grain_tiles(self):
        """8 full-frame grain gain maps (128 = no change), made once per process from the seed."""
        if self._grain is None:
            amt = float(self.cfg["treatments"]["grain"]["intensity"])
            rng = np.random.default_rng([self.seed, 23])
            tiles = []
            for _ in range(8):
                g = rng.standard_normal((self.H // 2 + 1, self.W // 2 + 1)).astype(np.float32)
                g = cv2.GaussianBlur(g, (0, 0), 0.7)
                g = cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                g = np.clip(128 * (1 + 0.5 * amt * g / max(g.std(), 1e-6)), 0, 255).astype(np.uint8)
                tiles.append(cv2.merge([g, g, g]))
            self._grain = tiles
        return self._grain

    def grain(self, out: np.ndarray, rng, on_paper: bool = False) -> np.ndarray:
        """Multiplicative grain on the ink only: black background (or white paper on a flash) stays clean."""
        tiles = self._grain_tiles()
        t = tiles[int(rng.integers(len(tiles)))]
        flip = int(rng.integers(4))
        if flip < 3:
            t = cv2.flip(t, flip - 1)               # -1 both, 0 vertical, 1 horizontal; 3 = as is
        if on_paper:
            return cv2.subtract(255, cv2.multiply(cv2.subtract(255, out), t, scale=1 / 128))
        return cv2.multiply(out, t, scale=1 / 128)

    def char_jitter(self, idx: np.ndarray, frac: float, n: int, rng) -> np.ndarray:
        if frac <= 0 or n < 3:
            return idx
        mask = rng.random(idx.shape) < frac
        if not self.include_blank:
            mask &= idx > 0
        k = int(mask.sum())
        if k:
            idx = idx.copy()
            idx[mask] = rng.integers(1, n, k)
        return idx

    # ---- the background video ------------------------------------------------------------------------
    def background(self, row: dict) -> np.ndarray:
        """The background video's frame row['bg'], at row['bg_gain'] brightness."""
        W, H, b = self.W, self.H, self.cfg["background"]
        plain = b["style"] == "plain"
        if plain:
            cols, rows = W, H
        else:
            cols, rows, _, _ = af.grid(W, H, int(b["cols"]))
        if self._reader is None:                     # enough pixels for the finest use of them
            self._reader = Reader(self.bgv, W / H, W if plain else max(256, cols * 4))
        src = self._reader.frame(int(row["bg"]))
        g = float(row["bg_gain"])
        if plain:
            out = cv2.resize(src, (W, H), interpolation=cv2.INTER_AREA if src.shape[1] > W else cv2.INTER_LINEAR)
            lv = self.bgv.get("levels")
            lo, k = (lv[0], 255 / (lv[1] - lv[0])) if lv else (0.0, 1.0)
            out = np.clip((out.astype(np.float32) - lo) * (k * g), 0, 255).astype(np.uint8)
            fill = (0, 0, 0)
        else:
            c = cv2.resize(src, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255)
            lv = self.bgv.get("levels")
            if lv:                                   # the clip's own auto-level (one stretch for all of it)
                c = np.clip((c - lv[0] / 255) * (255 / (lv[1] - lv[0])), 0, 1)
            if self.saturation != 1.0:
                lum = (c @ (af.LUMA * 255.0))[..., None]
                c = np.clip(lum + (c - lum) * self.saturation, 0, 1)
            if self.black_point > 0:
                c = np.clip((c - self.black_point) * (1 / (1 - self.black_point)), 0, 1)
            cells = (np.clip(c * g, 0, 1) * 255 + 0.5).astype(np.uint8)
            fs = af.FrameStyle(cols=int(b["cols"]), mix=float(np.clip(b["color"], 0, 1)),
                               chars=self.bg_ramp or self.ramps[row["band"]], tint=self.tint, bg=self.bg,
                               contrast=self.contrast, invert=False)
            out = af.draw_glyphs(af.cell_glyphs(cells, fs), cells, fs, W, H, self.font)
            fill = fs.bg
        if row["grain"] > 0:
            out = self.grain(out, np.random.default_rng([self.seed, 25, row["frame"]]))
        return shift(out, int(round(row["jx"] * W)), int(round(row["jy"] * H)), fill)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # ---- the frame -----------------------------------------------------------------------------------
    def render(self, row: dict) -> np.ndarray:
        W, H = self.W, self.H
        if row["image"] is None:
            if self.bgv and row.get("bg", -1) >= 0:
                out = self.background(row)
            else:
                out = np.zeros((H, W, 3), np.uint8)              # black leader
        else:
            rng = np.random.default_rng([self.seed, 22, row["frame"]])
            cols, rows, _, _ = af.grid(W, H, row["cols"])
            c = self.cells(row["image"], row["crop"], cols, rows)
            if row["layer"]:
                c2 = self.cells(row["layer"], row["layer_crop"], cols, rows)
                a = row["layer_alpha"]
                c = 1 - (1 - c) * (1 - a * c2) if self.blend == "screen" else (1 - a) * c + a * c2
            if row["hue"]:
                c = c @ hue_matrix(row["hue"]).T
            if self.saturation != 1.0:               # push colours away from grey, keeping each cell's brightness
                lum = (c @ (af.LUMA * 255.0))[..., None]   # af.LUMA is for 0-255 values; c is 0-1
                c = lum + (c - lum) * self.saturation
            c = np.clip(c, 0, 1)
            if self.black_point > 0:                 # crush the shadows so dark ground stays black leader
                c = np.clip((c - self.black_point) * (1 / (1 - self.black_point)), 0, 1)
            if row["expo"] != 1:
                c = 1 - (1 - c) ** row["expo"]
            c *= row["gain"]
            cells = (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)
            fs = af.FrameStyle(cols=row["cols"], mix=row["mix"], chars=self.ramps[row["band"]], tint=self.tint,
                               bg=self.bg, contrast=self.contrast, invert=False)
            if row["flash"]:
                fs = dataclasses.replace(fs, mix=0.0, tint=self.ink, bg=self.paper)
            idx = self.char_jitter(af.cell_glyphs(cells, fs), row["cjit"], len(fs.chars), rng)
            out = af.draw_glyphs(idx, cells, fs, W, H, self.font)
            if row["grain"] > 0:
                out = self.grain(out, rng, on_paper=row["flash"])
            out = shift(out, int(round(row["jx"] * W)), int(round(row["jy"] * H)), fs.bg)
        if self.overlay is not None:
            out = self.overlay.draw(np.ascontiguousarray(out), row)
        return out
