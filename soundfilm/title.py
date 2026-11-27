"""A title card before the film: your words in a corner of the frame over the background video (or black).

The card comes first, for `title.seconds`, in silence; then the film and the music start exactly as they
would without it. The background video plays under the card from `background.start` and carries straight on
into the film, so there's no jump back where the card ends. The text is plain, in the same monospace font as
the characters, steady (no gate weave or grain on it). Nothing changes until `title.text` has words in it.

A full render always starts with the card. A preview shows it only when it starts at 0 s.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ascii_film as af

CORNERS = ("top-right", "top-left", "bottom-right", "bottom-left", "centre")


def lines(cfg: dict) -> list[str]:
    """The text, line by line (trailing spaces dropped so right-aligned lines meet the margin)."""
    text = str(cfg["title"].get("text") or "").replace("\r\n", "\n").replace("\\n", "\n")
    out = [ln.rstrip() for ln in text.split("\n")]
    while out and not out[-1]:
        out.pop()
    while out and not out[0]:
        out.pop(0)
    return out


def frames(cfg: dict) -> int:
    """How many frames the card lasts (0: no card)."""
    t = cfg["title"]
    if not t["enabled"] or not lines(cfg):
        return 0
    return max(1, int(round(float(t["seconds"]) * cfg["video"]["fps"])))


def layer(cfg: dict, W: int, H: int, font_path: str) -> tuple[np.ndarray, np.ndarray]:
    """The text for a W x H frame: (colour (3,), coverage (H, W, 1) float32 0-1)."""
    t = cfg["title"]
    size = max(6, int(round(float(t["size"]) * H)))
    margin = int(round(float(t["margin"]) * H))
    font = ImageFont.truetype(font_path, size)
    text = "\n".join(lines(cfg))
    corner = t["corner"]
    align = "right" if corner.endswith("right") else ("center" if corner == "centre" else "left")
    spacing = int(round(size * 0.45))
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    x0, y0, x1, y1 = d.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=align)
    bw, bh = x1 - x0, y1 - y0
    x = {"left": margin, "right": W - margin - bw, "center": (W - bw) // 2}[align] - x0
    y = (H - margin - bh if corner.startswith("bottom") else (H - bh) // 2 if corner == "centre" else margin) - y0
    d.multiline_text((x, y), text, font=font, fill=255, spacing=spacing, align=align)
    color = af.parse_color(t["color"] or cfg["ascii"]["tint"])
    return np.asarray(color, np.float32), (np.asarray(mask, np.float32) * (1 / 255))[..., None]


def rows(cfg: dict, bg: dict | None, T: int, band: int) -> list[dict]:
    """The card's frames for the renderer: the background video from its start (or black), steady."""
    b = cfg["background"]
    level = float(np.clip(b["level"], 0, 1))
    loop = bg["frames"] - bg["start"] if bg else 0
    grain = float(cfg["treatments"]["grain"]["intensity"]) if cfg["treatments"]["grain"]["enabled"] else 0.0
    return [dict(title=True, kind="title", frame=i, image=None, band=band, grain=grain, jx=0.0, jy=0.0,
                 bg=bg["start"] + i % loop if bg else -1, bg_gain=level if bg else 0.0) for i in range(T)]
