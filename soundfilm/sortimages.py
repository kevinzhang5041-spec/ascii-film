"""./film sort: score photos and copy them into images/bright, images/texture and images/dark.

Each photo is judged by what the film will actually show of it: the 16:9 centre crop, auto-levelled the
way frames.py does it.

  texture  the most detailed third (mean gradient energy: leaves, gravel, bark, busy scenes)
  bright   the rest, upper half by brightness (median luminance: high-key, airy, light)
  dark     the rest, lower half (low-key, shadowy)

That mirrors the sound side: middle-of-the-spectrum noise -> texture, high glassy -> bright, low -> dark.

Sources are never modified. The copies are JPEGs resized to --max-side, with EXIF dropped (so no GPS
location travels with them). An existing file is never overwritten; a photo whose name is already in a
pool is skipped. Writes images/sort_report.csv and images_sorted.png (a contact sheet of the pools).
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import ascii_film as af

from .score import POOLS

VIEW = (384, 216)                  # scoring resolution of the 16:9 crop
ORDER = {"bright": -1, "texture": -1, "dark": 1}      # contact sheet: brightest / busiest / darkest first


def find_photos(sources: list[Path]) -> list[Path]:
    out = []
    for s in sources:
        s = Path(s).expanduser()
        files = [s] if s.is_file() else sorted(s.rglob("*")) if s.is_dir() else []
        out += [p for p in files if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in af.IMAGE_EXTS]
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def film_view(path: Path, cache: Path) -> np.ndarray:
    """RGB uint8 at VIEW size: the centre 16:9 crop, as the renderer frames it (before auto-level). A clip is
    judged by a still from its middle."""
    from .clips import still
    with Image.open(still(path, cache)) as im:
        im.draft("RGB", (VIEW[0] * 4, VIEW[0] * 4))
        arr = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    h, w = arr.shape[:2]
    t = VIEW[0] / VIEW[1]
    if w / h > t:
        cw = int(round(h * t))
        arr = arr[:, (w - cw) // 2:(w - cw) // 2 + cw]
    else:
        ch = int(round(w / t))
        arr = arr[(h - ch) // 2:(h - ch) // 2 + ch]
    return cv2.resize(arr, VIEW, interpolation=cv2.INTER_AREA)


def scores(rgb: np.ndarray) -> dict:
    lum = rgb.astype(np.float32) @ af.LUMA                       # 0-1
    lo, hi = np.percentile(lum, [0.5, 99.5])
    if hi - lo > 8 / 255:                                        # the renderer's auto_level
        lum = np.clip((lum - lo) / (hi - lo), 0, 1)
    lum = lum.astype(np.float32)                                 # (float64 percentiles promote it)
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    r, gg, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - gg, 0.5 * (r + gg) - b
    colour = (np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())) / 255
    return {"brightness": float(np.median(lum)), "mean": float(lum.mean()),
            "texture": float(np.hypot(gx, gy).mean()), "colour": float(colour)}


def assign(rows: list[dict], texture_share: float) -> None:
    n = len(rows)
    k = int(round(n * float(np.clip(texture_share, 0, 1))))
    by_tex = sorted(range(n), key=lambda i: -rows[i]["texture"])
    for i in by_tex[:k]:
        rows[i]["pool"] = "texture"
    rest = sorted(by_tex[k:], key=lambda i: rows[i]["brightness"])
    half = len(rest) // 2
    for j, i in enumerate(rest):
        rows[i]["pool"] = "dark" if j < half else "bright"


def contact_sheet(rows: list[dict], out: Path, thumb=(192, 108), cols=10) -> Path:
    try:
        font = ImageFont.truetype(af.find_font(), 26)
        small = ImageFont.truetype(af.find_font(), 17)
    except OSError:
        font = small = ImageFont.load_default()
    pad, head = 6, 56
    blocks = []
    for pool in POOLS:
        rs = sorted((r for r in rows if r["pool"] == pool),
                    key=lambda r: ORDER[pool] * (r["texture"] if pool == "texture" else r["brightness"]))
        nrow = max(1, -(-len(rs) // cols))
        blk = Image.new("RGB", (cols * (thumb[0] + pad) + pad, head + nrow * (thumb[1] + pad) + pad), (16, 18, 16))
        d = ImageDraw.Draw(blk)
        how = {"bright": "brightest first", "texture": "most detailed first", "dark": "darkest first"}[pool]
        d.text((pad + 4, 12), f"{pool}: {len(rs)} photos", font=font, fill=(235, 235, 230))
        d.text((pad + 330, 19), how, font=small, fill=(140, 145, 138))
        for j, r in enumerate(rs):
            x, y = pad + (j % cols) * (thumb[0] + pad), head + (j // cols) * (thumb[1] + pad)
            blk.paste(Image.fromarray(cv2.resize(r["_view"], thumb, interpolation=cv2.INTER_AREA)), (x, y))
        blocks.append(blk)
    sheet = Image.new("RGB", (blocks[0].width, sum(b.height for b in blocks)), (16, 18, 16))
    y = 0
    for b in blocks:
        sheet.paste(b, (0, y))
        y += b.height
    sheet.save(out)
    return out


def copy_resized(src: Path, dest: Path, cache: Path, max_side: int) -> None:
    """A JPEG copy, oriented, at most max_side pixels long, without EXIF. Refuses to overwrite."""
    with Image.open(af.readable_image(src, cache)) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        tmp = dest.with_name(f".sorting-{dest.name}")
        im.save(tmp, "JPEG", quality=90)
    try:
        os.link(tmp, dest)                                       # fails instead of overwriting
    finally:
        tmp.unlink(missing_ok=True)


def sort_photos(sources: list[Path], root: Path, cfg: dict, texture_share: float = 1 / 3, max_side: int = 2560,
                dry_run: bool = False, log=print) -> dict:
    root = Path(root)
    photos = find_photos(sources)
    if not photos:
        raise SystemExit(f"No photos found in {', '.join(map(str, sources))}")
    cache = root / ".work"
    log(f"scoring {len(photos)} photos ...")

    def one(p):
        try:
            v = film_view(p, cache)
            return dict(name=p.name, source=str(p), _view=v, **scores(v))
        except Exception as e:                                   # unreadable file: report it, keep going
            return dict(name=p.name, source=str(p), error=str(e))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, photos))
    bad = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    for r in bad:
        log(f"  skipped {r['name']}: {r['error'].strip()}")
    if not rows:
        raise SystemExit("None of the photos could be read.")
    assign(rows, texture_share)

    images = root / cfg["paths"]["images"]
    copied, skipped = 0, []
    for r in rows:
        dest = images / r["pool"] / (Path(r["name"]).stem + ".jpg")
        r["copy"] = str(dest.relative_to(root))
        if dry_run:
            continue
        if any((images / p / dest.name).exists() for p in POOLS):
            skipped.append(r["copy"])
            r["copy"] += " (already there, not copied)"
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        copy_resized(Path(r["source"]), dest, cache, max_side)
        copied += 1

    images.mkdir(parents=True, exist_ok=True)
    report = images / "sort_report.csv"
    with open(report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "copy", "brightness", "texture", "colour", "source"])
        for r in sorted(rows, key=lambda r: (POOLS.index(r["pool"]), r["name"])):
            w.writerow([r["pool"], r["copy"], f"{r['brightness']:.3f}", f"{r['texture']:.4f}", f"{r['colour']:.3f}",
                        r["source"]])
    sheet = contact_sheet(rows, root / "images_sorted.png")
    counts = {p: sum(r["pool"] == p for r in rows) for p in POOLS}
    log(f"sorted {len(rows)} photos: " + ", ".join(f"{p} {counts[p]}" for p in POOLS)
        + (f"; {len(bad)} unreadable" if bad else ""))
    if not dry_run:
        log(f"copied {copied} into {images}/" + (f"; {len(skipped)} already there, left alone" if skipped else ""))
    log(f"wrote {report} and {sheet}")
    return {"counts": counts, "copied": copied, "skipped": skipped, "unreadable": [r["name"] for r in bad],
            "report": report, "sheet": sheet}
