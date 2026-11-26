"""Step 2: the mapping rules. Analysis + config -> one decision per frame (the "score").

Everything random is drawn here, in order, from generators seeded by `seed`, so the score (and therefore
the film) is the same every time, and a --preview shows exactly what the full render will show there.
Each row of the score is a plain dict; renders/<name>.edl.csv is the same table, one line per frame.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import ascii_film as af

POOLS = ("bright", "texture", "dark")


# --------------------------------------------------------------------------- images

def gather_pools(root: Path, cfg: dict, cache_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """{pool: [(display name, readable path)]}. HEIC/HEIF is converted once to a cached JPEG."""
    base = root / cfg["paths"]["images"]
    pools = {}
    for name in POOLS:
        d = base / name
        files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")
                       and p.suffix.lower() in af.IMAGE_EXTS) if d.is_dir() else []
        pools[name] = [(f"{name}/{p.name}", str(af.readable_image(p, cache_dir))) for p in files]
    if not any(pools.values()):
        raise SystemExit(f"No images found in {base}/bright, {base}/dark or {base}/texture")
    return pools


# --------------------------------------------------------------------------- helpers

def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(x, sigma, mode="nearest") if sigma and sigma > 0 else x


def wsmooth(x: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Loudness-weighted smoothing: silent frames (whose centroid/flatness mean nothing) don't pull the
    curve around; a hold after a click in silence keeps the click's character."""
    if not sigma or sigma <= 0:
        return x
    num, den = smooth(x * weight, sigma), smooth(weight, sigma)
    ok = den > 1e-6 * max(float(weight.max()), 1e-12)
    return np.where(ok, num / np.where(ok, den, 1.0), x)


def slew(x: np.ndarray, max_step: float) -> np.ndarray:
    """Limit how fast a curve may change from one frame to the next."""
    if not max_step:
        return x
    y = np.array(x, np.float64)
    for i in range(1, len(y)):
        y[i] = min(max(y[i], y[i - 1] - max_step), y[i - 1] + max_step)
    return y


def lerp(a, b, t):
    return a + (b - a) * t


def onset_density(onset_frames: np.ndarray, n: int, window: int, fps: float) -> np.ndarray:
    """Onsets per second in a `window`-frame window centred on each frame."""
    marks = np.zeros(n)
    marks[onset_frames] = 1.0
    counts = np.convolve(marks, np.ones(max(1, window)), mode="same")
    return counts * fps / max(1, window)


# --------------------------------------------------------------------------- the score

@dataclass
class Score:
    rows: list[dict]           # one per frame, see build_score
    shots: list[dict]          # one per cut
    bands: list[dict]          # ascii.charset.bands, as configured
    warnings: list[str]


class _Picker:
    """Random image from a pool, never one used in the last `n` cuts (falls back to least recently used)."""

    def __init__(self, pools, n, rng):
        self.pools, self.rng = pools, rng
        self.recent = deque(maxlen=max(0, n)) if n else deque(maxlen=0)
        self.warned = set()

    def pool_for(self, want: str) -> str:
        order = {"bright": ("bright", "texture", "dark"), "texture": ("texture", "bright", "dark"),
                 "dark": ("dark", "texture", "bright")}[want]
        return next(p for p in order if self.pools[p])

    def pick(self, pool: str, exclude=(), remember=True, rng=None):
        items = self.pools[pool]
        cands = [it for it in items if it[1] not in self.recent and it[1] not in exclude]
        if not cands:
            self.warned.add(pool)
            seen = list(self.recent)
            cands = sorted((it for it in items if it[1] not in exclude) or items,
                           key=lambda it: seen.index(it[1]) if it[1] in seen else -1)[:1]
        it = cands[int((rng or self.rng).integers(len(cands)))]
        if remember:
            self.recent.append(it[1])
        return it


def build_score(an, cfg: dict, pools: dict, seed: int, views: dict | None = None) -> Score:
    """views: {photo path: hand crop} from images/crops.json (photos.views_for)."""
    views = views or {}
    fps, n = an.fps, an.n
    ed, asc, tr = cfg["editing"], cfg["ascii"], cfg["treatments"]
    on = lambda name: bool(tr[name]["enabled"])
    rms, rms_db, energy = an.f["rms_norm"], an.f["rms_db"], an.f["rms"]
    cent, flat = an.f["centroid"], an.f["flatness"]

    # ---- per-frame curves straight from the sound ------------------------------------------------
    # Analysis values are centred on each frame's start (k/fps), which is right for onsets. Frame k is on
    # screen from k/fps to (k+1)/fps, so for what you *hear* during it, look half a frame ahead too.
    ahead = lambda x: np.r_[x[1:], x[-1:]]
    silent = np.maximum(rms_db, ahead(rms_db)) < ed["silence"]["threshold_db"]
    heard = 0.5 * (rms + ahead(rms))

    res = asc["resolution"]
    level = np.clip(smooth(heard, res["smooth_frames"]), 0, 1) ** res["curve"]
    cols = np.round(slew(lerp(res["cols_quiet"], res["cols_loud"], level), res["max_step"])).astype(int)
    cols = np.maximum(cols, 2)

    cs = asc["charset"]
    bands = sorted(cs["bands"], key=lambda b: -b["above"])
    cent_s = wsmooth(cent, energy, cs["smooth_frames"])
    band_of = lambda c: next((i for i, b in enumerate(bands) if c >= b["above"]), len(bands) - 1)
    band = np.array([band_of(c) for c in cent_s])

    col = asc["color"]
    tonal = (1.0 - np.clip(wsmooth(flat, energy, col["smooth_frames"]), 0, 1)) ** col["curve"]
    mix = lerp(col["noisy_mix"], col["tonal_mix"], tonal)

    ex = tr["exposure"]
    loudness = smooth(heard, 0.7)

    def expo_of(lvl):
        """Exposure exponent g for 1 - (1 - c)^g: g < 1 darkens, g > 1 pushes toward white (4x range at 1)."""
        return 4.0 ** (ex["intensity"] * (2 * lvl - 1)) if on("exposure") else 1.0

    cj = tr["char_jitter"]
    follow = {"none": np.ones(n), "rms": rms, "flatness": flat, "centroid": cent}.get(cj.get("follow", "none"))
    if follow is None:
        raise SystemExit(f"config: treatments.char_jitter.follow must be none, rms, flatness or centroid")
    cjit = np.clip(cj["intensity"] * follow, 0, 1) if on("char_jitter") else np.zeros(n)

    rng_frames = np.random.default_rng([seed, 21])
    jit = rng_frames.uniform(-1, 1, (n, 2)) * tr["frame_jitter"]["intensity"] * 0.015
    if not on("frame_jitter"):
        jit[:] = 0.0
    jit[:, 1] *= 16 / 9                       # same pixel size vertically as horizontally on a 16:9 frame

    ly = tr["layering"]
    density = onset_density(an.onset_frames, n, ly["window_frames"], fps)

    # ---- the edit ----------------------------------------------------------------------------------
    rng_pick = np.random.default_rng([seed, 11])
    rng_crop = np.random.default_rng([seed, 12])
    rng_hue = np.random.default_rng([seed, 13])
    rng_layer = np.random.default_rng([seed, 14])
    picker = _Picker(pools, ed["no_repeat_within"], rng_pick)

    hold_cfg, btw, pl, fl = ed["hold"], ed["between_onsets"], ed["pools"], ed["flash"]
    onset_at = {int(f): float(s) for f, s in zip(an.onset_frames, an.strength)}
    next_onset = np.full(n, n)
    nxt = n
    for k in range(n - 1, -1, -1):
        next_onset[k] = nxt
        if k in onset_at:
            nxt = k

    exact = bool(tr["crop_zoom"].get("respect_hand_crops", True))

    def crop(rng=rng_crop, path=None):
        z = 1.0 + rng.random() * 2 * tr["crop_zoom"]["intensity"]
        cx, cy = rng.uniform(0.5 / z, 1 - 0.5 / z), rng.uniform(0.5 / z, 1 - 0.5 / z)
        if not on("crop_zoom") or (exact and path in views):   # (drawn anyway, so other shots don't change)
            return (0.5, 0.5, 1.0)
        return (round(cx, 4), round(cy, 4), round(z, 4))

    rows, shots = [], []
    shot, since, vis = None, 0, 1.0
    for k in range(n):
        reason = ""
        if k in onset_at:
            s = onset_at[k]
            win = slice(k, k + max(1, pl["centroid_window"]))
            c = float(np.average(cent[win], weights=energy[win])) if energy[win].sum() > 0 else float(cent[k])
            want = "bright" if c >= pl["bright_above"] else ("dark" if c < pl["dark_below"] else "texture")
            pool = picker.pool_for(want)
            name, path = picker.pick(pool)
            layer = None
            if on("layering") and density[k] >= ly["onsets_per_sec"]:
                lname, lpath = picker.pick(pool, exclude={path}, remember=False, rng=rng_layer)
                layer = dict(name=lname, path=lpath, crop=crop(rng_layer, lpath), alpha=float(ly["intensity"]))
            hue = float(rng_hue.uniform(-1, 1) * tr["color_shift"]["intensity"] * 180) if on("color_shift") else 0.0
            shot = dict(id=len(shots), frame=k, pool=pool, wanted=want, name=name, path=path, strength=s,
                        loud=float(max(rms[k:k + 2])),     # the cut and its hold are exposed for the onset itself
                        hold=int(round(lerp(hold_cfg["min_frames"], hold_cfg["max_frames"],
                                            s ** hold_cfg["curve"]))),
                        gap=int(next_onset[k] - k), crop=crop(path=path), hue=round(hue, 2), layer=layer,
                        flash=bool(fl["enabled"] and s >= fl["strength_above"]), band=int(band[k]),
                        centroid=round(c, 4))
            shots.append(shot)
            since, vis, kind = 0, 1.0, "cut"
            reason = f"onset s={s:.2f} -> hold {shot['hold']}f, gap {shot['gap']}f"
            if want != pool:
                reason += f" ({want}/ empty)"
        elif shot is None:
            kind = "silence" if silent[k] else "black"
            reason = "silence" if silent[k] else "no onset yet / faded out"
        else:
            since += 1
            in_hold = since < shot["hold"]
            if silent[k] and not (in_hold and not ed["silence"]["beats_hold"]):
                shot, kind, reason = None, "silence", "silence cuts the shot"
            elif in_hold:
                kind, reason = "hold", f"hold {since + 1}/{shot['hold']}"
            elif shot["gap"] <= btw["hold_if_gap_at_most"]:
                kind, reason = "hold", f"next onset in {shot['gap'] - since}f (gap {shot['gap']} <= "\
                                       f"{btw['hold_if_gap_at_most']})"
            else:
                vis *= btw["decay"]
                if vis < btw["black_below"]:
                    shot, kind, reason = None, "black", "decayed to black"
                else:
                    kind, reason = "decay", f"decay x{btw['decay']} -> {vis:.2f}"

        live = shot is not None and kind in ("cut", "hold", "decay")
        lvl = float(loudness[k])
        if live and since < shot["hold"]:           # a cut is never under-exposed: quiet onsets still show
            lvl = max(lvl, shot["loud"], float(ex.get("cut_floor", 0.0)))
        rows.append(dict(
            frame=k, time=round(k / fps, 4), kind=kind, reason=reason, onset=k in onset_at,
            strength=round(onset_at.get(k, 0.0), 3),
            shot=shot["id"] if live else -1,
            image=shot["path"] if live else None, name=shot["name"] if live else "",
            pool=shot["pool"] if live else "",
            crop=shot["crop"] if live else None, hue=shot["hue"] if live else 0.0,
            view=views.get(shot["path"]) if live else None,
            layer=shot["layer"]["path"] if live and shot["layer"] else None,
            layer_name=shot["layer"]["name"] if live and shot["layer"] else "",
            layer_crop=shot["layer"]["crop"] if live and shot["layer"] else None,
            layer_alpha=shot["layer"]["alpha"] if live and shot["layer"] else 0.0,
            flash=bool(live and kind == "cut" and shot["flash"]),
            hold=shot["hold"] if live else 0, since=since if live else 0,
            gain=round(vis, 4) if live else 0.0,
            cols=int(cols[k]),
            band=int(shot["band"] if live and cs["switch_on"] == "cut" else band[k]),
            mix=round(float(mix[k]), 3), expo=round(float(expo_of(lvl)), 4), cjit=round(float(cjit[k]), 4),
            jx=round(float(jit[k, 0]), 5), jy=round(float(jit[k, 1]), 5),
            grain=float(tr["grain"]["intensity"]) if on("grain") else 0.0,
            rms=round(float(rms[k]), 3), rms_db=round(float(rms_db[k]), 1),
            centroid=round(float(cent[k]), 3), centroid_hz=round(float(an.f["centroid_hz"][k])),
            flatness=round(float(flat[k]), 3), density=round(float(density[k]), 2),
        ))

    warnings = [f"images/{p}/ has too few images for no_repeat_within={ed['no_repeat_within']}: "
                f"some images repeat sooner" for p in sorted(picker.warned)]
    return Score(rows, shots, bands, warnings)


EDL_COLUMNS = ["frame", "time", "kind", "onset", "strength", "shot", "pool", "name", "view", "crop", "layer_name",
               "layer_alpha",
               "flash", "hold", "gain", "cols", "band", "mix", "expo", "cjit", "rms", "rms_db", "centroid",
               "centroid_hz", "flatness", "density", "reason"]


def _view_str(v: dict | None) -> str:
    """'x y w h' or 'x y w h rot' (rot in degrees clockwise); '- - - - rot' for a rotation without a crop."""
    if not v:
        return ""
    rect = " ".join(f"{v[k]:g}" for k in "xywh") if "w" in v else "- - - -"
    return rect + (f" {v['rot']:g}" if v.get("rot") else "")


def write_edl(rows: list[dict], bands: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EDL_COLUMNS)
        for r in rows:
            vals = dict(r, band=bands[r["band"]]["name"], onset=int(r["onset"]), flash=int(r["flash"]),
                        crop=" ".join(f"{x:g}" for x in r["crop"]) if r["crop"] else "",
                        view=_view_str(r.get("view")))
            w.writerow([vals[c] for c in EDL_COLUMNS])


def read_edl(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- how much each photo is used

def pool_usage(sc: Score, pools: dict, fps: float) -> dict:
    """How the edit uses the photos: per pool (photos, cuts, uses per photo, the closest repeat) and per photo
    (cuts, frames on screen, times layered in). Photo names are '<pool>/<file>', as in the EDL."""
    per: dict[str, dict] = {}

    def entry(name):
        return per.setdefault(name, {"cuts": 0, "frames": 0, "layered": 0, "at": []})
    for r in sc.rows:
        if r["kind"] not in ("cut", "hold", "decay") or not r["name"]:
            continue
        e = entry(r["name"])
        e["frames"] += 1
        if r["kind"] == "cut":
            e["cuts"] += 1
            e["at"].append(r["frame"])
            if r["layer_name"]:
                entry(r["layer_name"])["layered"] += 1
    out = {}
    for pool in POOLS:
        names = [n for n, _ in pools.get(pool, [])]
        uses = [per.get(n, {}).get("cuts", 0) for n in names]
        gaps = [(b - a, n) for n in names for a, b in zip(per.get(n, {}).get("at", []), per.get(n, {}).get("at", [])[1:])]
        top = max(names, key=lambda n: per.get(n, {}).get("cuts", 0)) if names else None
        cuts = sum(uses)
        out[pool] = dict(
            photos=len(names), cuts=cuts, wanted=sum(s["wanted"] == pool for s in sc.shots),
            used=sum(u > 0 for u in uses), unused=sum(u == 0 for u in uses),
            per_photo=round(cuts / len(names), 2) if names else None,
            most=max(uses) if uses else 0, most_name=top if uses and max(uses) else None,
            seconds=round(sum(per.get(n, {}).get("frames", 0) for n in names) / fps, 1),
            closest_repeat_s=round(min(gaps)[0] / fps, 2) if gaps else None,
            closest_repeat_name=min(gaps)[1] if gaps else None)
    photos = {n: {"cuts": e["cuts"], "seconds": round(e["frames"] / fps, 2), "layered": e["layered"],
                  "first_s": round(e["at"][0] / fps, 2) if e["at"] else None} for n, e in per.items()}
    return {"pools": out, "photos": photos, "cuts": len(sc.shots)}
