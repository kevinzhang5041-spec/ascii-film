"""--debug overlay: why each frame looks the way it does.

Top left: frame number, time, what the edit did and why, and the feature values -> what they chose.
Bottom strip: the sound around the playhead (RMS bars, centroid and flatness lines, onset ticks) and a lane
showing every frame's decision (cut / hold / decay / black), coloured by image pool.
"""
from __future__ import annotations

import cv2
import numpy as np

POOL_RGB = {"bright": (255, 210, 60), "texture": (110, 215, 120), "dark": (90, 150, 255), "": (60, 60, 60)}
KIND_LEVEL = {"cut": 1.0, "hold": 0.65, "decay": 0.35, "black": 0.0, "silence": 0.0}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def timeline(an, score) -> dict:
    """The compact per-frame data the overlay needs for the strip (sent to every render process)."""
    rows = score.rows
    return dict(
        rms=an.f["rms_norm"].astype(np.float32), centroid=an.f["centroid"].astype(np.float32),
        flatness=an.f["flatness"].astype(np.float32), env=np.clip(an.f["onset_env"], 0, 1).astype(np.float32),
        kind=[r["kind"] for r in rows], pool=[r["pool"] for r in rows],
        onset=np.array([r["onset"] for r in rows]), strength=np.array([r["strength"] for r in rows], np.float32),
        flash=np.array([r["flash"] for r in rows]),
        band_names=[b["name"] for b in score.bands],
    )


class DebugOverlay:
    def __init__(self, tl: dict, cfg: dict, W: int, H: int, fps: float):
        self.tl, self.W, self.H, self.fps = tl, W, H, fps
        self.s = H / 1080
        self.span = int(round(cfg["debug"]["timeline_seconds"] * fps))
        self.sil_db = cfg["editing"]["silence"]["threshold_db"]
        self.sil_norm = np.clip(1 + self.sil_db / cfg["analysis"]["rms_range_db"], 0, 1)
        p = cfg["editing"]["pools"]
        self.bright_above, self.dark_below = p["bright_above"], p["dark_below"]
        self.flash_above = cfg["editing"]["flash"]["strength_above"]

    def _text(self, img, txt, x, y, scale, color, thick=1):
        s = self.s
        cv2.putText(img, txt, (int(x), int(y)), FONT, scale * s, (0, 0, 0), max(1, int(round((thick + 2) * s))),
                    cv2.LINE_AA)
        cv2.putText(img, txt, (int(x), int(y)), FONT, scale * s, color, max(1, int(round(thick * s))), cv2.LINE_AA)

    def draw(self, img: np.ndarray, r: dict) -> np.ndarray:
        self._panel(img, r)
        self._strip(img, r["frame"])
        return img

    def _panel(self, img, r):
        s = self.s
        k = r["frame"]
        band = self.tl["band_names"][r["band"]]
        kind = r["kind"].upper() + (" + FLASH" if r["flash"] else "")
        kcol = {"CUT": (80, 255, 120), "HOLD": (230, 230, 230), "DECAY": (180, 180, 180)}.get(r["kind"].upper(),
                                                                                              (255, 90, 90))
        if r["flash"]:
            kcol = (255, 255, 0)
        if r["onset"]:
            l2 = f"ONSET s={r['strength']:.2f} -> hold {r['hold']}f" + (f"  >= {self.flash_above} FLASH"
                                                                        if r["flash"] else "")
        else:
            l2 = r["reason"]
        lines = [
            (f"F{k:05d}  {k / self.fps:7.3f}s  {kind}", 0.95, kcol, 2),
            (l2, 0.62, (255, 255, 255), 1),
            (f"RMS  {r['rms']:.2f} ({r['rms_db']:+.0f} dB, silence < {self.sil_db}) -> {r['cols']} cols, "
             f"expo {r['expo']:.2f}", 0.58, (230, 230, 230), 1),
            (f"CENT {r['centroid']:.2f} ({r['centroid_hz']} Hz) -> pool {r['pool'] or '-'}, chars {band}",
             0.58, POOL_RGB.get(r["pool"], (230, 230, 230)), 1),
            (f"FLAT {r['flatness']:.2f} -> colour mix {r['mix']:.2f}   char jitter {r['cjit']:.2f}",
             0.58, (230, 230, 230), 1),
            (f"IMG  {r['name'] or '-'}" + (f"  + {r['layer_name']} x{r['layer_alpha']:.2f} (dense "
                                            f"{r['density']:.0f}/s)" if r["layer_name"] else "")
             + (f"  gain {r['gain']:.2f}" if r["image"] else "")
             + (f"  background video frame {r['bg']} x{r['bg_gain']:.2f}" if r.get("bg", -1) >= 0 else "")
             + (f"   section {r['section']}" if r.get("section") else ""), 0.58, (230, 230, 230), 1),
        ]
        pad, lh = int(14 * s), int(34 * s)
        w = int(min(self.W - 2 * pad, 1180 * s))
        h = pad * 2 + lh * len(lines)
        roi = img[pad:pad + h, pad:pad + w]
        roi[:] = (roi.astype(np.uint16) * 36 >> 8).astype(np.uint8)          # darken 86%
        y = pad + int(28 * s)
        for txt, sc, col, th in lines:
            self._text(img, txt, pad * 2, y, sc, col, th)
            y += lh

    def _strip(self, img, k):
        s, W, H, tl, span = self.s, self.W, self.H, self.tl, self.span
        sh = int(170 * s)
        y0, y1 = H - sh, H
        roi = img[y0:y1]
        roi[:] = (roi.astype(np.uint16) * 36 >> 8).astype(np.uint8)
        n = len(tl["kind"])
        lo, hi = k - span, k + span
        px = lambda f: int(round((f - lo) / (hi - lo) * (W - 1)))
        bw = max(1, int(W / (2 * span + 1)))
        lane_h, top = int(22 * s), y0 + int(8 * s)
        plot_top, plot_bot = top + lane_h + int(8 * s), y1 - int(10 * s)
        ph = plot_bot - plot_top
        yv = lambda v: int(round(plot_bot - np.clip(v, 0, 1) * ph))
        fr = np.arange(max(0, lo), min(n, hi + 1))
        for f in fr:                                              # decision lane + RMS bars
            x = px(f)
            lvl = KIND_LEVEL[tl["kind"][f]]
            if lvl > 0:
                col = tuple(int(c * lvl) for c in POOL_RGB.get(tl["pool"][f], (200, 200, 200)))
                cv2.rectangle(img, (x - bw // 2, top), (x + bw // 2, top + lane_h), col, -1)
            cv2.rectangle(img, (x - bw // 2 + 1, yv(tl["rms"][f])), (x + bw // 2 - 1, plot_bot), (95, 95, 95), -1)
        cv2.line(img, (0, yv(self.sil_norm)), (W, yv(self.sil_norm)), (160, 60, 60), max(1, int(s)))
        for key, col in (("centroid", (80, 220, 255)), ("flatness", (230, 110, 230)), ("env", (200, 200, 120))):
            if len(fr) > 1:
                pts = np.array([[px(f), yv(tl[key][f])] for f in fr], np.int32)
                cv2.polylines(img, [pts], False, col, max(1, int(round(1.6 * s))), cv2.LINE_AA)
        for f in fr:                                              # onset ticks, height = strength
            if tl["onset"][f]:
                x = px(f)
                col = (255, 255, 255) if tl["flash"][f] else POOL_RGB.get(tl["pool"][f], (255, 255, 255))
                cv2.line(img, (x, plot_bot), (x, yv(0.15 + 0.85 * tl["strength"][f])), col,
                         max(2, int(round(3 * s))))
        xk = px(k)
        cv2.rectangle(img, (xk - bw // 2, y0), (xk + bw // 2, y1), (255, 60, 60), max(1, int(round(2 * s))))
        lx = int(12 * s)
        for i, (lab, col) in enumerate((("rms", (150, 150, 150)), ("centroid", (80, 220, 255)),
                                        ("flatness", (230, 110, 230)), ("onset env", (200, 200, 120)))):
            self._text(img, lab, lx + i * int(130 * s), plot_top + int(16 * s), 0.45, col)
