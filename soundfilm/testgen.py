"""Synthetic test assets, so every rule can be exercised before the real track and scans exist.

  audio/track.wav          20 s, 48 kHz: silence, isolated clicks weak -> strong, glassy high pings, heavy low
                           tones (one sustained), noise bursts, a dense glitch passage, a crescendo, silent gaps
  audio/track_events.json  the exact start time of every attack in it (ground truth for ./film verify)
  images/bright|dark|texture/  30 procedural images each (jpg, png and 2 heic per pool)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

SR = 48000
DUR = 20.0


# --------------------------------------------------------------------------- sounds

def _env(n, attack_s, decay_s):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack_s, 1e-5), 0, 1)
    return a * np.exp(-t / max(decay_s, 1e-5))


def click(rng, amp):
    n = int(0.004 * SR)
    sig = rng.uniform(-1, 1, n) * _env(n, 0.00005, 0.0006)
    sig[0] = 1.0
    return amp * sig / np.abs(sig).max()


def ping(freq, amp, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t) * _env(n, 0.0008, dur / 4)


def tone(freq, amp, dur, attack=0.004, release=0.06):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * freq * k * t) / k ** 1.5 for k in (1, 2, 3))
    env = np.clip(t / attack, 0, 1) * np.clip((dur - t) / release, 0, 1)
    return amp * sig / 1.6 * env


BANDS = {"white": None, "high": (3000, 14000), "mid": (500, 2000), "low": (40, 300)}


def noise_burst(rng, amp, dur, colour):
    """Noise with a 0.8 ms attack: white, or band-passed high / mid / low."""
    from scipy.signal import butter, sosfilt
    n = int(dur * SR)
    x = rng.standard_normal(n + 4000)
    if BANDS[colour]:
        x = sosfilt(butter(4, BANDS[colour], btype="bandpass", fs=SR, output="sos"), x)
    x = x[4000:]                                               # skip the filter's warm-up
    x /= max(np.abs(x).max(), 1e-9)
    env = np.clip(np.arange(n) / (0.0008 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.004 * SR), 0, 1)
    return amp * x * env


def mix(*sigs):
    """Sum signals of different lengths."""
    out = np.zeros(max(len(x) for x in sigs))
    for x in sigs:
        out[:len(x)] += x
    return out


def make_track(rng) -> tuple[np.ndarray, list[dict]]:
    y = np.zeros(int(DUR * SR))
    events: list[dict] = []

    def put(t, sig, kind=None, amp=None):
        i = int(round(t * SR))
        seg = sig[:len(y) - i]
        y[i:i + len(seg)] += seg
        if kind:
            events.append(dict(time=round(i / SR, 6), kind=kind, amp=round(float(amp), 3)))

    # 0.0-1.5 s: digital silence (black leader)
    # 1.5-4.5 s: isolated clicks in silence, weak -> strong (holds 1 -> 4 frames, decays, the last one flashes)
    for t, a in zip((1.62, 2.11, 2.57, 3.06, 3.52, 4.03), (0.04, 0.09, 0.18, 0.35, 0.6, 0.95)):
        put(t, click(rng, a), "click", a)

    # 4.5-7.5 s: glassy high pings over a faint high shimmer (tonal, bright -> colour, bright pool, sparse glyphs)
    tt = np.arange(int(3.0 * SR)) / SR
    put(4.5, 0.006 * (np.sin(2 * np.pi * 7300 * tt) + np.sin(2 * np.pi * 9100 * tt)) * np.clip(tt / 0.2, 0, 1)
        * np.clip((3.0 - tt) / 0.2, 0, 1))
    t = 4.57
    while t < 7.35:
        a = rng.uniform(0.15, 0.45)
        put(t, ping(rng.uniform(4200, 9000), a, rng.uniform(0.15, 0.35)), "ping", a)
        t += rng.uniform(0.22, 0.42)

    # 7.5-10.5 s: heavy low tones (dark pool, dense glyphs / blocks); the last one is held with no onset after it
    for t, f, a, d in ((7.58, 55, 0.8, 0.45), (8.21, 82, 0.7, 0.4), (8.69, 110, 0.75, 0.5), (9.27, 65, 0.8, 0.45),
                       (9.83, 147, 0.6, 0.62)):
        put(t, tone(f, a, d), "tone", a)

    # 10.5-11.0 s: silence. 11.0-13.5 s: noise bursts (noisy -> monochrome), high / mid / low bands -> all 3 pools
    t = 11.03
    while t < 13.4:
        a, d = rng.uniform(0.2, 0.7), rng.uniform(0.02, 0.12)
        put(t, noise_burst(rng, a, d, rng.choice(["high", "mid", "low"])), "noise", a)
        t += d + rng.uniform(0.1, 0.3)

    # 13.5-16.5 s: dense glitch, 10-18 events a second (layering, holds through short gaps, character jitter)
    put(13.5, 0.004 * rng.standard_normal(int(3.0 * SR)))
    t = 13.52
    while t < 16.4:
        kind = str(rng.choice(["click", "grain", "thump", "ping"], p=[0.3, 0.35, 0.2, 0.15]))
        a = rng.uniform(0.2, 0.65)
        sig = {"click": lambda: click(rng, a),
               "grain": lambda: noise_burst(rng, a, rng.uniform(0.008, 0.02), "mid"),
               "thump": lambda: tone(rng.uniform(70, 160), a, 0.05, attack=0.002, release=0.02),
               "ping": lambda: ping(rng.uniform(2500, 7000), a, 0.06)}[kind]()
        put(t, sig, kind, a)
        t += rng.uniform(0.055, 0.1)

    # 16.5-17.3 s: silence. 17.3-19.6 s: crescendo, a swelling bed with accelerating hits; the last hit flashes
    n = int(2.3 * SR)
    tt = np.arange(n) / SR
    bed = (0.5 * np.convolve(rng.standard_normal(n), np.ones(6) / 6, mode="same")
           + sum(np.sin(2 * np.pi * f * tt) for f in (220, 330, 440, 2500)) / 4)
    swell = 0.004 * (0.45 / 0.004) ** (tt / 2.3)
    gap = np.clip(np.abs(tt - 2.10) / 0.02 - 1, 0, 1)            # a 40 ms hole in the swell right before the last hit
    put(17.3, bed * swell * gap * np.clip((2.3 - tt) / 0.05, 0, 1))
    t, gap = 17.42, 0.42
    while t < 19.3:
        a = 0.25 + 0.5 * (t - 17.3) / 2.3
        put(t, mix(click(rng, a), noise_burst(rng, a * 0.7, 0.03, "white")), "hit", a)
        t += gap
        gap = max(0.12, gap * 0.8)
    put(19.42, mix(click(rng, 1.0), noise_burst(rng, 0.9, 0.12, "white"), tone(70, 0.6, 0.16)), "hit", 1.0)
    # 19.6-20.0 s: tail into silence
    y = np.clip(y, -1, 1) * 0.95 / max(np.abs(y).max(), 1e-9)
    return y, sorted(events, key=lambda e: e["time"])


def write_wav(path: Path, y: np.ndarray):
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.repeat(pcm, 2).tobytes())


# --------------------------------------------------------------------------- images

def _noise(rng, h, w, cell):
    small = rng.random((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm(rng, h, w, base=160, octaves=5):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for i in range(octaves):
        out += amp * _noise(rng, h, w, max(2, int(base / 2 ** i)))
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / max(np.ptp(out), 1e-6)


def _colourise(v, c0, c1):
    c0, c1 = np.array(c0, np.float32), np.array(c1, np.float32)
    return v[..., None] * (c1 - c0) + c0


def bright_image(rng, h, w):
    """Pale ground, drifting light particles, dark dust specks, a few frost-like crystal lines."""
    grounds = [(232, 226, 205), (218, 232, 238), (222, 234, 204), (238, 224, 220), (245, 240, 228)]
    g = np.array(grounds[rng.integers(len(grounds))], np.float32)
    img = g * (0.82 + 0.18 * fbm(rng, h, w, 300))[..., None]
    light = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(250, 900))):
        x, y, r = int(rng.integers(w)), int(rng.integers(h)), float(rng.uniform(1, 7))
        cv2.circle(light, (x, y), int(r), float(rng.uniform(0.3, 1.0)), -1, cv2.LINE_AA)
    light = cv2.GaussianBlur(light, (0, 0), 2.5)
    specks = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(40, 200))):
        x, y = int(rng.integers(w)), int(rng.integers(h))
        cv2.circle(specks, (x, y), int(rng.integers(1, 4)), 1.0, -1, cv2.LINE_AA)
    frost = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 12))):
        x, y, ang = float(rng.integers(w)), float(rng.integers(h)), float(rng.uniform(0, np.pi))
        for _ in range(int(rng.integers(4, 10))):
            L = rng.uniform(20, 140)
            x2, y2 = x + L * np.cos(ang), y + L * np.sin(ang)
            cv2.line(frost, (int(x), int(y)), (int(x2), int(y2)), 1.0, 1, cv2.LINE_AA)
            x, y, ang = x2, y2, ang + rng.choice([-1, 1]) * np.pi / 3
    frost = cv2.GaussianBlur(frost, (0, 0), 0.8)
    img = img + 70 * light[..., None] + 50 * frost[..., None] - 140 * specks[..., None] * np.array([0.9, 1, 1.1])
    return np.clip(img, 0, 255).astype(np.uint8)


def dark_image(rng, h, w):
    """Near-black ground with a few soft organic blobs (seeds, cells, spores), rim-lit, in moss/umber/amber."""
    palette = [(40, 72, 30), (78, 48, 24), (120, 78, 22), (66, 22, 34), (22, 56, 58), (96, 90, 40)]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 9))):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        r = rng.uniform(0.05, 0.16) * w
        sx, sy = rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.5)
        field += r * r / (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2 + 1)
    field *= 0.7 + 0.6 * fbm(rng, h, w, 120)
    mask = np.clip((field - 0.9) / 0.6, 0, 1)
    inner = fbm(rng, h, w, 40)
    c0, c1 = palette[rng.integers(len(palette))], palette[rng.integers(len(palette))]
    body = _colourise(inner, np.array(c0) * 0.5, np.array(c1) * 1.3)
    rim = np.clip(cv2.Laplacian(cv2.GaussianBlur(mask, (0, 0), 3), cv2.CV_32F) * -40, 0, 1)
    ground = 6 + 14 * fbm(rng, h, w, 260)
    vign = 1 - 0.6 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2.2)
    img = ground[..., None] * (1 - mask[..., None]) + body * mask[..., None] + 120 * rim[..., None]
    return np.clip(img * vign[..., None], 0, 255).astype(np.uint8)


def texture_image(rng, h, w):
    """Fibrous mid-tones: directional fibres, warped wood grain and a few veins, in bark/leaf/ochre colours."""
    ang = float(rng.uniform(0, 180))
    big = int(np.hypot(h, w)) + 4
    noise = rng.random((big, big)).astype(np.float32)
    k = int(rng.integers(40, 160))
    fib = cv2.blur(noise, (k, 1))
    M = cv2.getRotationMatrix2D((big / 2, big / 2), ang, 1.0)
    fib = cv2.warpAffine(fib, M, (big, big))[(big - h) // 2:(big - h) // 2 + h, (big - w) // 2:(big - w) // 2 + w]
    fib = (fib - fib.min()) / max(np.ptp(fib), 1e-6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = np.deg2rad(ang)
    grain = 0.5 + 0.5 * np.sin((xx * np.cos(t) + yy * np.sin(t)) * rng.uniform(0.03, 0.12)
                               + 9 * fbm(rng, h, w, 200))
    v = np.clip(0.55 * fib + 0.45 * grain, 0, 1)
    veins = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 8))):
        x, y = float(rng.integers(w)), float(rng.integers(h))
        a = rng.uniform(0, 2 * np.pi)
        pts = []
        for _ in range(60):
            pts.append((int(x), int(y)))
            a += rng.normal(0, 0.15)
            x, y = x + 14 * np.cos(a), y + 14 * np.sin(a)
        cv2.polylines(veins, [np.array(pts, np.int32)], False, 1.0, int(rng.integers(1, 4)), cv2.LINE_AA)
    v = np.clip(v - 0.35 * cv2.GaussianBlur(veins, (0, 0), 1.2), 0, 1)
    palettes = [((52, 34, 20), (172, 132, 88)), ((34, 50, 22), (150, 170, 90)), ((60, 40, 30), (205, 160, 90)),
                ((30, 30, 28), (160, 150, 130))]
    c0, c1 = palettes[rng.integers(len(palettes))]
    return np.clip(_colourise(v, c0, c1), 0, 255).astype(np.uint8)


MAKERS = {"bright": bright_image, "dark": dark_image, "texture": texture_image}


def make_images(images_dir: Path, rng, per_pool=30, size=(1600, 900)):
    w, h = size
    tmp = Path(tempfile.mkdtemp(prefix="soundfilm-heic-"))
    try:
        for pool, make in MAKERS.items():
            d = images_dir / pool
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_pool):
                img = cv2.cvtColor(make(rng, h, w), cv2.COLOR_RGB2BGR)
                name = f"{pool}_{i:02d}"
                if i >= per_pool - 2:                          # exercise the HEIC path
                    src = tmp / f"{name}.png"
                    cv2.imwrite(str(src), img)
                    subprocess.run(["sips", "-s", "format", "heic", str(src), "--out", str(d / f"{name}.heic")],
                                   check=True, capture_output=True)
                elif i % 5 == 4:
                    cv2.imwrite(str(d / f"{name}.png"), img)
                else:
                    cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def generate(root: Path, force: bool = False, seed: int = 3, log=print) -> None:
    audio, images = root / "audio", root / "images"
    existing = [p for p in (audio / "track.wav", images) if p.exists()]
    if existing and not force:
        raise SystemExit(f"Test assets already exist in {root} ({', '.join(p.name for p in existing)}). "
                         f"Pass --force to regenerate them (overwrites the files it made, deletes nothing else).")
    rng = np.random.default_rng(seed)
    audio.mkdir(parents=True, exist_ok=True)
    y, events = make_track(rng)
    write_wav(audio / "track.wav", y)
    (audio / "track_events.json").write_text(json.dumps({"sample_rate": SR, "duration": DUR, "events": events},
                                                        indent=1))
    log(f"wrote {audio / 'track.wav'} ({DUR:.0f} s, {len(events)} events)")
    make_images(images, rng)
    log(f"wrote {sum(1 for _ in images.rglob('*.*'))} images to {images}/bright, dark, texture")
