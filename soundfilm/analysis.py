"""Step 1: per-frame audio features at the film's frame rate, cached in analysis.json.

Per video frame k, from a window centred on its start time k/fps. That is the same half-frame either side
that "nearest frame" snapping gives to frame k, so an onset snapped to frame k is always inside frame k's
window (its centroid, loudness etc. describe the onset itself):
  rms_db / rms_norm   loudness, dB below the loudest frame, and 0-1 over `rms_range_db`
  centroid            spectral centroid ("brightness"), log-frequency, 0-1 over the track
  flatness            spectral flatness (0 = pure tone, 1 = noise), dB scale, 0-1 over the track
  onset_env           onset envelope, the strongest value within half a frame of k
Onsets: librosa peak-picking on a ~5 ms onset envelope, each snapped to the nearest frame.
Onset strength 0-1 = (1 - w) x spectral jump (envelope summed over +-onset_strength_ms) + w x loudness.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import ascii_film as af

VERSION = 3
AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".aac", ".ogg")


def find_track(root: Path, cfg: dict) -> Path:
    p = Path(cfg["paths"]["audio"])
    p = p if p.is_absolute() else root / p
    if p.suffix.lower() in AUDIO_EXTS:
        if p.exists():
            return p
    else:
        for ext in AUDIO_EXTS:
            for c in (p.with_name(p.name + ext), p.with_name(p.name + ext.upper())):
                if c.exists():
                    return c
    raise SystemExit(f"No soundtrack found. Put it at {p}.wav (or .mp3 / .m4a).")


def decode_audio(path, sr: int, channels: int = 1) -> np.ndarray:
    """Any audio (or the audio of a video) -> float32 samples at `sr`, via ffmpeg."""
    cmd = [af.tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", str(channels),
           "-ar", str(sr), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't decode {path}: {r.stderr.decode(errors='replace')[-500:]}")
    y = np.frombuffer(r.stdout, np.float32)
    return y.reshape(-1, channels).copy() if channels > 1 else y.copy()


def _norm(x: np.ndarray, mask: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ref = x[mask] if mask.any() else x
    a, b = np.percentile(ref, [lo, hi]) if ref.size else (0.0, 1.0)
    if b - a < 1e-9:
        return np.full_like(x, 0.5)
    return np.clip((x - a) / (b - a), 0.0, 1.0)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), x[-1] if len(x) else 0.0)])


def detect_onsets(y: np.ndarray, cfg: dict):
    """Onset times (s) and raw strengths. Shared by the analysis and by verify (on the muxed audio)."""
    import librosa
    a = cfg["analysis"]
    sr, hop = a["sample_rate"], a["onset_hop"]
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=a["onset_n_fft"])
    wait = max(1, int(round(a["onset_wait_ms"] / 1000 * sr / hop)))
    peaks = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames",
                                       delta=a["onset_delta"], wait=wait)
    k = int(round(a["onset_strength_ms"] / 1000 * sr / hop))     # sum the envelope around each peak: a click's
    csum = np.concatenate([[0.0], np.cumsum(env)])                 # jump is split across 2-4 envelope frames
    raw = np.array([csum[min(len(env), p + k + 1)] - csum[max(0, p - k)] for p in peaks])
    at = librosa.onset.onset_backtrack(peaks, env) if (a["onset_backtrack"] and len(peaks)) else peaks
    times = np.asarray(at, np.float64) * hop / sr + a["onset_offset_ms"] / 1000.0
    return times, np.asarray(raw, np.float64), env


def snap(times: np.ndarray, fps: float, n: int) -> np.ndarray:
    """Onset times -> nearest frame index."""
    return np.clip(np.round(np.asarray(times) * fps).astype(int), 0, max(0, n - 1))


def analyze(track: Path, cfg: dict, log=print) -> dict:
    import librosa
    a, fps = cfg["analysis"], cfg["video"]["fps"]
    sr = a["sample_rate"]
    hop = sr // fps                                          # one analysis frame per video frame
    t0 = time.time()
    y = decode_audio(track, sr)
    if not len(y):
        raise SystemExit(f"{track} has no audio")
    duration = len(y) / sr
    n = max(1, int(round(duration * fps)))

    # librosa centres frame k on sample k*hop = time k/fps (see the module docstring for why that's right)
    nf = a["n_fft"]
    rms = _fit(librosa.feature.rms(y=y, frame_length=nf, hop_length=hop)[0], n)
    cent = _fit(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=nf, hop_length=hop)[0], n)
    flat = _fit(librosa.feature.spectral_flatness(y=y, n_fft=nf, hop_length=hop)[0], n)

    rms_db = 20 * np.log10(np.maximum(rms, 1e-9) / max(rms.max(), 1e-9))
    rms_norm = np.clip(1.0 + rms_db / a["rms_range_db"], 0.0, 1.0)
    gate = rms_db > a["gate_db"]
    lo, hi = a["normalize_percentiles"]
    centroid = _norm(np.log2(np.maximum(cent, 20.0)), gate, lo, hi)
    flatness = _norm(10 * np.log10(np.maximum(flat, 1e-10)), gate, lo, hi)

    times, flux, env = detect_onsets(y, cfg)
    frames = snap(times, fps, n)
    # several onsets inside one frame -> one cut, keep the strongest
    best: dict[int, tuple] = {}
    for f, t, r in zip(frames, times, flux):
        if f not in best or r > best[f][1]:
            best[int(f)] = (float(t), float(r))
    of = sorted(best)
    oflux = np.array([best[f][1] for f in of])
    every = np.ones(len(of), bool)
    loud = np.array([rms_norm[f:f + 2].max() for f in of])
    w = a["onset_loudness_weight"]
    oraw = (1 - w) * (_norm(oflux, every, lo, hi) if len(of) > 1 else np.ones(len(of))) + w * loud
    strength = _norm(oraw, every, lo, hi) if len(of) > 1 else np.ones(len(of))

    # onset envelope per video frame (max within half a frame either side), for plots and the debug overlay
    env_t = np.arange(len(env)) * a["onset_hop"] / sr
    env_f = np.zeros(n)
    np.maximum.at(env_f, snap(env_t, fps, n), env)
    env_f = env_f / max(np.percentile(env[env > 0], 99.5) if (env > 0).any() else 1.0, 1e-9)

    r5 = lambda v: [round(float(x), 5) for x in v]
    data = {
        "version": VERSION,
        "track": str(track),
        "fps": fps,
        "sample_rate": sr,
        "duration": round(duration, 6),
        "n_frames": n,
        "seconds_to_analyse": round(time.time() - t0, 2),
        "onsets": [{"frame": f, "time": round(best[f][0], 5), "flux": round(best[f][1], 4), "loud": round(float(l), 4),
                    "strength": round(float(s), 4)} for f, l, s in zip(of, loud, strength)],
        "frames": {
            "rms": r5(rms), "rms_db": [round(float(x), 2) for x in rms_db], "rms_norm": r5(rms_norm),
            "centroid_hz": [round(float(x), 1) for x in cent], "centroid": r5(centroid),
            "flatness_raw": r5(flat), "flatness": r5(flatness), "onset_env": r5(np.clip(env_f, 0, 1.5)),
        },
    }
    log(f"analysed {track.name}: {duration:.2f} s, {n} frames, {len(of)} onsets ({data['seconds_to_analyse']} s)")
    return data


class Analysis:
    """analysis.json with numpy arrays: a.f['rms_norm'][k], a.onset_frames, a.strength ..."""

    def __init__(self, data: dict):
        self.data = data
        self.fps = data["fps"]
        self.n = data["n_frames"]
        self.duration = data["duration"]
        self.track = Path(data["track"])
        self.f = {k: np.asarray(v, np.float64) for k, v in data["frames"].items()}
        self.onsets = data["onsets"]
        self.onset_frames = np.array([o["frame"] for o in self.onsets], int)
        self.onset_times = np.array([o["time"] for o in self.onsets], float)
        self.strength = np.array([o["strength"] for o in self.onsets], float)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(track: Path, cfg: dict) -> dict:
    return {"version": VERSION, "sha1": _sha1(track), "fps": cfg["video"]["fps"], "analysis": cfg["analysis"]}


def load_or_analyze(root: Path, cfg: dict, force: bool = False, log=print) -> Analysis:
    track = find_track(root, cfg)
    path = root / cfg["paths"]["analysis"]
    key = cache_key(track, cfg)
    if path.exists() and not force:
        try:
            data = json.loads(path.read_text())
            if data.get("key") == key:
                return Analysis(data)
            log("analysis.json is out of date (track or analysis settings changed): re-analysing")
        except (json.JSONDecodeError, KeyError):
            log("analysis.json unreadable: re-analysing")
    data = analyze(track, cfg, log)
    data["key"] = key
    path.write_text(json.dumps(data, separators=(",", ":")))
    return Analysis(data)


def suggest_sections(track: Path, k: int, onset_times=None, sr: int = 22050) -> list[float]:
    """Where the track changes character: start times for k sections (the first is 0).

    Timbre (20 MFCCs) and loudness, smoothed over about 2 s so single hits don't count, are compared with
    themselves over time. At each moment a checkerboard kernel measures how different the next ~10 s are
    from the last ~10 s (novelty). The k-1 highest novelty peaks, kept apart by a minimum spacing, become
    the section starts, each moved onto the nearest onset within 1 s so the section begins on a cut."""
    import librosa
    from scipy.ndimage import uniform_filter1d
    y = decode_audio(track, sr)
    hop, nfft = 4096, 8192
    step = hop / sr
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=nfft, hop_length=hop, n_mels=64)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel + 1e-10), n_mfcc=20)
    loud = np.log(librosa.feature.rms(y=y, frame_length=nfft, hop_length=hop) + 1e-6)
    z = lambda m: (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)
    X = uniform_filter1d(np.vstack([z(mfcc), 2.0 * z(loud)]), size=max(3, int(round(2.0 / step))), axis=1)
    X = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
    S = X.T @ X                                             # self-similarity, cosine
    T = S.shape[0]
    L = max(4, min(int(round(10.0 / step)), T // 4))        # half the kernel: ~10 s
    g = np.exp(-0.5 * (np.linspace(-1, 1, 2 * L) / 0.5) ** 2)
    sign = np.sign(np.linspace(-1, 1, 2 * L) + 1e-9)
    kern = np.outer(g, g) * np.outer(sign, sign)            # +1 within past / within future, -1 across
    nov = np.zeros(T)
    for t in range(L, T - L):
        nov[t] = float((S[t - L:t + L, t - L:t + L] * kern).sum())
    nov = np.maximum(nov, 0)
    dur = len(y) / sr
    k = int(max(2, k))
    gap = max(4.0, dur / (k * 2.5)) / step                  # boundaries at least this far apart (frames)
    picks = []
    order = np.argsort(-nov)
    for t in order:
        if len(picks) >= k - 1 or nov[t] <= 0:
            break
        if all(abs(t - u) >= gap for u in picks):
            picks.append(int(t))
    times = sorted(t * step for t in picks)
    ons = np.asarray(onset_times if onset_times is not None else [], float)
    out = [0.0]
    for t in times:
        if len(ons):
            j = int(np.argmin(np.abs(ons - t)))
            if abs(ons[j] - t) <= 1.0:
                t = float(ons[j])
        if t - out[-1] >= 1.0:
            out.append(round(t, 3))
    return out
