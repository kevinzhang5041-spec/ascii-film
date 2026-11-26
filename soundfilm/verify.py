"""Checks on a finished render: is every cut on an onset frame, is every frame where it should be, is the
audio in and in sync, does the duration match. Plus stills/contact sheets for looking at the result.

  plan         every cut in the score (edl.csv) sits on an onset frame from analysis.json, and vice versa
  frames       the decoded video, frame by frame, matches what the renderer sent to the encoder (fp.npy)
               -> proves no frame was dropped, duplicated or shifted on the way through x264 / concat / mux
  audio        onsets re-detected from the muxed AAC audio land on the same frames as in the original track
  blind        pixels vs sound, no score used: around each isolated onset heard in the muxed audio, on which
               frame does the most new light/ink appear? (0 = exactly on the onset frame)
  truth        (test track only) detected onsets vs the known event times the test track was built from
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import ascii_film as af

from . import analysis, score
from .render import FP_SIZE, sidecar

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def ffprobe(path: Path) -> dict:
    r = subprocess.run([af.tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                        str(path)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout)


def decode_thumbs(video: Path) -> np.ndarray:
    """Every frame -> 32x18 RGB, the same way render.fingerprint() makes them (via 320x180)."""
    w, h = FP_SIZE[0] * 10, FP_SIZE[1] * 10
    cmd = [af.tool("ffmpeg"), "-v", "error", "-nostdin", "-i", str(video), "-an", "-fps_mode", "passthrough",
           "-vf", f"scale={w}:{h}:flags=area,format=rgb24", "-f", "rawvideo", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return np.stack([cv2.resize(f, FP_SIZE, interpolation=cv2.INTER_AREA) for f in frames])


def luma(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) @ LUMA


def audio_offsets(orig: np.ndarray, muxed: np.ndarray, sr: int, points: int = 9, win_s: float = 1.0,
                  reach_s: float = 0.1) -> list[int]:
    """Where the muxed soundtrack sits relative to the original, in samples, measured by cross-correlation
    at `points` places from start to end (so drift shows too). Silent stretches are skipped."""
    from scipy.signal import correlate
    win, reach = int(win_s * sr), int(reach_s * sr)
    n = min(len(orig), len(muxed))
    if n < win + 2 * reach:
        win = max(0, n - 2 * reach)
    lags = []
    for i in np.linspace(reach, max(reach, n - win - reach), points).astype(int):
        x = orig[i:i + win]
        if len(x) < sr // 20 or np.sqrt(np.mean(x ** 2)) < 1e-4:
            continue
        yy = muxed[i - reach:i + win + reach]
        if len(yy) < len(x) + 2 * reach:
            continue
        lags.append(int(np.argmax(correlate(yy, x, mode="valid", method="fft"))) - reach)
    return lags


def match(a: np.ndarray, b: np.ndarray, tol: int):
    """Pair each value of a with the nearest value of b within tol. Returns (errors b-a, unmatched a, unmatched b)."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    errs, used = [], set()
    un_a = []
    for x in a:
        if not len(b):
            un_a.append(x)
            continue
        i = int(np.argmin(np.abs(b - x)))
        if abs(b[i] - x) <= tol and i not in used:
            used.add(i)
            errs.append(b[i] - x)
        else:
            un_a.append(x)
    un_b = [b[i] for i in range(len(b)) if i not in used]
    return np.array(errs), un_a, un_b


def blind_offsets(L: np.ndarray, onset_frames, reach: int = 2) -> list[tuple[int, int]]:
    """For each onset with no other onset within `reach` frames: the offset (in frames) of the frame where the
    most new light/ink appears, within +-reach. Uses only the pixels and the onset times.

    "New light" (the positive part of the frame difference) rather than any change, because a hold at a coarse
    grid re-flows every glyph when the column count ticks up by one, and fades to black are changes too."""
    new_light = np.r_[0, np.maximum(np.diff(L, axis=0), 0).mean((1, 2))]
    on = sorted(set(int(f) for f in onset_frames))
    out = []
    for f in on:
        if reach <= f < len(L) - reach and not any(0 < abs(g - f) <= reach for g in on):
            out.append((f, int(np.argmax(new_light[f - reach:f + reach + 1]) - reach)))
    return out


def verify(video: Path, root: Path, cfg: dict, log=print) -> dict:
    video = Path(video)
    meta = json.loads(sidecar(video, ".json").read_text())
    fps, f0, f1 = meta["fps"], meta["first_frame"], meta["end_frame"]
    total = f1 - f0
    an = analysis.load_or_analyze(root, cfg, log=log)
    rep: dict = {"video": str(video), "frames_expected": total}

    # ---- container: streams, sizes, durations ---------------------------------------------------------
    pr = ffprobe(video)
    vs = next(s for s in pr["streams"] if s["codec_type"] == "video")
    aus = [s for s in pr["streams"] if s["codec_type"] == "audio"]
    num, den = map(int, vs["r_frame_rate"].split("/"))
    exp_dur = total / fps
    rep["container"] = dict(
        video_codec=vs["codec_name"], size=f"{vs['width']}x{vs['height']}", fps=num / den,
        nb_frames=int(vs.get("nb_frames", 0)), video_duration=float(vs["duration"]),
        audio=bool(aus), audio_codec=aus[0]["codec_name"] if aus else None,
        audio_channels=aus[0].get("channels") if aus else None,
        audio_rate=int(aus[0]["sample_rate"]) if aus else None,
        audio_duration=float(aus[0]["duration"]) if aus else None,
        audio_start=float(aus[0].get("start_time", 0)) if aus else None,
        video_start=float(vs.get("start_time", 0)),
        file_duration=float(pr["format"]["duration"]), expected_duration=round(exp_dur, 6),
        track_duration=an.duration,
    )
    c = rep["container"]
    c["video_vs_expected_ms"] = round((c["video_duration"] - exp_dur) * 1000, 2)
    c["audio_vs_expected_ms"] = round((c["audio_duration"] - exp_dur) * 1000, 2) if aus else None
    if not meta["preview"]:
        c["film_vs_track_ms"] = round((c["file_duration"] - an.duration) * 1000, 2)

    # ---- plan: cuts vs onsets ---------------------------------------------------------------------------
    edl = score.read_edl(sidecar(video, ".edl.csv"))
    cuts = np.array([int(r["frame"]) for r in edl if r["kind"] == "cut"])
    onsets = an.onset_frames[(an.onset_frames >= f0) & (an.onset_frames < f1)]
    errs, miss, extra = match(onsets, cuts, tol=10 ** 6)
    rep["plan"] = dict(onsets=len(onsets), cuts=len(cuts), onsets_without_cut=len(set(onsets) - set(cuts)),
                       cuts_off_onset=len(set(cuts) - set(onsets)),
                       max_error_frames=int(np.abs(errs).max()) if len(errs) else 0)

    # ---- frames: decoded video vs what the renderer encoded -------------------------------------------
    dec = luma(decode_thumbs(video))
    ref = luma(np.load(sidecar(video, ".fp.npy")))
    n = min(len(dec), len(ref))
    shifts = range(-4, 5)
    err = {s: float(np.abs(dec[max(0, -s):n - max(0, s)] - ref[max(0, s):n - max(0, -s)]).mean()) for s in shifts}
    best = min(err, key=err.get)
    # frames that differ from both neighbours (cuts, flashes) pin the alignment down frame by frame
    d = np.abs(np.diff(ref, axis=0)).mean((1, 2))
    distinct = [k for k in range(1, n - 1) if d[k - 1] > 6 and d[k] > 6]
    wrong = [k for k in distinct
             if np.argmin([np.abs(dec[k] - ref[k + o]).mean() for o in (-1, 0, 1)]) != 1]
    rep["frames"] = dict(decoded=len(dec), encoded=len(ref), best_offset=best,
                         mean_abs_error=round(err[0], 2), next_best_error=round(sorted(err.values())[1], 2),
                         distinct_frames_checked=len(distinct), distinct_frames_misplaced=len(wrong))

    # ---- audio: onsets in the muxed soundtrack ----------------------------------------------------------
    if aus:
        sr = cfg["analysis"]["sample_rate"]
        y = analysis.decode_audio(video, sr)
        t_mux, _, _ = analysis.detect_onsets(y, cfg)
        mux_frames = analysis.snap(t_mux, fps, total) + f0
        e_f, un_a, un_m = match(onsets, mux_frames, tol=2)
        ref_t = an.onset_times[(an.onset_frames >= f0) & (an.onset_frames < f1)] - f0 / fps
        e_t, _, _ = match(ref_t, t_mux, tol=0.05)
        lags = audio_offsets(analysis.decode_audio(an.track, sr)[int(round(f0 / fps * sr)):], y, sr)
        rep["audio"] = dict(offsets_samples=lags, max_offset_ms=round(max(map(abs, lags)) / sr * 1000, 3) if lags else None,
                            onsets_in_muxed_audio=len(mux_frames), matched=len(e_f),
                            same_frame=int((e_f == 0).sum()) if len(e_f) else 0,
                            max_error_frames=int(np.abs(e_f).max()) if len(e_f) else None,
                            mean_shift_ms=round(float(e_t.mean() * 1000), 2) if len(e_t) else None,
                            max_shift_ms=round(float(np.abs(e_t).max() * 1000), 2) if len(e_t) else None,
                            unmatched_original=len(un_a), unmatched_muxed=len(un_m))
        # ---- blind: pixels vs sound, no score involved ----------------------------------------------------
        heard = np.intersect1d(mux_frames, onsets) - f0       # heard in both the muxed and the original audio
        offs = blind_offsets(dec, heard)
        o = np.array([x for _, x in offs])
        rep["blind"] = dict(isolated_onsets=len(offs), exact=int((o == 0).sum()) if len(o) else 0,
                            max_error_frames=int(np.abs(o).max()) if len(o) else None,
                            off=[(f + f0, x) for f, x in offs if x])

    # ---- truth: the synthetic track's own event list ----------------------------------------------------
    ev_path = an.track.with_name(an.track.stem + "_events.json")
    if ev_path.exists():
        ev = json.loads(ev_path.read_text())["events"]
        gt_t = np.array([e["time"] for e in ev if f0 / fps <= e["time"] < f1 / fps])
        gt_f = analysis.snap(gt_t, fps, an.n)
        e_f, un_gt, un_det = match(gt_f, onsets, tol=2)
        sel = (an.onset_frames >= f0) & (an.onset_frames < f1)
        e_t, _, _ = match(gt_t, an.onset_times[sel], tol=0.05)
        rep["truth"] = dict(events=len(gt_t), detected=len(onsets), matched=len(e_f),
                            exact_frame=int((e_f == 0).sum()) if len(e_f) else 0,
                            max_error_frames=int(np.abs(e_f).max()) if len(e_f) else None,
                            mean_time_error_ms=round(float(np.abs(e_t).mean() * 1000), 2) if len(e_t) else None,
                            max_time_error_ms=round(float(np.abs(e_t).max() * 1000), 2) if len(e_t) else None,
                            missed_events=[round(float(t), 3) for t in gt_t[np.isin(gt_f, un_gt)]],
                            extra_onset_frames=[int(f) for f in un_det])

    # ---- verdict ---------------------------------------------------------------------------------------
    ok = dict(
        plan=rep["plan"]["onsets_without_cut"] == 0 and rep["plan"]["cuts_off_onset"] == 0,
        frame_count=len(dec) == len(ref) == total == c["nb_frames"],
        frame_alignment=best == 0 and not wrong,
        audio_present=bool(aus) and c["audio_channels"] == 2,
        audio_sync=bool(aus) and rep["audio"]["max_offset_ms"] is not None and rep["audio"]["max_offset_ms"] <= 1.0,
        duration=abs(c["video_vs_expected_ms"]) < 1 and (not aus or abs(c["audio_vs_expected_ms"]) < 1000 / fps),
    )
    rep["ok"] = ok
    rep["passed"] = all(ok.values())
    sidecar(video, ".verify.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def summary(rep: dict) -> str:
    c, p, f = rep["container"], rep["plan"], rep["frames"]
    L = [f"{Path(rep['video']).name}: {'PASS' if rep['passed'] else 'FAIL'}",
         f"  container  {c['video_codec']} {c['size']} {c['fps']:g} fps, {c['nb_frames']} frames; audio "
         f"{c['audio_codec']} {c['audio_channels']}ch {c['audio_rate']} Hz",
         f"  duration   video {c['video_duration']:.4f} s, audio {c['audio_duration']:.4f} s, expected "
         f"{c['expected_duration']:.4f} s (video {c['video_vs_expected_ms']:+.1f} ms, audio "
         f"{c['audio_vs_expected_ms']:+.1f} ms)" + (f"; film vs track {c['film_vs_track_ms']:+.1f} ms"
                                                   if "film_vs_track_ms" in c else ""),
         f"  plan       {p['cuts']} cuts for {p['onsets']} onsets: {p['onsets_without_cut']} onsets without a cut, "
         f"{p['cuts_off_onset']} cuts off an onset, max sync error {p['max_error_frames']} frames",
         f"  frames     {f['decoded']} decoded / {f['encoded']} encoded, best alignment offset {f['best_offset']:+d} "
         f"(error {f['mean_abs_error']} vs {f['next_best_error']} next best), {f['distinct_frames_misplaced']} of "
         f"{f['distinct_frames_checked']} cut/flash frames misplaced"]
    if "audio" in rep:
        a, b = rep["audio"], rep["blind"]
        lags = a.get("offsets_samples") or []
        L.append(f"  audio      muxed soundtrack vs original at {len(lags)} points: offset "
                 + (f"{min(lags):+d}..{max(lags):+d} samples (max {a['max_offset_ms']} ms)" if lags else "not measurable")
                 + f"; onsets re-detected in the AAC: {a.get('same_frame', '?')}/{rep['plan']['onsets']} on the same "
                 f"frame, {a['matched'] - a.get('same_frame', 0)} within 2 frames (detector, not timing)")
        L.append(f"  blind      biggest burst of new light within +-2 frames of each isolated onset: on the onset "
                 f"frame for {b['exact']}/{b['isolated_onsets']}" + (f"; off: " + ", ".join(
                     f"frame {f} by {x:+d}" for f, x in b["off"]) if b["off"] else ""))
    if "truth" in rep:
        t = rep["truth"]
        L.append(f"  truth      {t['matched']}/{t['events']} test-track events detected ({t['exact_frame']} on the "
                 f"exact frame, max error {t['max_error_frames']} frames; timing error mean "
                 f"{t['mean_time_error_ms']} ms, max {t['max_time_error_ms']} ms); "
                 f"{len(t['extra_onset_frames'])} extra onsets")
    bad = [k for k, v in rep["ok"].items() if not v]
    if bad:
        L.append("  FAILED: " + ", ".join(bad))
    return "\n".join(L)


# --------------------------------------------------------------------------- stills

def pick_stills(edl: list[dict], limit: int = 12) -> list[tuple[int, str]]:
    """Frames worth looking at, chosen from the score: a flash, a cut per pool and per charset, layering..."""
    rows = [dict(r, frame=int(r["frame"]), cols=int(r["cols"]), flash=r["flash"] == "1", strength=float(r["strength"]))
            for r in edl]
    lit = [r for r in rows if r["kind"] in ("cut", "hold") and float(r["gain"]) >= 0.99 and not r["flash"]]
    strongest = lambda cs: max(cs, key=lambda r: r["strength"]) if cs else None
    picks: list[tuple[int, str]] = []

    def add(r, why):
        if r is not None and all(abs(r["frame"] - p) > 0 for p, _ in picks):
            picks.append((r["frame"], why))

    first = lambda cond: next((r for r in rows if cond(r)), None)
    add(first(lambda r: r["flash"]), "invert flash")
    for pool in ("bright", "texture", "dark"):
        add(strongest([r for r in lit if r["kind"] == "cut" and r["pool"] == pool]), f"cut -> {pool}/")
    for band in dict.fromkeys(r["band"] for r in lit):
        add(strongest([r for r in lit if r["band"] == band and r["kind"] == "cut"]), f"chars: {band}")
    add(strongest([r for r in lit if r["kind"] == "cut" and float(r["mix"]) > 0.8]), "tonal -> colour")
    add(strongest([r for r in lit if r["kind"] == "cut" and float(r["mix"]) < 0.1]), "noisy -> mono")
    add(first(lambda r: r["layer_name"] and r["kind"] == "cut"), "layered (dense)")
    if lit:
        add(min(lit, key=lambda r: (r["cols"], -r["strength"])), "coarsest grid")
        add(max(lit, key=lambda r: r["cols"]), "finest grid")
    add(first(lambda r: r["kind"] == "decay"), "decay toward black")
    add(first(lambda r: r["kind"] == "silence" and r["frame"] > 0), "silence = black")
    return sorted(picks[:limit])


def stills(video: Path, out_dir: Path | None = None, frames: list[tuple[int, str]] | None = None) -> Path:
    """Export chosen frames as full-size PNGs plus a labelled contact sheet. Returns the contact sheet."""
    from PIL import Image, ImageDraw, ImageFont
    video = Path(video)
    meta = json.loads(sidecar(video, ".json").read_text())
    edl = score.read_edl(sidecar(video, ".edl.csv"))
    frames = frames or pick_stills(edl)
    f0 = meta["first_frame"]
    out_dir = Path(out_dir or video.with_name(video.stem + "_stills"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = "+".join(f"eq(n\\,{f - f0})" for f, _ in frames)
    tmpl = out_dir / "_%03d.png"
    subprocess.run([af.tool("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-an", "-vf", f"select='{sel}'",
                    "-fps_mode", "passthrough", str(tmpl)], check=True)
    by_frame = {int(r["frame"]): r for r in edl}
    tiles = []
    for i, (f, why) in enumerate(frames):
        src = out_dir / f"_{i + 1:03d}.png"
        dst = out_dir / f"frame{f:05d}.png"
        src.replace(dst)
        r = by_frame[f]
        label = (f"{why}   frame {f} ({f / meta['fps']:.3f} s)\n{r['kind']}  pool {r['pool'] or '-'}  "
                 f"chars {r['band']}  {r['cols']} cols  colour {float(r['mix']):.2f}")
        tiles.append((Image.open(dst).convert("RGB"), label))
    tw, th, pad, lab = 640, 360, 10, 44
    ncol = 3
    nrow = -(-len(tiles) // ncol)
    sheet = Image.new("RGB", (ncol * (tw + pad) + pad, nrow * (th + lab + pad) + pad), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(af.find_font(), 13)
    except OSError:
        font = ImageFont.load_default()
    for i, (im, label) in enumerate(tiles):
        x, y = pad + (i % ncol) * (tw + pad), pad + (i // ncol) * (th + lab + pad)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
        d.text((x + 2, y + th + 5), label, fill=(225, 225, 225), font=font)
    contact = video.with_name(video.stem + "_contact.png")
    sheet.save(contact)
    return contact
