"""Step 3b: render the score to H.264 with the original audio.

Same design as ascii_film.render: the frame range is cut into segments, each worker process draws its
frames in memory and pipes them straight into its own ffmpeg/x264 encoder, then the segments are joined
by stream copy (frame-exact) and the soundtrack is trimmed to the same range and muxed in.

Next to every video it writes:
  <name>.edl.csv   the score: one line per frame (what happened and why)
  <name>.fp.npy    a 32x18 thumbnail of every frame as it was sent to the encoder (verify.py uses it)
  <name>.json      seed, frame range, sizes, config digest
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import cv2
import numpy as np

import ascii_film as af

from . import analysis, background, clips, config, debug, score
from .frames import FrameRenderer, work_width

MIN_SEGMENT_FRAMES = 96
FP_SIZE = (32, 18)
_W: dict = {}


def fingerprint(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, FP_SIZE, interpolation=cv2.INTER_AREA)


def _init(counter, cancel):
    cv2.setNumThreads(1)
    _W.update(counter=counter, cancel=cancel)


def _segment(job: dict) -> np.ndarray:
    """Draw and encode rows job['rows'] to job['out']. Returns their fingerprints."""
    W, H = job["W"], job["H"]
    overlay = debug.DebugOverlay(job["timeline"], job["cfg"], W, H, job["fps"]) if job["timeline"] else None
    fr = FrameRenderer(job["cfg"], job["bands"], W, H, job["font"], job["seed"], overlay, job.get("views"),
                       job.get("background"), job.get("clips"))
    counter, cancel = _W.get("counter"), _W.get("cancel")
    enc = af.Encoder(Path(job["out"]), W, H, job["fps"], job["codec"], job["threads"])
    fps_ = np.zeros((len(job["rows"]), FP_SIZE[1], FP_SIZE[0], 3), np.uint8)
    pending = 0
    try:
        for i, row in enumerate(job["rows"]):
            if cancel is not None and cancel.is_set():
                raise af.Cancelled()
            img = np.ascontiguousarray(fr.render(row))
            enc.write(img.data)
            fps_[i] = fingerprint(img)
            pending += 1
            if pending >= 6 and counter is not None:
                with counter.get_lock():
                    counter.value += pending
                pending = 0
        enc.close()
    except BaseException:
        enc.abort()
        raise
    finally:
        fr.close()
    if pending and counter is not None:
        with counter.get_lock():
            counter.value += pending
    return fps_


def mux(parts: list[Path], dst: Path, track: Path, t0: float, dur: float, bitrate: str) -> None:
    """Join the video segments (stream copy) and add the soundtrack, trimmed to [t0, t0 + dur) exactly."""
    lst = dst.with_suffix(".txt")
    lst.write_text("".join("file '{}'\n".format(str(p).replace("'", "'\\''")) for p in parts))
    af_chain = f"atrim=start={t0:.6f}:duration={dur:.6f},asetpts=PTS-STARTPTS,apad=whole_dur={dur:.6f}"
    cmd = [af.tool("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(track),
           "-map", "0:v:0", "-map", "1:a:0", "-af", af_chain, "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
           "-ar", "48000", "-ac", "2", "-movflags", "+faststart", "-f", "mp4", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lst.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr[-1500:]}")


def sidecar(video: Path, ext: str) -> Path:
    """renders/film_seed1.mp4 -> renders/film_seed1<ext>"""
    return Path(video).with_name(Path(video).stem + ext)


def output_name(seed: int, preview, debug_: bool) -> str:
    stem = f"preview_{preview[0]:g}-{preview[1]:g}s_seed{seed}" if preview else f"film_seed{seed}"
    return stem + ("_debug" if debug_ else "") + ".mp4"


def render(root: Path, cfg: dict, *, seed: int | None = None, debug_: bool = False, preview=None,
           out: Path | None = None, jobs: int | None = None, reanalyze: bool = False, log=print,
           progress=None) -> dict:
    t_start = time.time()
    seed = int(cfg["seed"] if seed is None else seed)
    fps = cfg["video"]["fps"]
    an = analysis.load_or_analyze(root, cfg, force=reanalyze, log=log)
    cache = root / ".work"
    pools = score.gather_pools(root, cfg, cache)
    from .photos import views_for
    from .sets import sections_for
    views = views_for(root, cfg, pools)                # your hand crops (images/crops.json)
    sections = sections_for(root, cfg, pools, an.fps, an.n)   # which photos each part of the track uses
    if sections:
        log(f"{len(sections)} sections: " + ", ".join(f"{s['label']} from {s['start'] / an.fps:.1f}s"
                                                     for s in sections))
    bg = background.plan(root, cfg)                    # the video in place of black, if there is one
    sc = score.build_score(an, cfg, pools, seed, views, sections, bg)
    if bg:
        b = cfg["background"]
        log(f"background video {bg['name']} ({bg['frames'] / fps:.1f} s) in the {b['show_in']} frames: "
            f"{sum(r['bg'] >= 0 for r in sc.rows) / fps:.1f} s of the film, {b['style']}, clock {b['clock']}")
    for w in sc.warnings:
        log("warning: " + w)

    if preview:
        f0 = int(np.clip(round(preview[0] * fps), 0, an.n - 1))
        f1 = int(np.clip(round(preview[1] * fps), f0 + 1, an.n))
        v = cfg["preview"]
    else:
        f0, f1, v = 0, an.n, cfg["video"]
    W, H = af.even(v["width"]), af.even(v["height"])
    codec = ["-c:v", "libx264", "-preset", str(v["preset"]), "-crf", str(v["crf"])]
    rows = sc.rows[f0:f1]
    total = len(rows)

    renders = root / cfg["paths"]["renders"]
    dst = Path(out) if out else renders / output_name(seed, preview, debug_)
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = dst.parent / f".{dst.stem}.parts"
    shutil.rmtree(work, ignore_errors=True)            # our own scratch folder from an interrupted render
    work.mkdir(parents=True)
    tmp = dst.with_name(f".{dst.stem}.partial.mp4")

    jobs = max(1, int(jobs or af.default_jobs()))
    need = clips.needed(rows)                          # clip frames this render shows
    clip_frames = clips.levels_for(root, cfg, need)
    nseg = max(1, min(jobs, total // MIN_SEGMENT_FRAMES))
    bounds = [round(i * total / nseg) for i in range(nseg + 1)]
    cpu = os.cpu_count() or 4
    base = dict(cfg=cfg, bands=sc.bands, W=W, H=H, fps=fps, font=af.find_font(), seed=seed, codec=codec,
                threads=max(2, cpu // nseg) if nseg > 1 else 0,
                timeline=debug.timeline(an, sc) if debug_ else None, views=views, background=bg,
                clips=clip_frames)
    seg_jobs = [dict(base, rows=rows[bounds[i]:bounds[i + 1]], out=str(work / f"seg{i:03d}.mp4"))
                for i in range(nseg)]

    log(f"rendering {total} frames ({f0 / fps:.2f}-{f1 / fps:.2f} s) at {W}x{H}, seed {seed}, "
        f"{len(sc.shots)} cuts, {nseg} segment{'s' if nseg > 1 else ''}{', debug overlay' if debug_ else ''}")
    pool = ev = None
    ok = False
    try:
        if need:                                       # decode each clip once, before drawing (clips.py)
            t0 = time.time()
            (work / "clips").mkdir()
            cjobs = [dict(path=p, idx=idx, view=views.get(p), aspect=W / H, work_w=work_width(cfg), fps=fps,
                          levels=clip_frames[p]["levels"], out=str(work / "clips" / f"clip{i:03d}.npy"))
                     for i, (p, idx) in enumerate(sorted(need.items(), key=lambda kv: -len(kv[1])))]
            with ThreadPoolExecutor(max(1, min(len(cjobs), jobs))) as ex:
                for f in as_completed([ex.submit(clips.extract, j) for j in cjobs]):
                    m = f.result()
                    clip_frames[m["path"]].update(file=m["file"], index=m["index"])
                    if progress:
                        progress(0, total)
            log(f"{len(need)} clip{'s' if len(need) > 1 else ''}: decoded {sum(map(len, need.values()))} frames "
                f"in {time.time() - t0:.1f} s")
        if nseg == 1:
            _W.clear()
            fp = [_segment(seg_jobs[0])]
        else:
            ctx = mp.get_context("spawn")
            counter, ev = ctx.Value("i", 0), ctx.Event()
            pool = ProcessPoolExecutor(nseg, mp_context=ctx, initializer=_init, initargs=(counter, ev))
            futs = [pool.submit(_segment, j) for j in seg_jobs]
            while True:
                done, running = wait(futs, timeout=0.25, return_when=FIRST_EXCEPTION)
                bad = next((f for f in done if f.exception() is not None), None)
                if bad is not None:
                    raise bad.exception()
                if progress:
                    progress(counter.value, total)
                if not running:
                    break
            fp = [f.result() for f in futs]
        mux([Path(j["out"]) for j in seg_jobs], tmp, an.track, f0 / fps, total / fps, cfg["video"]["audio_bitrate"])
        os.replace(tmp, dst)
        ok = True
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        if pool is not None:
            af.stop_pool(pool, ev, graceful=ok)
        shutil.rmtree(work, ignore_errors=True)
    if progress:
        progress(total, total)

    score.write_edl(rows, sc.bands, sidecar(dst, ".edl.csv"))
    np.save(sidecar(dst, ".fp.npy"), np.concatenate(fp))
    secs = time.time() - t_start
    meta = dict(video=dst.name, seed=seed, fps=fps, first_frame=f0, end_frame=f1, frames=total, width=W, height=H,
                debug=debug_, preview=list(preview) if preview else None, track=str(an.track),
                background=bg["name"] if bg else None, clips=len(need),
                analysis_sha1=an.data["key"]["sha1"], config=config.digest(cfg), cuts=sum(r["kind"] == "cut"
                                                                                         for r in rows),
                seconds=round(secs, 2), fps_rendered=round(total / secs, 1))
    sidecar(dst, ".json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {dst}  ({total} frames in {secs:.1f} s, {total / secs:.0f} fps)")
    return meta
