#!/usr/bin/env python3
"""
studio.py - the batch workflow behind ./ascii (see README.md).

  ./ascii                              render everything in input/ that isn't rendered yet
  ./ascii --preset focus               ...with a saved look (presets.json); style flags override it
  ./ascii IMG_0042.MOV --force         re-render one clip
  ./ascii --preview 10                 contact sheet: a frame every 10 s of every clip -> previews/
  ./ascii assemble --length 3:00       join output/ in filename order, trim, fade -> final/film.mp4
  ./ascii presets                      list saved looks
  ./ascii dashboard                    open the browser dashboard
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import ascii_film as af

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ASCII_FILM_ROOT") or HERE).resolve()
INPUT, OUTPUT, PREVIEWS, FINAL = ROOT / "input", ROOT / "output", ROOT / "previews", ROOT / "final"
WORK = ROOT / ".work"
META = OUTPUT / ".meta"
PRESETS = ROOT / "presets.json"
MEDIA_EXTS = af.VIDEO_EXTS | af.IMAGE_EXTS
RENDER_DEFAULTS = {"fps": 30, "size": 1920, "photo_seconds": 5.0}


def set_root(path) -> None:
    """Point the workflow at another project folder (used by tests and ./ascii dashboard --root)."""
    global ROOT, INPUT, OUTPUT, PREVIEWS, FINAL, WORK, META, PRESETS
    ROOT = Path(path).expanduser().resolve()
    INPUT, OUTPUT, PREVIEWS, FINAL = ROOT / "input", ROOT / "output", ROOT / "previews", ROOT / "final"
    WORK, META, PRESETS = ROOT / ".work", ROOT / "output" / ".meta", ROOT / "presets.json"
    _PROBE_CACHE.clear()


def ensure_dirs():
    for d in (INPUT, OUTPUT, PREVIEWS, FINAL, WORK, META):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- files

def list_inputs() -> list[Path]:
    if not INPUT.exists():
        return []
    return sorted((p for p in INPUT.iterdir() if p.is_file() and not p.name.startswith(".")
                   and p.suffix.lower() in MEDIA_EXTS), key=lambda p: p.name.lower())


def list_outputs() -> list[Path]:
    if not OUTPUT.exists():
        return []
    return sorted((p for p in OUTPUT.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
                   and not p.name.startswith(".")), key=lambda p: p.name.lower())


def output_path(src: Path) -> Path:
    # Live Photos come as IMG_1234.HEIC + IMG_1234.MOV, so keep the extension in the name.
    return OUTPUT / f"{src.stem}-{src.suffix[1:].lower()}.mp4"


def meta_path(out: Path) -> Path:
    return META / f"{out.name}.json"


def read_meta(out: Path) -> dict | None:
    try:
        return json.loads(meta_path(out).read_text())
    except (OSError, ValueError):
        return None


def resolve_files(names: list[str]) -> list[Path]:
    if not names:
        return list_inputs()
    found, inputs = [], list_inputs()
    for n in names:
        p = Path(n).expanduser()
        if p.is_file():
            found.append(p.resolve())
            continue
        hits = [q for q in inputs if q.name.lower() == p.name.lower()] or \
               [q for q in inputs if Path(q.name.lower()).match(p.name.lower())] or \
               [q for q in inputs if q.stem.lower() == p.name.lower()]
        if not hits:
            raise SystemExit(f"Not found in input/: {n}")
        found.extend(h for h in hits if h not in found)
    return found


_PROBE_CACHE: dict = {}


def probe_cached(p: Path) -> af.MediaInfo:
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime)
    info = _PROBE_CACHE.get(key)
    if info is None:
        info = af.probe(p, WORK)
        _PROBE_CACHE[key] = info
    return info


# --------------------------------------------------------------------------- presets / style

def load_presets() -> dict:
    for f in (PRESETS, HERE / "presets.json"):
        if f.exists():
            try:
                return json.loads(f.read_text())
            except ValueError as e:
                raise SystemExit(f"{f} is not valid JSON: {e}")
    return {}


def save_preset(name: str, style: dict, description: str = "") -> None:
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", name):
        raise ValueError("Preset names: letters, digits, - and _ only")
    presets = load_presets()
    entry = {"description": description} if description else {}
    entry.update({k: af._spec_str(v) for k, v in style.items() if k in af.STYLE_KINDS and v not in (None, "")})
    af.Style(entry)  # validate
    presets[name] = entry
    tmp = PRESETS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, PRESETS)


def resolve_style(preset: str | None = None, overrides: dict | None = None) -> dict:
    spec = {}
    if preset:
        presets = load_presets()
        if preset not in presets:
            raise SystemExit(f"No preset named {preset!r}. Saved presets: {', '.join(presets) or '(none)'}")
        spec.update({k: v for k, v in presets[preset].items() if k in af.STYLE_KINDS})
    spec.update({k: v for k, v in (overrides or {}).items() if k in af.STYLE_KINDS and v not in (None, "")})
    full = af.Style(spec).spec  # validates + fills defaults
    return full


def parse_duration(s) -> float:
    s = str(s).strip()
    parts = s.split(":")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        raise ValueError(f"Can't read duration {s!r}; use 3:00 or 180") from None
    secs = 0.0
    for v in vals:
        secs = secs * 60 + v
    return secs


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    m, s = divmod(sec, 60)
    return f"{int(m)}:{s:04.1f}" if s % 1 else f"{int(m)}:{int(s):02d}"


# --------------------------------------------------------------------------- render

def render_one(src: Path, style: dict, *, fps=30, size=1920, photo_seconds=5.0, jobs=None,
               encoder="x264", progress=None, cancel=None) -> dict:
    ensure_dirs()
    out = output_path(src)
    stats = af.render(src, out, style, fps=fps, size=size, photo_seconds=photo_seconds, jobs=jobs,
                      encoder=encoder, progress=progress, cancel=cancel, cache_dir=WORK)
    st = src.stat()
    meta = {**stats, "render_speed_fps": stats["fps"],
            "source": src.name, "source_size": st.st_size, "source_mtime": st.st_mtime,
            "style": style, "fps": fps, "size": size, "photo_seconds": photo_seconds,
            "rendered_at": datetime.now().isoformat(timespec="seconds")}
    meta_path(out).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"output": out, **stats}


def render_state(src: Path, style: dict | None = None, settings: dict | None = None) -> tuple[str, str]:
    """('new'|'done'|'stale'|'differs', reason)"""
    out = output_path(src)
    if not out.exists():
        return "new", ""
    meta = read_meta(out) or {}
    st = src.stat()
    if meta and (meta.get("source_size") != st.st_size or abs(meta.get("source_mtime", 0) - st.st_mtime) > 1):
        return "stale", "source file changed since it was rendered"
    if style is not None and meta.get("style") and meta.get("style") != style:
        return "differs", "rendered with a different style"
    if settings and meta and any(meta.get(k) != v for k, v in settings.items() if k in meta):
        return "differs", "rendered with different fps/size"
    return "done", ""


def cmd_render(files, style, *, force, fps, size, photo_seconds, jobs, encoder):
    files = resolve_files(files)
    if not files:
        print(f"Nothing to render: drop .mov / .heic / .jpg files into {INPUT}")
        return
    settings = {"fps": fps, "size": size}
    t0 = time.time()
    done = skipped = 0
    for src in files:
        out = output_path(src)
        state, why = render_state(src, style, settings)
        if state != "new" and not force:
            note = f" ({why}; add --force to redo)" if why else ""
            print(f"  skip  {src.name} -> {out.relative_to(ROOT)}{note}")
            skipped += 1
            continue
        print(f"render  {src.name} -> {out.relative_to(ROOT)}")
        r = render_one(src, style, fps=fps, size=size, photo_seconds=photo_seconds, jobs=jobs, encoder=encoder,
                       progress=af.cli_progress("        "))
        print(f"        {fmt_time(r['duration'])} of video in {r['seconds']:.1f}s ({r['fps']} fps, "
              f"{r['width']}x{r['height']})")
        done += 1
    total = sum(probe_cached(p).duration for p in list_outputs())
    print(f"\n{done} rendered, {skipped} skipped in {time.time() - t0:.1f}s. "
          f"output/ now holds {fmt_time(total)} of footage.")


# --------------------------------------------------------------------------- contact sheet

def _sheet_init():
    cv2.setNumThreads(1)


def _sheet_task(task):
    info, t, fs, W, H, tile_h, font = task
    aw, ah = af.analysis_size(info.width, info.height, fs.cols)
    img = af.render_frame(af.grab_frame(info, t, aw, ah), fs, W, H, font)
    tw = max(2, round(W * tile_h / H))
    return cv2.resize(img, (tw, tile_h), interpolation=cv2.INTER_AREA)


def _label_font(size):
    return af._font(af.find_font(), size)


def contact_sheet(files: list[Path], style: dict, *, every: float = 10.0, size: int = 1920,
                  photo_seconds: float = 5.0, tile_h: int = 270, title: str = "", out: Path | None = None,
                  progress=None, cancel=None) -> Path:
    ensure_dirs()
    st = af.Style(style)
    font = af.find_font()
    clips = []
    for src in files:
        info = probe_cached(src)
        dur = info.duration if info.kind == "video" else photo_seconds
        if info.kind == "video":
            times = list(np.arange(0.0, max(dur - 0.05, 0.001), max(every, 0.5)))
        else:
            times = [0.0] if st.is_static() else [0.0, dur / 2, dur * 0.999]
        W, H = af.out_size(info.width, info.height, size)
        clips.append((src, info, dur, W, H, times))
    tasks = [(info, float(t), st.at(float(t), dur), W, H, tile_h, font)
             for src, info, dur, W, H, times in clips for t in times]
    if not tasks:
        raise SystemExit("No clips to preview.")

    tiles = []
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(min(8, max(1, (os.cpu_count() or 4) - 2), len(tasks)), mp_context=ctx,
                               initializer=_sheet_init)
    ok = False
    try:
        futs = [pool.submit(_sheet_task, t) for t in tasks]
        for i, f in enumerate(futs):
            while True:
                if cancel is not None and cancel.is_set():
                    raise af.Cancelled()
                try:
                    tiles.append(f.result(timeout=0.2))   # BrokenProcessPool if a worker died
                    break
                except FutureTimeout:
                    continue
            if progress:
                progress(i + 1, len(tasks))
        ok = True
    finally:
        af.stop_pool(pool, graceful=ok)

    # layout: a header per clip, then its tiles flowing left to right
    gap, pad, head_h, top_h = 10, 24, 34, 64
    sheet_w = max(2 * pad + 5 * (round(tile_h * 16 / 9) + gap) - gap,      # five 16:9 tiles per row
                  max(t.shape[1] for t in tiles) + 2 * pad)
    f_title, f_head, f_tile = _label_font(22), _label_font(17), _label_font(14)
    blocks, k = [], 0
    for src, info, dur, W, H, times in clips:
        items = tiles[k:k + len(times)]
        k += len(times)
        rows, row, x = [], [], pad
        for t_, tile in zip(times, items):
            if row and x + tile.shape[1] > sheet_w - pad:
                rows.append(row)
                row, x = [], pad
            row.append((t_, tile, x))
            x += tile.shape[1] + gap
        if row:
            rows.append(row)
        blocks.append((src, info, dur, rows))
    height = top_h + sum(head_h + len(rows) * (tile_h + gap) + 18 for *_, rows in blocks) + pad
    sheet = Image.new("RGB", (sheet_w, height), (17, 18, 17))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 20), title or "contact sheet", font=f_title, fill=(232, 232, 226))
    y = top_h
    for src, info, dur, rows in blocks:
        meta = f"{fmt_time(dur)}  {info.width}x{info.height}" + ("  photo" if info.kind == "image" else "")
        d.text((pad, y + 8), src.name, font=f_head, fill=(156, 196, 122))
        d.text((pad + d.textlength(src.name, font=f_head) + 16, y + 8), meta, font=f_head, fill=(130, 132, 126))
        y += head_h
        for row in rows:
            for t_, tile, x in row:
                sheet.paste(Image.fromarray(tile), (x, y))
                lab = fmt_time(t_)
                tw = d.textlength(lab, font=f_tile)
                d.rectangle((x, y + tile_h - 22, x + tw + 12, y + tile_h), fill=(0, 0, 0))
                d.text((x + 6, y + tile_h - 20), lab, font=f_tile, fill=(240, 240, 235))
            y += tile_h + gap
        y += 18
    if out is None:
        out = PREVIEWS / f"contact_{datetime.now():%Y%m%d-%H%M%S}.png"
    sheet.save(out, optimize=False, compress_level=3)
    return out


def style_summary(style: dict, preset: str | None) -> str:
    keys = [k for k in af.STYLE_KINDS if style.get(k) != af.DEFAULT_STYLE[k]]
    bits = " · ".join(f"{k} {style[k]}" for k in keys) or "default style"
    return (f"{preset}  —  " if preset else "") + bits


# --------------------------------------------------------------------------- assembly

def run_ffmpeg(cmd: list, total_seconds: float, progress=None, cancel=None):
    cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL, text=True)
    try:
        for line in proc.stdout:
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise af.Cancelled()
            if line.startswith("out_time_us=") and progress:
                try:
                    us = int(line.split("=", 1)[1])
                    progress(min(us / 1e6, total_seconds), total_seconds)
                except ValueError:
                    pass
        if proc.wait() != 0:
            err.seek(0)
            raise RuntimeError(err.read().decode(errors="replace")[-1500:])
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        err.close()


def assemble(length="3:00", fade: float = 1.0, *, out: Path | None = None, size: str | None = None,
             fps: float | None = None, progress=None, cancel=None, log=print) -> dict:
    ensure_dirs()
    clips = list_outputs()
    if not clips:
        raise SystemExit("output/ is empty: render some clips first (./ascii)")
    infos = [probe_cached(c) for c in clips]
    total = sum(i.duration for i in infos)
    want = parse_duration(length)
    L = want
    if total + 1e-3 < want:
        log(f"warning: output/ only holds {fmt_time(total)} of footage, less than the {fmt_time(want)} asked for; "
            f"the film will be {fmt_time(total)}.")
        L = total
    if size:
        m = re.fullmatch(r"(\d+)[x:](\d+)", size.strip())
        if not m:
            raise ValueError("--size for assemble looks like 1920x1080")
        W, H = af.even(int(m.group(1))), af.even(int(m.group(2)))
    else:
        W, H = Counter((i.width, i.height) for i in infos).most_common(1)[0][0]
    fps = fps or round(Counter(round(i.fps, 3) for i in infos).most_common(1)[0][0], 3) or 30
    nframes = int(round(L * fps))
    L = nframes / fps
    fade = max(0.0, min(float(fade), L / 3))
    n = len(clips)
    fc = []
    for k in range(n):
        fc.append(f"[{k}:v:0]scale={W}:{H}:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},format=yuv420p[v{k}]")
        fc.append(f"[{k}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{k}]")
    fc.append("".join(f"[v{k}][a{k}]" for k in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    vchain = f"[vc]trim=end_frame={nframes},setpts=PTS-STARTPTS"
    achain = f"[ac]atrim=duration={L:.6f},asetpts=PTS-STARTPTS,apad=whole_dur={L:.6f}"
    if fade > 0:
        vchain += f",fade=t=in:st=0:d={fade},fade=t=out:st={L - fade:.6f}:d={fade}"
        achain += f",afade=t=in:st=0:d={fade},afade=t=out:st={L - fade:.6f}:d={fade}"
    fc += [vchain + "[vo]", achain + "[ao]"]
    out = Path(out) if out else FINAL / "film.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.partial.mp4")
    cmd = [af.tool("ffmpeg"), "-v", "error", "-y"]
    for c in clips:
        cmd += ["-i", str(c)]
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vo]", "-map", "[ao]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-r", str(fps), "-frames:v", str(nframes),
            "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-t", f"{L:.6f}",
            "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    t0 = time.time()
    try:
        run_ffmpeg(cmd, L, progress, cancel)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"output": out, "clips": [c.name for c in clips], "length": L, "fade": fade, "width": W,
            "height": H, "fps": fps, "seconds": round(time.time() - t0, 1), "footage": total}


# --------------------------------------------------------------------------- CLI

def _fix_optional_args(argv: list[str]) -> list[str]:
    """--preview and --invert take an optional value; don't let them swallow a filename."""
    out = list(argv)
    i = 0
    while i < len(out):
        tok = out[i]
        nxt = out[i + 1] if i + 1 < len(out) else None
        if tok == "--preview" and (nxt is None or not re.fullmatch(r"\d+(\.\d+)?", nxt)):
            out.insert(i + 1, "10")
        if tok == "--invert" and (nxt is None or not re.fullmatch(
                r"(?i)(true|false|yes|no|on|off|[01])|\d+(\.\d+)?%?\s*:.*", nxt)):
            out.insert(i + 1, "true")
        i += 1
    return out


def open_file(p: Path):
    if sys.platform == "darwin":
        subprocess.run(["open", str(p)])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv and argv[0] in ("render", "preview", "assemble", "presets", "dashboard") else "render"
    if cmd == "preview":
        argv = ["--preview"] + argv
    ensure_dirs()

    if cmd == "presets":
        presets = load_presets()
        if not presets:
            print("No presets yet. Save one from the dashboard or add it to presets.json.")
        for name, p in presets.items():
            desc = p.get("description", "")
            print(f"{name:12s} {desc}")
            print(" " * 13 + "  ".join(f"--{k} {json.dumps(str(v), ensure_ascii=False)}"
                                       for k, v in p.items() if k in af.STYLE_KINDS))
        return

    if cmd == "dashboard":
        ap = argparse.ArgumentParser(prog="./ascii dashboard")
        ap.add_argument("--port", type=int, default=8777)
        ap.add_argument("--no-open", action="store_true", help="don't open the browser")
        ap.add_argument("--root", help=argparse.SUPPRESS)
        a = ap.parse_args(argv)
        import dashboard  # it imports this file as the module "studio", a separate object from __main__
        if a.root:
            dashboard.st.set_root(a.root)
        dashboard.serve(a.port, open_browser=not a.no_open)
        return

    if cmd == "assemble":
        ap = argparse.ArgumentParser(prog="./ascii assemble", description="Join output/*.mp4 in filename order "
                                     "into one film, trimmed to an exact length, with fades.")
        ap.add_argument("--length", default="3:00", help="exact running time, e.g. 3:00 or 175.5 (default 3:00)")
        ap.add_argument("--fade", type=float, default=1.0, help="fade in/out seconds, picture and sound (default 1)")
        ap.add_argument("--size", help="frame size, e.g. 1920x1080 (default: the most common clip size)")
        ap.add_argument("--out", help="output file (default final/film.mp4)")
        ap.add_argument("--no-open", action="store_true", help="don't open the film when done")
        a = ap.parse_args(argv)
        clips = list_outputs()
        print("Joining, in this order:")
        for c in clips:
            print(f"  {c.name:40s} {fmt_time(probe_cached(c).duration)}")
        r = assemble(a.length, a.fade, out=a.out, size=a.size, progress=_secs_progress())
        sys.stderr.write("\n")
        print(f"\n{r['output']}  {fmt_time(r['length'])} exactly, {r['width']}x{r['height']} @ {r['fps']:g} fps, "
              f"{r['fade']:g}s fades  ({r['seconds']}s)")
        if not a.no_open:
            open_file(r["output"])
        return

    ap = argparse.ArgumentParser(prog="./ascii", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="limit to these files in input/ (names or globs); default: all")
    ap.add_argument("--preset", help="named look from presets.json; style flags override its values")
    af.add_style_args(ap)
    ap.add_argument("--force", action="store_true", help="re-render files that already have an output")
    ap.add_argument("--preview", type=float, metavar="N",
                    help="don't render: make a contact sheet with a frame every N seconds (default 10)")
    ap.add_argument("--tile", type=int, default=270, help="contact-sheet tile height in px (default 270)")
    ap.add_argument("--no-open", action="store_true", help="don't open the contact sheet when done")
    ap.add_argument("--fps", type=float, default=RENDER_DEFAULTS["fps"])
    ap.add_argument("--size", type=int, default=RENDER_DEFAULTS["size"], help="long edge in px (default 1920)")
    ap.add_argument("--photo-seconds", type=float, default=RENDER_DEFAULTS["photo_seconds"],
                    help="length of the clip made from each photo (default 5)")
    ap.add_argument("--jobs", type=int, help=f"parallel segments per clip (default {af.default_jobs()})")
    ap.add_argument("--encoder", choices=list(af.ENCODERS), default="x264")
    a = ap.parse_args(_fix_optional_args(argv))
    try:
        style = resolve_style(a.preset, af.style_from_args(a))
    except ValueError as e:
        ap.error(str(e))

    if a.preview is not None:
        files = resolve_files(a.files)
        if not files:
            raise SystemExit(f"Nothing to preview: drop clips into {INPUT}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = PREVIEWS / f"contact_{a.preset or 'custom'}_{stamp}.png"
        title = f"{style_summary(style, a.preset)}   |   every {a.preview:g}s   |   {stamp}"
        print(f"Contact sheet: every {a.preview:g}s of {len(files)} file(s)")
        p = contact_sheet(files, style, every=a.preview, size=a.size, photo_seconds=a.photo_seconds,
                          tile_h=a.tile, title=title, out=out, progress=af.cli_progress("  "))
        print(p)
        if not a.no_open:
            open_file(p)
        return

    cmd_render(a.files, style, force=a.force, fps=a.fps, size=a.size, photo_seconds=a.photo_seconds,
               jobs=a.jobs, encoder=a.encoder)


def _secs_progress():
    t0 = time.time()

    def cb(done, total):
        bar = "#" * int(30 * done / max(total, 1e-9))
        sys.stderr.write(f"\r  [{bar:<30}] {fmt_time(done)} / {fmt_time(total)}  {time.time() - t0:5.1f}s ")
        sys.stderr.flush()
    return cb


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("\nstopped\n")
        sys.exit(130)
