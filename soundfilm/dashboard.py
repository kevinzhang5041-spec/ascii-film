"""./film dashboard: the sound film in a browser tab (http://127.0.0.1:8778).

Drop the track and scans in, analyse, render previews and the full film, verify them, look at stills and
edit config.yaml, all from one page. It calls the same code as the command line.

It never deletes anything, and an upload never overwrites a file: a name that already exists is refused.
Saving config.yaml first copies the old version to .work/config-history/. Runs on 127.0.0.1 only.
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np

import ascii_film as af
import dashboard as ascii_dashboard          # the ./ascii dashboard: reuse its HTTP plumbing (video range requests)

from . import analysis, config, photos, plot, render, score, sets, verify

WEB = Path(__file__).resolve().parent / "web"
ROOT = Path(".")                              # set by serve()
CONFIG_ARG: str | None = None


# --------------------------------------------------------------------------- jobs (one at a time)

class _Lines(io.TextIOBase):
    """A text stream that appends every printed line to a job's log."""

    def __init__(self, log: list):
        self.log, self.buf = log, ""

    def writable(self):
        return True

    def write(self, s):
        self.buf += s
        *lines, self.buf = re.split(r"[\r\n]", self.buf)
        self.log.extend(x for x in lines if x.strip())
        return len(s)

    def close_out(self):
        if self.buf.strip():
            self.log.append(self.buf)
        self.buf = ""


class Jobs:
    """A queue run by one background thread: a render already uses every core. What a job prints goes to
    its log (only this thread redirects stdout, so the request threads never touch it)."""

    def __init__(self):
        self.items: list[dict] = []
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kind: str, label: str, fn) -> dict:
        job = dict(id=uuid.uuid4().hex[:8], kind=kind, label=label, status="queued", progress=0.0, log=[],
                   result=None, error="", created=time.time(), started=None, finished=None, _fn=fn)
        with self.cv:
            live = [j for j in self.items if j["status"] in ("queued", "running")]
            old = [j for j in self.items if j["status"] not in ("queued", "running")][-15:]
            self.items = sorted(old + live, key=lambda j: j["created"]) + [job]
            self.cv.notify()
        return {"job": job["id"]}

    def active(self, kind: str) -> bool:
        with self.cv:
            return any(j["kind"] == kind and j["status"] in ("queued", "running") for j in self.items)

    def busy(self) -> bool:
        with self.cv:
            return any(j["status"] in ("queued", "running") for j in self.items)

    def public(self) -> list[dict]:
        with self.cv:
            return [{k: (list(v[-300:]) if k == "log" else v) for k, v in j.items() if not k.startswith("_")}
                    for j in self.items]

    def _run(self):
        while True:
            with self.cv:
                job = None
                while job is None:
                    job = next((j for j in self.items if j["status"] == "queued"), None)
                    if job is None:
                        self.cv.wait()
                job["status"], job["started"] = "running", time.time()
            out = _Lines(job["log"])
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    job["result"] = job["_fn"](job)
                job["status"], job["progress"] = "done", 1.0
            except (Exception, SystemExit) as e:          # SystemExit: the friendly errors the CLI prints
                job["status"], job["error"] = "error", (str(e) or e.__class__.__name__)[-900:]
                if not isinstance(e, SystemExit):
                    traceback.print_exc()
            out.close_out()
            job["finished"] = time.time()


JOBS = Jobs()


# --------------------------------------------------------------------------- reading the project

_memo: dict = {}


def _cached(key, make):
    """One cached value per name, recomputed when `key` changes (file mtimes etc)."""
    name = key[0]
    hit = _memo.get(name)
    if hit is None or hit[0] != key:
        hit = (key, make())
        _memo[name] = hit
    return hit[1]


def _mtime(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def load_cfg() -> tuple[dict, Path, list[str]]:
    path = config.config_path(ROOT, CONFIG_ARG)

    def make():
        warns: list[str] = []
        return config.parse(path.read_text(), warn=warns.append), path, warns
    return _cached(("cfg", str(path), _mtime(path), _mtime(config.DEFAULTS_FILE)), make)


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def url(p: Path) -> str:
    return f"/file/{quote(rel(p))}?v={_mtime(p) // 1_000_000}"


def _track(cfg) -> dict:
    try:
        t = analysis.find_track(ROOT, cfg)
    except SystemExit as e:
        return {"missing": str(e), "expected": cfg["paths"]["audio"]}
    return {"name": rel(t), "bytes": t.stat().st_size, "url": url(t), "path": t}


def _analysis(cfg, track: dict) -> dict | None:
    p = ROOT / cfg["paths"]["analysis"]
    if not p.exists():
        return None
    try:
        data = _cached(("an_json", str(p), _mtime(p)), lambda: json.loads(p.read_text()))
        an = {"duration": data["duration"], "frames": data["n_frames"], "onsets": len(data["onsets"]),
              "fps": data["fps"], "track": Path(data["track"]).name}
    except (json.JSONDecodeError, KeyError, OSError):
        return {"unreadable": True, "fresh": False}
    t = track.get("path")
    if t is not None:
        key = _cached(("an_key", str(t), _mtime(t), json.dumps(cfg["analysis"], sort_keys=True), cfg["video"]["fps"]),
                      lambda: analysis.cache_key(t, cfg))
        an["fresh"] = data.get("key") == key
    else:
        an["fresh"] = False
    return an


def _pool_files(cfg, pool: str) -> list[Path]:
    d = ROOT / cfg["paths"]["images"] / pool
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in af.IMAGE_EXTS)


def _pools(cfg) -> dict:
    out = {}
    for pool in score.POOLS:
        files = _pool_files(cfg, pool)
        out[pool] = {"count": len(files), "folder": f"{cfg['paths']['images']}/{pool}",
                     "files": [{"name": p.name, "thumb": f"/thumb/{pool}/{quote(p.name)}?v={_mtime(p) // 1_000_000}"}
                               for p in files]}
    return out


def _verify_info(video: Path) -> dict | None:
    p = render.sidecar(video, ".verify.json")
    if not p.exists():
        return None

    def make():
        rep = json.loads(p.read_text())
        return {"passed": rep["passed"], "summary": verify.summary(rep), "ok": rep.get("ok", {})}
    try:
        v = dict(_cached(("verify:" + str(p), _mtime(p)), make))
    except (json.JSONDecodeError, KeyError, OSError):
        return None
    v["stale"] = _mtime(p) < _mtime(video)                  # the video was re-rendered after this check
    return v


def _renders(cfg) -> list[dict]:
    d = ROOT / cfg["paths"]["renders"]
    if not d.is_dir():
        return []
    vids = sorted((p for p in d.glob("*.mp4") if not p.name.startswith(".")), key=_mtime, reverse=True)
    out = []
    for v in vids:
        e = {"name": v.name, "url": url(v), "bytes": v.stat().st_size, "mtime": _mtime(v) / 1e9}
        mp = render.sidecar(v, ".json")
        if mp.exists():
            try:
                m = json.loads(mp.read_text())
                e["meta"] = {k: m.get(k) for k in ("seed", "frames", "fps", "preview", "debug", "cuts", "width",
                                                   "height", "seconds", "fps_rendered")}
            except (json.JSONDecodeError, OSError):
                pass
        e["verify"] = _verify_info(v)
        c = v.with_name(v.stem + "_contact.png")
        if c.exists():
            e["contact"] = url(c)
        out.append(e)
    return out


def state() -> dict:
    out = {"root": str(ROOT), "jobs": JOBS.public(), "busy": JOBS.busy()}
    try:
        cfg, cpath, warns = load_cfg()
    except (SystemExit, OSError) as e:
        out["config"] = {"path": str(config.config_path(ROOT, CONFIG_ARG)), "error": str(e)}
        return out
    track = _track(cfg)
    out["config"] = {"path": str(cpath), "warnings": warns, "shared": cpath.parent != ROOT,
                     "builtin": cpath == config.DEFAULTS_FILE, "mtime": _mtime(cpath)}
    out["fps"] = cfg["video"]["fps"]
    out["analysis"] = _analysis(cfg, track)
    track.pop("path", None)
    out["track"] = track
    out["pools"] = _pools(cfg)
    out["renders"] = _renders(cfg)
    out["plots"] = [{"name": p.name, "url": url(p)}
                    for p in sorted(ROOT.glob("analysis*.png"), key=lambda p: (p.name != "analysis.png", p.name))]
    return out


_THUMBS: OrderedDict = OrderedDict()
_THUMB_LOCK = threading.Lock()


def find_photo(cfg: dict, where: str, name: str) -> Path:
    """images/<where>/<name>; where = 'any' looks in every folder (a photo the sorter has moved since)."""
    name = Path(name).name
    for w in (photos.FOLDERS if where == "any" else (where,)):
        if w not in photos.FOLDERS:
            break
        p = ROOT / cfg["paths"]["images"] / w / name
        if p.is_file() and p.suffix.lower() in af.IMAGE_EXTS and not name.startswith("."):
            return p
    raise FileNotFoundError(f"{where}/{name}")


def thumb(where: str, name: str, h: int = 120, crop: str = "", rot: float = 0.0) -> bytes:
    """A small JPEG of an input photo. crop='film': only the part the film shows (your rotation and crop, or
    the centre 16:9). rot: the whole photo turned clockwise (for the crop editor and the photo panel)."""
    from .frames import apply_view, rotate
    cfg, _, _ = load_cfg()
    src = find_photo(cfg, where, name)
    h = max(40, min(1600, int(h)))
    hand = photos.crop_of(ROOT, cfg, src) if crop == "film" else None
    key = (str(src), _mtime(src), h, crop, json.dumps(hand, sort_keys=True), round(rot, 2))
    with _THUMB_LOCK:
        if key in _THUMBS:
            _THUMBS.move_to_end(key)
            return _THUMBS[key]
    flag = cv2.IMREAD_REDUCED_COLOR_4 if h <= 300 else cv2.IMREAD_COLOR
    img = cv2.imread(str(af.readable_image(src, ROOT / ".work")), flag)
    if img is None:
        raise ValueError(f"can't read {where}/{name}")
    if hand:                                              # your rotation and crop, as the renderer uses them
        img = apply_view(img, hand)
    elif rot:
        img = rotate(img, rot)
    if crop == "film":                                    # the same centre 16:9 crop as the renderer
        ih, iw = img.shape[:2]
        if iw / ih > 16 / 9:
            cw = round(ih * 16 / 9)
            img = img[:, (iw - cw) // 2:(iw - cw) // 2 + cw]
        else:
            ch = round(iw * 9 / 16)
            img = img[(ih - ch) // 2:(ih - ch) // 2 + ch]
    img = cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)
    body = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    with _THUMB_LOCK:
        _THUMBS[key] = body
        while len(_THUMBS) > 4000:
            _THUMBS.popitem(last=False)
    return body


def project_file(rel_path: str) -> Path:
    """A file inside the project folder (not hidden), for the player, plots and contact sheets."""
    p = (ROOT / rel_path).resolve()
    try:
        parts = p.relative_to(ROOT).parts
    except ValueError:
        raise FileNotFoundError(rel_path)
    if not parts or any(x.startswith(".") for x in parts) or not p.is_file():
        raise FileNotFoundError(rel_path)
    return p


# --------------------------------------------------------------------------- adding inputs

class Refused(Exception):
    def __init__(self, msg: str, code: int = 409):
        super().__init__(msg)
        self.code = code


class Conflict(Refused):
    """Adding this file would replace `replaces`; done only when the request says replace."""

    def __init__(self, dest: Path, replaces: list[Path]):
        super().__init__(f"{rel(dest)} would replace {', '.join(rel(x) for x in replaces)}")
        self.dest, self.replaces = dest, replaces


def _track_files(cfg: dict) -> list[Path]:
    """Every file that counts as the soundtrack (audio/track.wav, .flac, ...)."""
    want = ROOT / cfg["paths"]["audio"]
    if want.suffix.lower() in analysis.AUDIO_EXTS:
        return [want] if want.exists() else []
    if not want.parent.is_dir():
        return []
    names = {(want.name + ext).lower() for ext in analysis.AUDIO_EXTS}
    # the real directory listing: the Mac's disk ignores case, so track.wav and track.WAV are one file
    return sorted(p for p in want.parent.iterdir() if p.name.lower() in names and p.is_file())


def input_file(cfg: dict, ref: str) -> Path:
    """'bright/IMG_1.jpg' -> that file in images/, or FileNotFoundError."""
    pool, _, name = str(ref).partition("/")
    p = ROOT / cfg["paths"]["images"] / pool / Path(name).name
    if pool not in score.POOLS or not p.is_file():
        raise FileNotFoundError(f"{ref} is not in {cfg['paths']['images']}/")
    return p


def plan(cfg: dict, to: str, name: str, old: str | None = None) -> tuple[Path, list[Path]]:
    """Where a new file goes, and which existing files it would replace.
    Track: audio/track.<ext> replaces any current track. Image: images/<pool>/<name> replaces a file of the
    same name, and `old` (e.g. 'bright/IMG_1.jpg') when the new photo is meant to take that one's place."""
    name = Path(name).name.strip()
    ext = Path(name).suffix.lower()
    if to == "audio":
        if ext not in analysis.AUDIO_EXTS:
            raise Refused(f"{name}: not an audio file this reads ({', '.join(analysis.AUDIO_EXTS)})", 415)
        want = ROOT / cfg["paths"]["audio"]
        dest = want if want.suffix.lower() in analysis.AUDIO_EXTS else want.with_name(want.name + ext)
        if dest.suffix.lower() != ext:
            raise Refused(f"config paths.audio wants {rel(dest)}; this is a {ext} file", 415)
        return dest, _track_files(cfg)
    if to in score.POOLS or to == "inbox":
        if not name or name.startswith(".") or ext not in af.IMAGE_EXTS:
            raise Refused(f"{name or 'file'}: not an image this reads ({', '.join(sorted(af.IMAGE_EXTS))})", 415)
        dest = ROOT / cfg["paths"]["images"] / to / name
        replaces = [dest] if dest.exists() else []
        if old:
            o = input_file(cfg, old)
            if o not in replaces:
                replaces.append(o)
        return dest, replaces
    raise Refused(f"unknown destination {to!r}", 400)


def retire(p: Path, stamp: str) -> Path:
    """Move a replaced file to .work/replaced/<stamp>/<its path>: out of the film, not deleted."""
    dest = ROOT / ".work" / "replaced" / stamp / p.resolve().relative_to(ROOT)
    n = 1
    while dest.exists():
        n += 1
        dest = dest.with_name(f"{p.stem}-{n}{p.suffix}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.rename(p, dest)
    return dest


def add_file(to: str, name: str, write, replace: bool = False, old: str | None = None) -> dict:
    """Add a file; write(f) fills it. If it would replace files, raise Conflict unless `replace` is set;
    then the old files are moved to .work/replaced/ first (never deleted)."""
    cfg, _, _ = load_cfg()
    dest, olds = plan(cfg, to, name, old)
    if olds and not replace:
        raise Conflict(dest, olds)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".adding-{uuid.uuid4().hex[:8]}-{dest.name}")
    kept: list[tuple[Path, Path]] = []
    try:
        with open(tmp, "wb") as f:
            write(f)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            for o in olds:
                kept.append((o, retire(o, stamp)))
            os.link(tmp, dest)                              # fails instead of overwriting
        except BaseException:
            for o, k in reversed(kept):                     # any failure: put the old files back
                os.rename(k, o)
            raise
    except FileExistsError:
        raise Refused(f"{rel(dest)} appeared meanwhile; nothing was overwritten.")
    finally:
        tmp.unlink(missing_ok=True)
    return {"saved": rel(dest), "kept": [rel(k) for _, k in kept]}


CHOOSER = """
activate
set picked to choose file with prompt "{prompt}"{multiple}
if class of picked is not list then set picked to {{picked}}
set out to ""
repeat with f in picked
    set out to out & POSIX path of f & linefeed
end repeat
return out
"""


def choose_files(prompt: str, multiple: bool) -> list[Path] | None:
    """The macOS file chooser, opened by the server. Works in any browser, including ones that can't take
    a drag from Finder (like the Claude app's browser pane). None if cancelled."""
    script = CHOOSER.format(prompt=prompt.replace('"', "'"),
                            multiple=" with multiple selections allowed" if multiple else "")
    r = subprocess.run(["osascript", "-"], input=script, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        if "-128" in r.stderr:                              # the Cancel button
            return None
        raise RuntimeError(f"the file chooser failed: {r.stderr.strip()}")
    return [Path(x) for x in r.stdout.splitlines() if x.strip()]


def _copier(src: Path):
    def write(f):
        with open(src, "rb") as fin:
            shutil.copyfileobj(fin, f, 4 << 20)
    return write


PENDING: dict[str, dict] = {}       # chooser picks waiting for "replace?" to be confirmed on the page


def _add_paths(to: str, paths: list[Path], replace: bool, old: str | None) -> dict:
    out = {"saved": [], "kept": [], "errors": [], "conflicts": []}
    for src in paths:
        try:
            if not src.is_file():
                raise Refused(f"{src.name}: not a file")
            r = add_file(to, src.name, _copier(src), replace=replace, old=old)
            out["saved"].append(r["saved"])
            out["kept"] += r["kept"]
        except Conflict as c:
            out["conflicts"].append({"name": src.name, "src": src, "replaces": [rel(x) for x in c.replaces]})
        except Refused as e:
            out["errors"].append(str(e))
    return out


def act_choose(req: dict) -> dict:
    """Pick files in the macOS chooser and add them. Ones that would replace something come back as
    `conflicts` with a token; the page asks, then sends the token to act_confirm."""
    to, old = str(req.get("to", "")), req.get("old") or None
    if to != "audio" and to not in score.POOLS and to != "inbox":
        raise ValueError(f"unknown destination {to!r}")
    if old:
        cfg, _, _ = load_cfg()
        input_file(cfg, old)
        what = f"a photo to replace {old}"
    elif to == "audio":
        what = "the soundtrack (wav, flac, mp3, m4a, aiff)"
    else:
        what = f"images for {to} (Cmd-click or Cmd-A to pick several)"
    paths = choose_files(f"Choose {what}", multiple=to != "audio" and not old)
    if paths is None:
        return {"cancelled": True, "saved": [], "kept": [], "errors": [], "conflicts": []}
    out = _add_paths(to, paths, replace=False, old=old)
    if out["conflicts"]:
        token = uuid.uuid4().hex
        for k in [k for k, v in PENDING.items() if time.time() - v["t"] > 1800]:
            PENDING.pop(k, None)
        PENDING[token] = {"t": time.time(), "to": to, "old": old, "paths": [c["src"] for c in out["conflicts"]]}
        out["token"] = token
    for c in out["conflicts"]:
        c.pop("src")
    return out


def act_confirm(req: dict) -> dict:
    """The page said yes: replace with the files picked in act_choose."""
    job = PENDING.pop(str(req.get("token", "")), None)
    if job is None:
        raise ValueError("That choice has expired; pick the files again.")
    out = _add_paths(job["to"], job["paths"], replace=True, old=job["old"])
    for c in out["conflicts"]:
        c.pop("src")
    return out


# --------------------------------------------------------------------------- actions

def _video(name: str) -> Path:
    cfg, _, _ = load_cfg()
    p = ROOT / cfg["paths"]["renders"] / Path(str(name)).name
    if not p.is_file() or p.suffix != ".mp4":
        raise FileNotFoundError(f"{name} is not in {cfg['paths']['renders']}/")
    return p


def _fresh_cfg() -> dict:
    """Each job reads config.yaml again, so a save applies to the next render."""
    return config.load(ROOT, CONFIG_ARG)[0]


def act_analyze(req: dict) -> dict:
    force = bool(req.get("force"))

    def fn(job):
        cfg = _fresh_cfg()
        an = analysis.load_or_analyze(ROOT, cfg, force=force)
        if not force:
            print(f"analysis.json is up to date: {an.duration:.2f} s, {an.n} frames, {len(an.onset_frames)} onsets")
        plot.for_root(ROOT, cfg, an)
    return JOBS.submit("analyze", "re-analyse the track" if force else "analyse the track", fn)


def act_plot(req: dict) -> dict:
    span = req.get("span")
    if span:
        span = (float(span[0]), float(span[1]))
        if not span[1] > span[0]:
            raise ValueError("plot range: the end must be after the start")

    def fn(job):
        cfg = _fresh_cfg()
        out = plot.for_root(ROOT, cfg, analysis.load_or_analyze(ROOT, cfg), span)
        return {"plot": out.name}
    return JOBS.submit("plot", f"plot {span[0]:g}-{span[1]:g} s" if span else "plot the whole track", fn)


def act_render(req: dict) -> dict:
    preview = req.get("preview")
    if preview:
        preview = (float(preview[0]), float(preview[1]))
        if preview[0] < 0 or not preview[1] > preview[0]:
            raise ValueError("preview: the end must be after the start")
    seed = req.get("seed")
    seed = None if seed in (None, "") else int(seed)
    debug_, check = bool(req.get("debug")), bool(req.get("verify", True))
    label = (f"preview {preview[0]:g}-{preview[1]:g} s" if preview else "full film") + \
        (" with debug overlay" if debug_ else "") + (f", seed {seed}" if seed is not None else "")

    def fn(job):
        cfg = _fresh_cfg()
        share = 0.9 if check else 1.0

        def progress(done, total):
            job["progress"] = share * done / max(1, total)
        meta = render.render(ROOT, cfg, seed=seed, debug_=debug_, preview=preview, progress=progress)
        video = ROOT / cfg["paths"]["renders"] / meta["video"]
        plot.for_root(ROOT, cfg, analysis.load_or_analyze(ROOT, cfg))      # keep analysis.png in step
        if check:
            print(verify.summary(verify.verify(video, ROOT, cfg)))
        return {"video": meta["video"]}
    return JOBS.submit("render", label, fn)


def act_verify(req: dict) -> dict:
    video = _video(req.get("video", ""))

    def fn(job):
        print(verify.summary(verify.verify(video, ROOT, _fresh_cfg())))
        return {"video": video.name}
    return JOBS.submit("verify", f"verify {video.name}", fn)


def act_stills(req: dict) -> dict:
    video = _video(req.get("video", ""))

    def fn(job):
        out = verify.stills(video)
        print(f"wrote {rel(out)}")
        return {"video": video.name, "contact": url(out)}
    return JOBS.submit("stills", f"stills from {video.name}", fn)


def _config_file() -> tuple[Path, str]:
    """The config.yaml the project uses, and its text (the built-in defaults if there's none yet)."""
    path = config.config_path(ROOT, CONFIG_ARG)
    return path, path.read_text()


def _write_config(text: str) -> dict:
    """Save config.yaml (validated by the caller). The old version goes to .work/config-history/ next to it."""
    path = config.config_path(ROOT, CONFIG_ARG)
    if path == config.DEFAULTS_FILE:                      # no config.yaml yet: never edit the built-in defaults
        path = ROOT / "config.yaml"
    backup = None
    if path.exists():
        if path.read_text() == text:
            return {"saved": str(path), "unchanged": True}
        hist = path.parent / ".work" / "config-history"
        hist.mkdir(parents=True, exist_ok=True)
        stamp, n = f"{datetime.now():%Y%m%d-%H%M%S}", 1
        backup = hist / f"{path.stem}-{stamp}.yaml"
        while backup.exists():                            # two saves in one second: keep both old versions
            n += 1
            backup = hist / f"{path.stem}-{stamp}-{n}.yaml"
        backup.write_text(path.read_text())
    tmp = path.with_name(f".{path.name}.saving")
    tmp.write_text(text)
    os.replace(tmp, path)
    return {"saved": str(path), "backup": str(backup) if backup else None}


def act_save_config(req: dict) -> dict:
    """The config.yaml editor's Save: the whole text."""
    text = str(req.get("text", ""))
    warns: list[str] = []
    config.parse(text, warn=warns.append)                 # SystemExit (-> 400) if it can't be used
    return dict(_write_config(text), warnings=warns)


def _changes(req: dict) -> dict:
    ch = req.get("changes") or {}
    if not isinstance(ch, dict) or not all(isinstance(k, str) and re.fullmatch(r"[\w.-]+", k) for k in ch):
        raise ValueError("changes must be {\"section.key\": value}")
    return ch


def act_tune(req: dict) -> dict:
    """The Tune panel's Save: set single values in config.yaml, comments and layout untouched."""
    _, text = _config_file()
    new, _ = config.set_values(text, _changes(req))       # SystemExit (-> 400) if a value doesn't fit
    return _write_config(new)


# ---- what-if: the film's numbers and a few frames for settings that aren't saved yet ----------------------

_SIM_LOCK = threading.Lock()
_SIM: dict = {}                     # cached analysis and frame renderer between what-if requests


def _sim_analysis(cfg: dict):
    """The analysis for these settings: analysis.json if it matches, else computed in memory (not saved)."""
    track = analysis.find_track(ROOT, cfg)
    want = json.dumps([str(track), _mtime(track), cfg["analysis"], cfg["video"]["fps"]], sort_keys=True)
    if _SIM.get("an_want") != want:
        data, p = None, ROOT / cfg["paths"]["analysis"]
        if p.exists():
            try:
                d = json.loads(p.read_text())
                data = d if d.get("key") == analysis.cache_key(track, cfg) else None
            except (json.JSONDecodeError, OSError):
                pass
        if data is None:
            data = analysis.analyze(track, cfg, log=lambda *_: None)
        _SIM.update(an_want=want, an=analysis.Analysis(data))
    return _SIM["an"]


def score_stats(sc) -> dict:
    rows = sc.rows
    n = max(1, len(rows))
    kinds = {k: sum(r["kind"] == k for r in rows) / n for k in ("cut", "hold", "decay", "black", "silence")}
    cuts = [r for r in rows if r["kind"] == "cut"]
    vis = [r for r in rows if r["kind"] in ("cut", "hold", "decay")]
    nv = max(1, len(vis))
    cols = sorted(r["cols"] for r in vis) or [0]
    return dict(
        frames=len(rows), cuts=len(cuts), flashes=sum(r["flash"] for r in rows), kinds=kinds,
        pools={p: sum(r["pool"] == p for r in cuts) / max(1, len(cuts)) for p in score.POOLS},
        bands={b["name"]: sum(r["band"] == i for r in vis) / nv for i, b in enumerate(sc.bands)},
        cols=dict(low=cols[len(cols) // 20], median=cols[len(cols) // 2], high=cols[-1 - len(cols) // 20]),
        colour=dict(mono=sum(r["mix"] < 0.25 for r in vis) / nv, colour=sum(r["mix"] > 0.75 for r in vis) / nv),
        layered=sum(bool(r["layer"]) for r in vis) / nv,
        photos=len({r["name"] for r in cuts}), warnings=sc.warnings)


def _look(cfg: dict, sc, t: float, views: dict, n: int = 4) -> str | None:
    """The first n cut frames from t seconds, 960x540 each, tiled 2x2 with a label: a JPEG data URL."""
    import base64
    from .frames import FrameRenderer
    fps = cfg["video"]["fps"]
    k0 = int(round(max(0.0, t) * fps))
    picks = [r for r in sc.rows[k0:] if r["kind"] == "cut"][:n]
    if not picks:
        return None
    W, H = 960, 540
    key = config.digest([cfg["ascii"], cfg["treatments"], cfg["editing"]["flash"], cfg["seed"],
                         [b["chars"] for b in sc.bands], views])
    if _SIM.get("fr_key") != key:
        _SIM.update(fr_key=key, fr=FrameRenderer(cfg, sc.bands, W, H, af.find_font(), int(cfg["seed"]), views=views))
    fr = _SIM["fr"]
    tiles = []
    for r in picks:
        img = fr.render(r)
        lab = f"{r['frame'] / fps:7.2f}s  {r['pool']}  {sc.bands[r['band']]['name']}  {r['cols']} cols" + \
              ("  FLASH" if r["flash"] else "")
        cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), -1)
        cv2.putText(img, lab, (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 230), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) < n:
        tiles.append(np.zeros((H, W, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, n, 2)])
    cv2.line(grid, (W, 0), (W, grid.shape[0]), (60, 60, 60), 2)
    cv2.line(grid, (0, H), (grid.shape[1], H), (60, 60, 60), 2)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def act_simulate(req: dict) -> dict:
    """What the film would look like with these (unsaved) changes: numbers for the whole film, plus the
    first cuts from the preview start, drawn for real."""
    _, text = _config_file()
    ch = _changes(req)
    cfg = config.set_values(text, ch)[1] if ch else config.parse(text, warn=lambda *_: None)
    with _SIM_LOCK:
        an = _sim_analysis(cfg)
        pools = score.gather_pools(ROOT, cfg, ROOT / ".work")
        views = photos.views_for(ROOT, cfg, pools)
        sc = score.build_score(an, cfg, pools, int(cfg["seed"]), views, sets.sections_for(ROOT, cfg, pools, an.fps, an.n))
        out = {"stats": score_stats(sc), "onsets": len(an.onset_frames), "duration": an.duration,
               "warnings": sc.warnings}
        out["stats"]["usage"] = score.pool_usage(sc, pools, an.fps)["pools"]
        out["stats"]["sections"] = sc.sections
        if req.get("look", True):
            out["look"] = _look(cfg, sc, float(req.get("t") or 0), views)
    return out


def act_reveal(req: dict) -> dict:
    """Open the project folder (or its renders folder) in Finder."""
    cfg, _, _ = load_cfg()
    p = ROOT / cfg["paths"]["renders"] if req.get("what") == "renders" else ROOT
    subprocess.run(["open", str(p if p.is_dir() else ROOT)], check=False)
    return {"ok": True}


# ---- the photo sorter (/sort) and the photo panel beside the player ------------------------------------

def photo_listing() -> dict:
    cfg, _, _ = load_cfg()
    out = photos.listing(ROOT, cfg)
    if out["unscored"] and not JOBS.active("score"):
        JOBS.submit("score", f"score {out['unscored']} photos for suggestions", _score_job)
    out["scoring"] = JOBS.active("score")
    out["exts"] = sorted(af.IMAGE_EXTS)
    return out


def _score_job(job):
    cfg = _fresh_cfg()
    todo = photos.unscored(ROOT, cfg)
    n = photos.score_photos(ROOT, todo, progress=lambda d, t: job.__setitem__("progress", d / max(1, t)))
    print(f"scored {n} photos")


def act_move(req: dict) -> dict:
    refs = req.get("refs") or []
    if not isinstance(refs, list):
        raise ValueError("refs must be a list")
    return photos.move(ROOT, load_cfg()[0], [str(r) for r in refs], str(req.get("to", "")))


def act_crop(req: dict) -> dict:
    """{'ref': 'bright/IMG_1.jpg', 'rect': {x, y, w, h}, 'rot': degrees} saves how you framed it;
    rect null and rot 0 go back to the default (the centre)."""
    rect = req.get("rect")
    return {"crop": photos.save_crop(ROOT, load_cfg()[0], str(req.get("ref", "")), rect if rect else None,
                                     float(req.get("rot") or 0))}


def act_import(req: dict) -> dict:
    """Import from the Photos app: {'selection': true} or {'album_id': ..., 'album_name': ...}."""
    source = {"album_id": str(req["album_id"])} if req.get("album_id") else {"selection": True}
    label = f"import album {req.get('album_name') or ''}".strip() if req.get("album_id") else "import Photos selection"

    def fn(job):
        cfg = _fresh_cfg()
        r = photos.import_from_photos(ROOT, cfg, source,
                                      progress=lambda d, t: job.__setitem__("progress", 0.8 * d / max(1, t)))
        print(f"added {len(r['added'])} photos to the inbox")
        todo = photos.unscored(ROOT, cfg)
        if todo:
            photos.score_photos(ROOT, todo, progress=lambda d, t: job.__setitem__("progress", 0.8 + 0.2 * d / max(1, t)))
        return {"added": len(r["added"])}
    return JOBS.submit("import", label, fn)


_USAGE: dict = {}


def usage_report() -> dict:
    """How the next render would use the photos (saved settings, current pools and crops)."""
    cfg, cpath, _ = load_cfg()
    base = ROOT / cfg["paths"]["images"]
    files = sorted(f"{p.parent.name}/{p.name}" for pool in score.POOLS for p in photos.photos_in(base / pool))
    try:
        track = analysis.find_track(ROOT, cfg)
    except SystemExit as e:
        return {"error": str(e)}
    if not files:
        return {"error": "No photos in the pools yet."}
    key = json.dumps([str(cpath), _mtime(cpath), _mtime(base / "crops.json"), sets.mtime(ROOT, cfg), str(track),
                      _mtime(track), files])
    if _USAGE.get("key") != key:
        with _SIM_LOCK:
            an = _sim_analysis(cfg)
            pools = score.gather_pools(ROOT, cfg, ROOT / ".work")
            sc = score.build_score(an, cfg, pools, int(cfg["seed"]), photos.views_for(ROOT, cfg, pools),
                                   sets.sections_for(ROOT, cfg, pools, an.fps, an.n))
            u = score.pool_usage(sc, pools, an.fps)
            u["sections"], u["warnings"] = sc.sections, sc.warnings
        u["no_repeat_within"] = cfg["editing"]["no_repeat_within"]
        _USAGE.update(key=key, value=u)
    return _USAGE["value"]


# ---- photo sets and track sections (soundfilm/sets.py) ----------------------------------------------------

_ENV: dict = {}
_SUGGEST: dict = {}


def _envelope(an, cfg: dict, bins: int = 720) -> dict:
    """The track at a glance for the sections timeline: per bin, the loudest moment (0-1) and the pool the
    sound asks for there (b / t / d, '' when silent)."""
    pl, sil = cfg["editing"]["pools"], cfg["editing"]["silence"]["threshold_db"]
    key = (id(an), pl["bright_above"], pl["dark_below"], sil, bins)
    if _ENV.get("key") != key:
        rms, db, cent, e = an.f["rms_norm"], an.f["rms_db"], an.f["centroid"], an.f["rms"]
        edges = np.linspace(0, an.n, min(bins, an.n) + 1).astype(int)
        lvl, want = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            b = max(b, a + 1)
            lvl.append(round(float(rms[a:b].max()), 3))
            if db[a:b].max() < sil:
                want.append("")
                continue
            w = e[a:b]
            c = float(np.average(cent[a:b], weights=w)) if w.sum() > 0 else float(cent[a:b].mean())
            want.append("b" if c >= pl["bright_above"] else "d" if c < pl["dark_below"] else "t")
        _ENV.update(key=key, value={"level": lvl, "want": "".join(w or "-" for w in want)})
    return _ENV["value"]


def sets_state() -> dict:
    """Sets (with how many photos each has per folder), sections, and the track for the timeline."""
    cfg, _, _ = load_cfg()
    d = sets.load(ROOT, cfg)
    cnt = sets.counts(ROOT, cfg, d)
    out = {"sets": [dict(s, counts=cnt.get(s["name"], {})) for s in d["sets"]], "none": cnt[""],
           "sections": d["sections"], "duration": None}
    try:
        an = _sim_analysis(cfg)
        out.update(duration=an.duration, fps=an.fps, onsets=[round(float(t), 3) for t in an.onset_times],
                   env=_envelope(an, cfg))
    except (SystemExit, FileNotFoundError, RuntimeError) as e:
        out["error"] = str(e)
    return out


def act_sets(req: dict) -> dict:
    """{'op': 'create' | 'rename' | 'recolor' | 'remove', 'name', 'to', 'color'}"""
    cfg, _, _ = load_cfg()
    op, name = req.get("op"), str(req.get("name") or "")
    if op == "create":
        sets.create(ROOT, cfg, name)
    elif op == "rename":
        sets.rename(ROOT, cfg, name, str(req.get("to") or ""))
    elif op == "recolor":
        sets.recolor(ROOT, cfg, name, str(req.get("color") or ""))
    elif op == "remove":
        sets.remove(ROOT, cfg, name)
    else:
        raise ValueError("op must be create, rename, recolor or remove")
    return sets_state()


def act_members(req: dict) -> dict:
    """{'refs': ['bright/IMG_1.jpg', ...], 'set': name, 'on': true (add) / false (take out)}"""
    refs = req.get("refs") or []
    if not isinstance(refs, list):
        raise ValueError("refs must be a list")
    return sets.set_members(ROOT, load_cfg()[0], [str(r) for r in refs], str(req.get("set") or ""),
                            bool(req.get("on", True)))


def act_sections(req: dict) -> dict:
    """{'sections': [{start, sets, any_pool, label}, ...]}: replaces the sections."""
    cfg, _, _ = load_cfg()
    dur = None
    with contextlib.suppress(Exception):
        dur = _sim_analysis(cfg).duration
    sets.save_sections(ROOT, cfg, req.get("sections") or [], dur)
    return sets_state()


def act_suggest(req: dict) -> dict:
    """{'n': sections} -> {'starts': [seconds]}: where the track changes character (nothing is saved)."""
    cfg, _, _ = load_cfg()
    n = int(req.get("n") or 4)
    if not 2 <= n <= 24:
        raise ValueError("ask for 2 to 24 sections")
    an = _sim_analysis(cfg)
    key = (str(an.track), _mtime(an.track), n)
    if key not in _SUGGEST:
        _SUGGEST[key] = analysis.suggest_sections(an.track, n, an.onset_times)
    return {"starts": _SUGGEST[key]}


def edl_frames(video: str) -> dict:
    """Per frame of a render: which photo is on screen and which part of it (for the photo panel)."""
    v = _video(video)
    p = render.sidecar(v, ".edl.csv")
    if not p.exists():
        raise FileNotFoundError(f"{p.name} is missing")
    meta = json.loads(render.sidecar(v, ".json").read_text()) if render.sidecar(v, ".json").exists() else {}
    rows = score.read_edl(p)
    return {"fps": meta.get("fps", 24), "first_frame": int(rows[0]["frame"]) if rows else 0,
            "name": [r["name"] for r in rows], "kind": [r["kind"] for r in rows],
            "crop": [r.get("crop", "") for r in rows], "view": [r.get("view", "") for r in rows],
            "layer": [r.get("layer_name", "") for r in rows],
            "flash": [r["flash"] == "1" for r in rows]}


ACTIONS = {"analyze": act_analyze, "plot": act_plot, "render": act_render, "verify": act_verify,
           "stills": act_stills, "config": act_save_config, "reveal": act_reveal, "choose": act_choose,
           "confirm": act_confirm, "tune": act_tune, "simulate": act_simulate, "move": act_move,
           "import": act_import, "crop": act_crop, "sets": act_sets, "members": act_members,
           "sections": act_sections, "suggest": act_suggest}


# --------------------------------------------------------------------------- HTTP

class Handler(ascii_dashboard.Handler):
    server_version = "soundfilm"

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(state())
            if path == "/api/config":
                p, text = _config_file()
                try:
                    values = config.parse(text, warn=lambda *_: None)
                except SystemExit:
                    values = None                         # the editor still opens, to fix it
                return self._json({"path": str(p), "text": text, "mtime": _mtime(p), "values": values,
                                   "defaults": config.defaults()})
            m = re.fullmatch(r"/web/([\w-]+\.js)", path)
            if m and (WEB / m.group(1)).is_file():
                return self._send(200, (WEB / m.group(1)).read_bytes(), "text/javascript; charset=utf-8")
            if path in ("/sort", "/sort.html"):
                return self._send(200, (WEB / "sort.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/photos":
                return self._json(photo_listing())
            if path == "/api/sets":
                return self._json(sets_state())
            if path == "/api/usage":
                return self._json(usage_report())
            if path == "/api/albums":
                return self._json(photos.albums())
            if path == "/api/edl":
                return self._json(edl_frames(parse_qs(u.query).get("video", [""])[0]))
            m = re.fullmatch(r"/thumb/(\w+)/([^/]+)", path)
            if m:
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                h = q.get("h", "120")
                try:
                    rot = float(q.get("rot", 0) or 0)
                except ValueError:
                    rot = 0.0
                return self._send(200, thumb(m.group(1), m.group(2), int(h) if h.isdigit() else 120, q.get("crop", ""),
                                             rot), "image/jpeg", {"Cache-Control": "max-age=3600"})
            if path.startswith("/file/"):
                return self._file(project_file(path[len("/file/"):]))
            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._error(e, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(e, 500)

    def _cross_site(self) -> bool:
        """A request sent by some other web page (it could replace files): refuse it."""
        if self.headers.get("Sec-Fetch-Site", "same-origin") not in ("same-origin", "none"):
            return True
        origin = self.headers.get("Origin")
        return bool(origin) and urlparse(origin).netloc != self.headers.get("Host")

    def do_POST(self):
        u = urlparse(self.path)
        if self._cross_site():
            return self._json({"error": "refused: request from another site"}, 403)
        try:
            if u.path == "/api/upload":
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                return self._upload(q.get("to", ""), q.get("name", ""), q.get("replace") == "1", q.get("old"))
            m = re.fullmatch(r"/api/(\w+)", u.path)
            if m and m.group(1) in ACTIONS:
                return self._json(ACTIONS[m.group(1)](self._body_json()))
            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._error(e, 404)
        except (ValueError, TypeError, SystemExit, RuntimeError) as e:
            self._error(e, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(e, 500)

    def _upload(self, to: str, name: str, replace: bool, old: str | None):
        def write(f):
            left = int(self.headers.get("Content-Length") or 0)
            while left > 0:
                chunk = self.rfile.read(min(4 << 20, left))
                if not chunk:
                    raise ConnectionError("upload interrupted")
                f.write(chunk)
                left -= len(chunk)
        try:
            r = add_file(to, name, write, replace=replace, old=old)
        except Conflict as c:
            return self._json({"error": str(c), "conflict": True, "replaces": [rel(x) for x in c.replaces]}, 409)
        except Refused as e:
            return self._json({"error": str(e)}, e.code)
        self._json(r)


def serve(root: Path, config_arg: str | None = None, port: int = 8778, open_browser: bool = True) -> None:
    global ROOT, CONFIG_ARG
    ROOT, CONFIG_ARG = Path(root).expanduser().resolve(), config_arg
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", p), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"No free port between {port} and {port + 19}")
    httpd.daemon_threads = True
    link = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"soundfilm dashboard: {link}\nproject folder: {ROOT}\nCtrl-C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(link)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
