#!/usr/bin/env python3
"""
dashboard.py - a local browser dashboard for ascii-film. Start it with:  ./ascii dashboard

Drop iPhone clips/photos in, tune the look against a live preview frame, render, make contact
sheets and assemble the film. Runs only on 127.0.0.1; files never leave the Mac.
Never deletes or overwrites anything in input/ (an upload whose name already exists is refused).
"""
from __future__ import annotations

import http.server
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
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
import studio as st

WEB = Path(__file__).resolve().parent / "web"


# --------------------------------------------------------------------------- job queue (one at a time)

class Jobs:
    def __init__(self):
        self.items: list[dict] = []
        self.cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kind: str, label: str, fn, file: str | None = None) -> dict:
        job = {"id": uuid.uuid4().hex[:8], "kind": kind, "label": label, "file": file, "status": "queued",
               "progress": 0.0, "message": "", "result": None, "created": time.time(),
               "started": None, "finished": None, "_fn": fn, "_cancel": threading.Event()}
        with self.cv:
            self.items.append(job)
            self.items = [j for j in self.items if j["status"] in ("queued", "running")] + \
                [j for j in self.items if j["status"] not in ("queued", "running")][-60:]
            self.items.sort(key=lambda j: j["created"])
            self.cv.notify()
        return job

    def active(self, kind: str, file: str | None = None) -> dict | None:
        with self.cv:
            return next((j for j in self.items if j["kind"] == kind and j["status"] in ("queued", "running")
                         and (file is None or j["file"] == file)), None)

    def cancel(self, jid: str | None = None) -> int:
        n = 0
        with self.cv:
            for j in self.items:
                if (jid is None or j["id"] == jid) and j["status"] in ("queued", "running"):
                    if j["status"] == "queued":
                        j["status"], j["finished"] = "cancelled", time.time()
                    else:
                        j["_cancel"].set()
                    n += 1
        return n

    def public(self) -> list[dict]:
        with self.cv:
            return [{k: v for k, v in j.items() if not k.startswith("_")} for j in self.items]

    def _run(self):
        while True:
            with self.cv:
                job = None
                while job is None:
                    job = next((j for j in self.items if j["status"] == "queued"), None)
                    if job is None:
                        self.cv.wait()
                job["status"], job["started"] = "running", time.time()
            try:
                job["result"] = job["_fn"](job)
                job["status"], job["progress"] = "done", 1.0
            except af.Cancelled:
                job["status"] = "cancelled"
            except (Exception, SystemExit) as e:  # SystemExit: friendly errors raised by studio helpers
                job["status"], job["message"] = "error", (str(e) or e.__class__.__name__)[-700:]
                traceback.print_exc()
            job["finished"] = time.time()


JOBS = Jobs()


# --------------------------------------------------------------------------- helpers

def _input_file(name: str) -> Path:
    p = st.INPUT / Path(str(name)).name
    if not p.is_file() or p.name.startswith(".") or p.suffix.lower() not in st.MEDIA_EXTS:
        raise FileNotFoundError(f"{name} is not in input/")
    return p


def _settings(d: dict | None) -> dict:
    d = d or {}
    out = dict(st.RENDER_DEFAULTS)
    for k, cast in (("fps", float), ("size", int), ("photo_seconds", float)):
        if d.get(k) not in (None, ""):
            out[k] = cast(d[k])
    if out["fps"] == int(out["fps"]):
        out["fps"] = int(out["fps"])
    out["size"] = max(320, min(4096, out["size"]))
    return out


def _style(d: dict | None) -> dict:
    return af.Style({k: v for k, v in (d or {}).items() if k in af.STYLE_KINDS}).spec  # validates


def _clip_entry(p: Path) -> dict:
    stt = p.stat()
    e = {"name": p.name, "bytes": stt.st_size, "mtime": stt.st_mtime,
         "thumb": f"/thumb/{quote(p.name)}?v={int(stt.st_mtime)}"}
    try:
        info = st.probe_cached(p)
        e.update(kind=info.kind, duration=info.duration, width=info.width, height=info.height, hdr=info.hdr,
                 audio=info.audio_index is not None)
    except Exception as ex:  # unreadable file: show it with the error rather than hiding it
        e.update(kind="unknown", error=str(ex)[-300:])
    out = st.output_path(p)
    if out.exists():
        meta = st.read_meta(out) or {}
        state, why = st.render_state(p)
        o = out.stat()
        e.update(output=out.name, output_url=f"/media/output/{quote(out.name)}?v={int(o.st_mtime)}",
                 rendered_style=meta.get("style"), rendered_at=meta.get("rendered_at"),
                 rendered_settings={k: meta.get(k) for k in ("fps", "size", "photo_seconds")},
                 output_duration=meta.get("duration"), state=state, why=why)
    else:
        e.update(state="new")
    return e


def state() -> dict:
    outputs = []
    for o in st.list_outputs():
        try:
            info = st.probe_cached(o)
            outputs.append({"name": o.name, "duration": info.duration, "width": info.width, "height": info.height,
                            "url": f"/media/output/{quote(o.name)}?v={int(o.stat().st_mtime)}"})
        except Exception as ex:
            outputs.append({"name": o.name, "duration": 0, "error": str(ex)[-200:]})
    final = st.FINAL / "film.mp4"
    fin = None
    if final.exists():
        try:
            fi = st.probe_cached(final)
            fin = {"name": final.name, "duration": fi.duration, "width": fi.width, "height": fi.height,
                   "url": f"/media/final/{quote(final.name)}?v={int(final.stat().st_mtime)}",
                   "modified": datetime.fromtimestamp(final.stat().st_mtime).strftime("%b %d %H:%M")}
        except Exception:
            fin = None
    return {"root": str(st.ROOT), "clips": [_clip_entry(p) for p in st.list_inputs()], "outputs": outputs,
            "final": fin, "presets": st.load_presets(), "jobs": JOBS.public(),
            "defaults": af.DEFAULT_STYLE, "render_defaults": st.RENDER_DEFAULTS,
            "charsets": af.CHARSETS, "colors": af.NAMED_COLORS}


_FRAMES: OrderedDict = OrderedDict()
_FRAMES_LOCK = threading.Lock()


def _source_frame(p: Path, info: af.MediaInfo, t: float, aw: int, ah: int) -> np.ndarray:
    key = (str(p), p.stat().st_mtime, round(t, 2), aw, ah)
    with _FRAMES_LOCK:
        if key in _FRAMES:
            _FRAMES.move_to_end(key)
            return _FRAMES[key]
    fr = af.grab_frame(info, t, aw, ah)
    with _FRAMES_LOCK:
        _FRAMES[key] = fr
        while len(_FRAMES) > 48:
            _FRAMES.popitem(last=False)
    return fr


def frame_image(req: dict) -> tuple[bytes, str, dict]:
    p = _input_file(req.get("file", ""))
    info = st.probe_cached(p)
    settings = _settings(req.get("settings"))
    t = float(req.get("t") or 0)
    dur = info.duration if info.kind == "video" else settings["photo_seconds"]
    if req.get("mode") == "source":
        aw, ah = af.analysis_size(info.width, info.height, 480)
        img = _source_frame(p, info, t, aw, ah)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes(), "image/jpeg", {}
    style = af.Style(_style(req.get("style")))
    fs = style.at(t, dur)
    W, H = af.out_size(info.width, info.height, settings["size"])
    aw, ah = af.analysis_size(info.width, info.height, max(fs.cols, 160))
    img = af.render_frame(_source_frame(p, info, t, aw, ah), fs, W, H, af.find_font())
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    cols, rows, _, _ = af.grid(W, H, fs.cols)
    info_hdr = json.dumps({"cols": cols, "rows": rows, "mix": fs.mix, "chars": fs.chars, "width": W, "height": H,
                           "contrast": fs.contrast, "invert": fs.invert}, ensure_ascii=True)
    return buf.tobytes(), "image/png", {"X-Frame-Style": info_hdr}


def thumb(name: str) -> Path:
    p = _input_file(name)
    tdir = st.WORK / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"{p.name}-{int(p.stat().st_mtime)}.jpg"
    if not out.exists():
        info = st.probe_cached(p)
        aw, ah = af.analysis_size(info.width, info.height, 80)          # 320 px wide
        img = af.grab_frame(info, min(1.0, info.duration / 3) if info.kind == "video" else 0, aw, ah)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# --------------------------------------------------------------------------- actions

def act_render(req: dict) -> dict:
    style, settings, force = _style(req.get("style")), _settings(req.get("settings")), bool(req.get("force"))
    names = req.get("files") or [p.name for p in st.list_inputs()]
    queued, skipped = [], []
    for name in names:
        try:
            p = _input_file(name)
        except FileNotFoundError as e:
            skipped.append({"name": name, "why": str(e)})
            continue
        if JOBS.active("render", p.name):
            skipped.append({"name": p.name, "why": "already queued"})
            continue
        state_, why = st.render_state(p, style, {"fps": settings["fps"], "size": settings["size"]})
        if state_ != "new" and not force:
            skipped.append({"name": p.name, "why": why or "already rendered"})
            continue

        def fn(job, p=p):
            def prog(d, tot):
                job["progress"] = d / max(tot, 1)
                job["message"] = f"{d}/{tot} frames"
            r = st.render_one(p, style, fps=settings["fps"], size=settings["size"],
                              photo_seconds=settings["photo_seconds"], progress=prog, cancel=job["_cancel"])
            job["message"] = f"{st.fmt_time(r['duration'])} in {r['seconds']:.1f}s ({r['fps']} fps)"
            return {"output": r["output"].name}
        JOBS.submit("render", f"Render {p.name}", fn, file=p.name)
        queued.append(p.name)
    return {"queued": queued, "skipped": skipped}


def act_contact(req: dict) -> dict:
    style, settings = _style(req.get("style")), _settings(req.get("settings"))
    every = max(0.5, float(req.get("every") or 10))
    files = [_input_file(n) for n in (req.get("files") or [])] or st.list_inputs()
    if not files:
        raise ValueError("No clips in input/ yet")
    preset = req.get("preset") or None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    def fn(job):
        def prog(d, tot):
            job["progress"] = d / max(tot, 1)
            job["message"] = f"{d}/{tot} frames"
        out = st.contact_sheet(files, style, every=every, size=settings["size"],
                               photo_seconds=settings["photo_seconds"], tile_h=int(req.get("tile") or 270),
                               title=f"{st.style_summary(style, preset)}   |   every {every:g}s   |   {stamp}",
                               out=st.PREVIEWS / f"contact_{preset or 'custom'}_{stamp}.png",
                               progress=prog, cancel=job["_cancel"])
        return {"name": out.name, "url": f"/media/previews/{quote(out.name)}"}
    job = JOBS.submit("contact", f"Contact sheet · every {every:g}s", fn)
    return {"job": job["id"]}


def act_assemble(req: dict) -> dict:
    length, fade = str(req.get("length") or "3:00"), float(req.get("fade") if req.get("fade") not in (None, "") else 1)
    st.parse_duration(length)  # validate now
    if not st.list_outputs():
        raise ValueError("output/ is empty: render some clips first")
    if JOBS.active("assemble"):
        raise ValueError("The film is already being assembled")

    def fn(job):
        def prog(d, tot):
            job["progress"] = d / max(tot, 1e-9)
            job["message"] = f"{st.fmt_time(d)} / {st.fmt_time(tot)}"
        notes = []
        r = st.assemble(length, fade, size=req.get("size") or None, progress=prog, cancel=job["_cancel"],
                        log=notes.append)
        job["message"] = " ".join(notes) or f"{st.fmt_time(r['length'])} · {r['width']}x{r['height']}"
        return {"name": r["output"].name, "length": r["length"], "warning": " ".join(notes)}
    job = JOBS.submit("assemble", f"Assemble film · {length}", fn)
    return {"job": job["id"]}


def act_preset(req: dict) -> dict:
    name = str(req.get("name") or "").strip()
    if name in st.load_presets() and not req.get("overwrite"):
        return {"exists": True}
    st.save_preset(name, _style(req.get("style")), str(req.get("description") or "").strip())
    return {"saved": name}


def act_open(req: dict) -> dict:
    """Open a project folder in Finder, or reveal/open a file in it."""
    folders = {"input": st.INPUT, "output": st.OUTPUT, "previews": st.PREVIEWS, "final": st.FINAL, "root": st.ROOT}
    base = folders.get(req.get("folder"))
    if base is None:
        raise ValueError("unknown folder")
    target = base / Path(str(req.get("name"))).name if req.get("name") else base
    if not target.exists():
        raise FileNotFoundError(str(target))
    args = ["open", "-R", str(target)] if req.get("reveal") else ["open", str(target)]
    if sys.platform == "darwin":
        subprocess.run(args)
    return {"opened": str(target)}


ACTIONS = {"render": act_render, "contact": act_contact, "assemble": act_assemble, "preset": act_preset,
           "open": act_open}


# --------------------------------------------------------------------------- HTTP

class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "ascii-film"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _error(self, e: Exception, code: int = 400):
        self._json({"error": str(e) or e.__class__.__name__}, code)

    def _file(self, path: Path, cacheable: bool = False):
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "max-age=3600" if cacheable else "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(state())
            if path.startswith("/thumb/"):
                return self._file(thumb(path[len("/thumb/"):]), cacheable=True)
            m = re.fullmatch(r"/media/(output|previews|final|input)/([^/]+)", path)
            if m:
                base = {"output": st.OUTPUT, "previews": st.PREVIEWS, "final": st.FINAL, "input": st.INPUT}[m.group(1)]
                f = base / m.group(2)
                if f.is_file() and not f.name.startswith("."):
                    return self._file(f)
            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._error(e, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(e, 500)

    def _body_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                return self._upload(parse_qs(url.query).get("name", [""])[0])
            req = self._body_json()
            if url.path == "/api/frame":
                body, ctype, extra = frame_image(req)
                return self._send(200, body, ctype, extra)
            if url.path == "/api/cancel":
                return self._json({"cancelled": JOBS.cancel(req.get("id"))})
            m = re.fullmatch(r"/api/(\w+)", url.path)
            if m and m.group(1) in ACTIONS:
                return self._json(ACTIONS[m.group(1)](req))
            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._error(e, 404)
        except (ValueError, SystemExit, RuntimeError) as e:
            self._error(e, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(e, 500)

    def _upload(self, raw_name: str):
        name = Path(unquote(raw_name)).name.strip()
        if not name or name.startswith(".") or Path(name).suffix.lower() not in st.MEDIA_EXTS:
            return self._json({"error": f"{name or 'file'}: not a video or photo this tool reads "
                                        f"({', '.join(sorted(st.MEDIA_EXTS))})"}, 415)
        dest = st.INPUT / name
        if dest.exists():
            return self._json({"error": f"{name} is already in input/. Rename the new file, or remove the "
                                        f"old one yourself first; the dashboard never overwrites input/."}, 409)
        st.INPUT.mkdir(parents=True, exist_ok=True)
        tmp = st.INPUT / f".upload-{uuid.uuid4().hex[:8]}-{name}"
        left = int(self.headers.get("Content-Length") or 0)
        try:
            with open(tmp, "wb") as f:
                while left > 0:
                    chunk = self.rfile.read(min(4 << 20, left))
                    if not chunk:
                        raise ConnectionError("upload interrupted")
                    f.write(chunk)
                    left -= len(chunk)
            os.link(tmp, dest)  # fails instead of overwriting if the name appeared meanwhile
        except FileExistsError:
            return self._json({"error": f"{name} appeared in input/ during the upload; nothing was overwritten."}, 409)
        finally:
            tmp.unlink(missing_ok=True)
        self._json({"saved": name})


def serve(port: int = 8777, open_browser: bool = True):
    st.ensure_dirs()
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
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"ascii-film dashboard: {url}\nproject folder: {st.ROOT}\nCtrl-C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOBS.cancel()
        httpd.server_close()


if __name__ == "__main__":
    serve()
