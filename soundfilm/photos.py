"""The photo sorter behind the dashboard's /sort page.

Every photo lives in exactly one folder under images/:

  inbox/      not sorted yet (the film doesn't use these)
  bright/     \
  texture/     > the pools the film draws from
  dark/       /
  rejected/   set aside (the film doesn't use these either)

Sorting a photo moves its file from one folder to another. Nothing is ever deleted, and a move never
overwrites: a name that's taken gets -2, -3 ... Photos come in from the Photos app (the current selection
or an album), from the file chooser, or by copying files into images/inbox/ in Finder.

Hand crops: images/crops.json holds how you framed a photo for the film: an optional rotation (rot, degrees
clockwise) and the part to show ({x, y, w, h} as fractions of the rotated photo, 16:9). The photo file itself is never changed; without a crop the film uses the centre 16:9.

Each photo also gets a *suggested* pool from the automatic scores in sortimages.py (most detailed third ->
texture, the rest split by brightness). It's only a suggestion; you decide.
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ascii_film as af

from . import score, sortimages

FOLDERS = ("inbox",) + score.POOLS + ("rejected",)


def folder(root: Path, cfg: dict, where: str) -> Path:
    if where not in FOLDERS:
        raise ValueError(f"unknown folder {where!r}")
    return Path(root) / cfg["paths"]["images"] / where


def photos_in(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir()
                   if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in af.IMAGE_EXTS),
                  key=lambda p: p.name.lower())


def resolve(root: Path, cfg: dict, ref: str) -> Path:
    """'inbox/IMG_1.jpeg' -> the file, or FileNotFoundError."""
    where, _, name = str(ref).partition("/")
    p = folder(root, cfg, where) / Path(name).name
    if not p.is_file() or p.name.startswith("."):
        raise FileNotFoundError(f"{ref} is not there")
    return p


def unique(d: Path, name: str) -> Path:
    p = d / name
    n = 1
    while p.exists():
        n += 1
        p = d / f"{Path(name).stem}-{n}{Path(name).suffix}"
    return p


def move(root: Path, cfg: dict, refs: list[str], to: str) -> dict:
    """Move photos into folder `to`. Returns {'moved': [[old ref, new ref]], 'errors': [...]}."""
    dest = folder(root, cfg, to)
    dest.mkdir(parents=True, exist_ok=True)
    moved, errors = [], []
    for ref in refs:
        try:
            src = resolve(root, cfg, ref)
            if src.parent == dest:
                continue
            target = unique(dest, src.name)
            old_key = crop_key(src)
            os.rename(src, target)                  # same disk: instant, and never replaces (name is free)
            if target.name != src.name:
                rekey_crop(root, cfg, old_key, crop_key(target))
            moved.append([ref, f"{to}/{target.name}"])
        except (FileNotFoundError, ValueError, OSError) as e:
            errors.append(str(e))
    return {"moved": moved, "errors": errors}


# --------------------------------------------------------------------------- hand crops

def _crops_file(root: Path, cfg: dict) -> Path:
    return Path(root) / cfg["paths"]["images"] / "crops.json"


def crop_key(p: Path) -> str:
    """A crop follows its photo between folders: keyed by file name and size."""
    return f"{p.name}|{p.stat().st_size}"


def load_crops(root: Path, cfg: dict) -> dict:
    try:
        return json.loads(_crops_file(root, cfg).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _write_crops(root: Path, cfg: dict, crops: dict) -> None:
    f = _crops_file(root, cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(crops, indent=1, sort_keys=True))
    os.replace(tmp, f)


def crop_of(root: Path, cfg: dict, p: Path, crops: dict | None = None) -> dict | None:
    return (crops if crops is not None else load_crops(root, cfg)).get(crop_key(p))


def save_crop(root: Path, cfg: dict, ref: str, rect: dict | None, rot: float = 0.0) -> dict | None:
    """Save how a photo is framed: rot (degrees clockwise) and rect ({x, y, w, h}, fractions of the rotated
    photo). No rect: the centre 16:9 of the rotated photo. No rect and no rotation: back to the default."""
    p = resolve(root, cfg, ref)
    crops = load_crops(root, cfg)
    rot = round(((float(rot) + 180) % 360) - 180, 2)     # -180 .. 180
    if rot == -180:
        rot = 180.0
    if rect is None:
        crops.pop(crop_key(p), None)
        if rot:
            crops[crop_key(p)] = {"rot": rot}
    else:
        try:
            x, y, w, h = (float(rect[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError):
            raise ValueError("a crop needs x, y, w and h")
        w, h = min(max(w, 0.02), 1.0), min(max(h, 0.02), 1.0)
        x, y = min(max(x, 0.0), 1.0 - w), min(max(y, 0.0), 1.0 - h)
        crops[crop_key(p)] = {"x": round(x, 5), "y": round(y, 5), "w": round(w, 5), "h": round(h, 5),
                              **({"rot": rot} if rot else {})}
    _write_crops(root, cfg, crops)
    return crops.get(crop_key(p))


def rekey_crop(root: Path, cfg: dict, old: str, new: str) -> None:
    crops = load_crops(root, cfg)
    if old in crops and old != new:
        crops[new] = crops.pop(old)
        _write_crops(root, cfg, crops)


def views_for(root: Path, cfg: dict, pools: dict) -> dict:
    """{readable path the renderer opens: hand crop} for the photos in the pools that have one."""
    crops = load_crops(root, cfg)
    if not crops:
        return {}
    base = Path(root) / cfg["paths"]["images"]
    out = {}
    for items in pools.values():
        for display, readable in items:
            src = base / display                    # display name is "<pool>/<file>"
            if src.is_file():
                c = crops.get(crop_key(src))
                if c:
                    out[readable] = c
    return out


# --------------------------------------------------------------------------- automatic scores

def _key(p: Path) -> str:
    """Scores follow the photo when it's moved: keyed by name, size and time, not folder."""
    st = p.stat()
    return f"{p.name}|{st.st_size}|{st.st_mtime_ns}"


def _scores_file(root: Path) -> Path:
    return Path(root) / ".work" / "photo_scores.json"


def load_scores(root: Path) -> dict:
    try:
        return json.loads(_scores_file(root).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def all_photos(root: Path, cfg: dict) -> dict[str, list[Path]]:
    return {w: photos_in(folder(root, cfg, w)) for w in FOLDERS}


def unscored(root: Path, cfg: dict) -> list[Path]:
    have = load_scores(root)
    return [p for ps in all_photos(root, cfg).values() for p in ps if _key(p) not in have]


def score_photos(root: Path, paths: list[Path], progress=None) -> int:
    """Score photos (brightness and detail of the film's 16:9 crop) into .work/photo_scores.json."""
    cache = Path(root) / ".work"
    have = load_scores(root)
    done = 0

    def one(p):
        try:
            return _key(p), sortimages.scores(sortimages.film_view(p, cache))
        except Exception:                               # unreadable: no suggestion for it
            return None
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, paths), 1):
            if r:
                k, s = r
                have[k] = {"brightness": round(s["brightness"], 4), "texture": round(s["texture"], 5)}
                done += 1
            if progress:
                progress(i, len(paths))
    f = _scores_file(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(have))
    os.replace(tmp, f)
    return done


def listing(root: Path, cfg: dict) -> dict:
    """Every photo, where it is, its scores and the automatic suggestion."""
    have = load_scores(root)
    crops = load_crops(root, cfg)
    found = all_photos(root, cfg)
    rows = []
    for where, ps in found.items():
        for p in ps:
            s = have.get(_key(p))
            rows.append(dict(ref=f"{where}/{p.name}", where=where, name=p.name, mtime=p.stat().st_mtime,
                             crop=crop_of(root, cfg, p, crops),
                             **({"brightness": s["brightness"], "texture": s["texture"]} if s else {})))
    scored = [r for r in rows if "texture" in r and r["where"] != "rejected"]
    if scored:
        sortimages.assign(scored, 1 / 3)                # sets r["pool"]
        for r in scored:
            r["suggest"] = r.pop("pool")
    return {"photos": rows, "counts": {w: len(found[w]) for w in FOLDERS},
            "unscored": sum("texture" not in r for r in rows)}


# --------------------------------------------------------------------------- the Photos app

def _osa(script: str, timeout: int = 600) -> str:
    try:
        r = subprocess.run(["osascript", "-"], input=script, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Photos didn't answer. If macOS is asking whether this may control Photos, click OK "
                           "(or allow it in System Settings > Privacy & Security > Automation), then try again.")
    if r.returncode != 0:
        msg = r.stderr.strip()
        if "-1743" in msg or "Not authorized" in msg:
            raise RuntimeError("macOS didn't allow controlling Photos. Allow it in System Settings > Privacy & "
                               "Security > Automation, then try again.")
        raise RuntimeError(f"Photos: {msg}")
    return r.stdout


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def albums() -> list[dict]:
    out = _osa('''
tell application "Photos"
    set out to ""
    repeat with a in albums
        set out to out & (id of a) & tab & (name of a) & tab & (count of media items of a) & linefeed
    end repeat
end tell
return out''', 45)
    rows = [line.split("\t") for line in out.splitlines() if line.count("\t") == 2]
    return [{"id": i, "name": n, "count": int(c)} for i, n, c in rows]


def items(source: dict) -> list[tuple[str, str]]:
    """[(Photos id, filename)] for {'selection': true} or {'album_id': ...}."""
    if source.get("album_id"):
        pick = f"media items of album id {_q(source['album_id'])}"
    else:
        pick = "selection"
    out = _osa(f'''
tell application "Photos"
    set its_ to {pick}
    set ids to {{}}
    set fns to {{}}
    repeat with m in its_
        set end of ids to id of m
        set end of fns to filename of m
    end repeat
end tell
set out to ""
repeat with i from 1 to count of ids
    set out to out & (item i of ids) & tab & (item i of fns) & linefeed
end repeat
return out''', 120)
    return [tuple(line.split("\t", 1)) for line in out.splitlines() if "\t" in line]


def import_from_photos(root: Path, cfg: dict, source: dict, progress=None, log=print) -> dict:
    """Export photos (not videos or screenshots-as-PNG? -> PNGs are kept: you choose) from the Photos app
    into images/inbox/. A photo imported before is skipped (remembered in .work/photos_imported.json)."""
    found = items(source)
    if not found:
        raise SystemExit("Nothing to import: select photos in the Photos app first." if not source.get("album_id")
                         else "That album is empty.")
    stills = [(i, f) for i, f in found if Path(f).suffix.lower() in af.IMAGE_EXTS]
    log(f"{len(found)} items, {len(stills)} photos ({len(found) - len(stills)} videos and other files skipped)")
    seen_file = Path(root) / ".work" / "photos_imported.json"
    try:
        seen = json.loads(seen_file.read_text())
    except (OSError, json.JSONDecodeError):
        seen = {}
    todo = [(i, f) for i, f in stills if i not in seen]
    if len(todo) < len(stills):
        log(f"{len(stills) - len(todo)} already imported before, skipped")
    inbox = folder(root, cfg, "inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    tmp = inbox / f".importing-{uuid.uuid4().hex[:8]}"
    added: list[str] = []
    try:
        for b in range(0, len(todo), 25):
            batch = todo[b:b + 25]
            tmp.mkdir(exist_ok=True)
            ids = " , ".join(f"media item id {_q(i)}" for i, _ in batch)
            _osa(f'''
with timeout of 1800 seconds
    tell application "Photos" to export {{{ids}}} to (POSIX file {_q(str(tmp))})
end timeout''', 1900)
            for f in sorted(tmp.iterdir()):
                if f.is_file() and f.suffix.lower() in af.IMAGE_EXTS:
                    dest = unique(inbox, f.name)
                    os.rename(f, dest)
                    added.append(dest.name)
            for i, name in batch:
                seen[i] = name
            seen_file.parent.mkdir(parents=True, exist_ok=True)
            seen_file.write_text(json.dumps(seen))
            if progress:
                progress(min(b + 25, len(todo)), len(todo))
            log(f"exported {min(b + 25, len(todo))}/{len(todo)}")
    finally:
        if tmp.exists() and not any(tmp.iterdir()):
            tmp.rmdir()                                 # our own empty scratch folder
    return {"added": added, "skipped": len(stills) - len(todo), "not_photos": len(found) - len(stills)}
