"""Photo sets and track sections: different parts of the soundtrack draw from different photos.

A *set* is a named group of photos ("forest", "faces" ...). A photo can be in any number of sets, and it
stays in its pool folder (bright / texture / dark); sets are tags on top of the pools. You tag photos in
the sorter.

A *section* is a stretch of the track, from its start to the next section's start. Each section lists the
sets its cuts draw from; with no sets it uses every photo, as the film did before sections existed. Inside
a section the sound still chooses bright / texture / dark, unless the section has `any_pool` set. When
its sets have no photo in the pool the sound asks for, the nearest pool that has one stands in. When they
have none in the pools at all, the section uses every photo. A render warns about both.

Everything lives in images/sets.json, next to crops.json:

  {"sets":     [{"name": "forest", "color": "#a3d96b"}, ...],
   "members":  {"IMG_1.jpg|482113": ["forest", "faces"], ...},    # name|size, like crops: follows moves
   "sections": [{"start": 0.0, "sets": [], "any_pool": false, "label": ""},
                {"start": 45.25, "sets": ["forest"], "any_pool": false, "label": "walk"}, ...]}

No file, or a single section with no sets, makes exactly the film you'd get without this feature.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from . import photos

PALETTE = ["#ff9f6b", "#c792ea", "#5fd7d7", "#f78fb3", "#a3d96b", "#e8c66a", "#7aa2ff", "#d9a3ff",
           "#ff7a7a", "#8fe3b0"]
_NAME = re.compile(r"^\w[\w .'&+-]{0,31}$")
MIN_GAP = 0.25                                  # seconds between two section starts


def _file(root: Path, cfg: dict) -> Path:
    return Path(root) / cfg["paths"]["images"] / "sets.json"


def load(root: Path, cfg: dict) -> dict:
    try:
        d = json.loads(_file(root, cfg).read_text())
    except (OSError, json.JSONDecodeError):
        d = {}
    if not isinstance(d, dict):
        d = {}
    d.setdefault("sets", [])
    d.setdefault("members", {})
    d.setdefault("sections", [])
    return d


def mtime(root: Path, cfg: dict) -> float:
    try:
        return _file(root, cfg).stat().st_mtime
    except OSError:
        return 0.0


def _save(root: Path, cfg: dict, d: dict) -> None:
    f = _file(root, cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    d["members"] = {k: v for k, v in sorted(d["members"].items()) if v}
    tmp = f.with_name(f".{f.name}.{uuid.uuid4().hex[:6]}")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, f)


def names(d: dict) -> list[str]:
    return [s["name"] for s in d["sets"]]


def _find(d: dict, name: str) -> dict:
    for s in d["sets"]:
        if s["name"] == name:
            return s
    raise ValueError(f"there's no set called {name!r}")


def _check_name(d: dict, name: str, but: str | None = None) -> str:
    name = " ".join(str(name or "").split())
    if not _NAME.match(name):
        raise ValueError("a set name is 1-32 letters, digits, spaces or . ' & + - (starting with a letter or digit)")
    if any(n.lower() == name.lower() and n != but for n in names(d)):
        raise ValueError(f"there's already a set called {name!r}")
    return name


# --------------------------------------------------------------------------- sets

def create(root: Path, cfg: dict, name: str) -> dict:
    d = load(root, cfg)
    name = _check_name(d, name)
    used = {s.get("color") for s in d["sets"]}
    color = next((c for c in PALETTE if c not in used), PALETTE[len(d["sets"]) % len(PALETTE)])
    d["sets"].append({"name": name, "color": color})
    _save(root, cfg, d)
    return d


def rename(root: Path, cfg: dict, name: str, to: str) -> dict:
    d = load(root, cfg)
    s = _find(d, name)
    to = _check_name(d, to, but=name)
    s["name"] = to
    for k, v in d["members"].items():
        d["members"][k] = [to if x == name else x for x in v]
    for sec in d["sections"]:
        sec["sets"] = [to if x == name else x for x in sec.get("sets", [])]
    _save(root, cfg, d)
    return d


def recolor(root: Path, cfg: dict, name: str, color: str) -> dict:
    d = load(root, cfg)
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)):
        raise ValueError("a colour is #rrggbb")
    _find(d, name)["color"] = color.lower()
    _save(root, cfg, d)
    return d


def remove(root: Path, cfg: dict, name: str) -> dict:
    """Remove a set: its photos leave it (the photos themselves stay where they are) and sections stop
    using it."""
    d = load(root, cfg)
    _find(d, name)
    d["sets"] = [s for s in d["sets"] if s["name"] != name]
    d["members"] = {k: [x for x in v if x != name] for k, v in d["members"].items()}
    for sec in d["sections"]:
        sec["sets"] = [x for x in sec.get("sets", []) if x != name]
    _save(root, cfg, d)
    return d


def set_members(root: Path, cfg: dict, refs: list[str], name: str, on: bool) -> dict:
    """Put photos (refs like 'bright/IMG_1.jpg') into set `name`, or take them out of it."""
    d = load(root, cfg)
    _find(d, name)
    changed, errors = 0, []
    for ref in refs:
        try:
            key = photos.crop_key(photos.resolve(root, cfg, ref))
        except (FileNotFoundError, ValueError, OSError) as e:
            errors.append(str(e))
            continue
        have = d["members"].get(key, [])
        if on and name not in have:
            d["members"][key] = have + [name]
            changed += 1
        elif not on and name in have:
            d["members"][key] = [x for x in have if x != name]
            changed += 1
    if changed:
        _save(root, cfg, d)
    return {"changed": changed, "errors": errors}


def rekey(root: Path, cfg: dict, old: str, new: str) -> None:
    """A photo was renamed (moved onto a taken name): its sets go with it."""
    d = load(root, cfg)
    if old in d["members"] and old != new:
        d["members"][new] = d["members"].pop(old)
        _save(root, cfg, d)


def members_of(d: dict, p: Path) -> list[str]:
    try:
        return list(d["members"].get(photos.crop_key(p), []))
    except OSError:
        return []


def counts(root: Path, cfg: dict, d: dict | None = None) -> dict:
    """{set: {folder: photos}} plus '' for photos in no set."""
    d = d or load(root, cfg)
    out = {n: {w: 0 for w in photos.FOLDERS} for n in names(d)}
    out[""] = {w: 0 for w in photos.FOLDERS}
    for w, ps in photos.all_photos(root, cfg).items():
        for p in ps:
            m = [x for x in members_of(d, p) if x in out] or [""]
            for x in m:
                out[x][w] += 1
    return out


# --------------------------------------------------------------------------- sections

def normalize_sections(raw, known: list[str], duration: float | None = None) -> list[dict]:
    """Clean up sections from the page: sorted by start, the first at 0, no two closer than MIN_GAP,
    only sets that exist."""
    out = []
    for s in raw or []:
        try:
            start = max(0.0, float(s.get("start", 0)))
        except (TypeError, ValueError, AttributeError):
            raise ValueError("each section needs a start time in seconds")
        if duration and start >= duration:
            continue
        sets_ = [x for x in dict.fromkeys(s.get("sets") or []) if x in known]
        out.append({"start": round(start, 3), "sets": sets_, "any_pool": bool(s.get("any_pool")),
                    "label": " ".join(str(s.get("label") or "").split())[:40]})
    out.sort(key=lambda s: s["start"])
    if not out:
        return []
    out[0]["start"] = 0.0
    kept = [out[0]]
    for s in out[1:]:
        if s["start"] - kept[-1]["start"] >= MIN_GAP:
            kept.append(s)
    return kept


def save_sections(root: Path, cfg: dict, raw, duration: float | None = None) -> dict:
    d = load(root, cfg)
    d["sections"] = normalize_sections(raw, names(d), duration)
    _save(root, cfg, d)
    return d


def active(d: dict) -> bool:
    """Do the sections change anything? (One section using every photo, in pools, doesn't.)"""
    secs = d["sections"]
    return len(secs) > 1 or any(s.get("sets") or s.get("any_pool") for s in secs)


def label(i: int, s: dict) -> str:
    return f"{i + 1}:" + (s.get("label") or "+".join(s.get("sets") or []) or "every photo")


def sections_for(root: Path, cfg: dict, pools: dict, fps: float, n: int) -> list[dict] | None:
    """The sections for score.build_score: frame ranges, and the photos (display names, '<pool>/<file>')
    each may use (None: every photo). None when there are no sections."""
    d = load(root, cfg)
    if not active(d):
        return None
    base = Path(root) / cfg["paths"]["images"]
    sets_of = {}
    for items in pools.values():
        for display, _ in items:
            sets_of[display] = set(members_of(d, base / display))
    secs = d["sections"]
    starts = [min(n, max(0, int(round(s["start"] * fps)))) for s in secs] + [n]
    out = []
    for i, s in enumerate(secs):
        want = set(s.get("sets") or [])
        allow = {k for k, v in sets_of.items() if v & want} if want else None
        out.append(dict(index=i, start=starts[i], end=max(starts[i], starts[i + 1]), sets=list(s.get("sets") or []),
                        any_pool=bool(s.get("any_pool")), label=label(i, s), allow=allow))
    return out
