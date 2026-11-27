"""Load config.yaml on top of the built-in defaults (defaults.yaml), so a key you delete falls back
instead of crashing, and a misspelt key gets a warning instead of being silently ignored."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

from . import HERE

DEFAULTS_FILE = Path(__file__).resolve().parent / "defaults.yaml"


def defaults() -> dict:
    return yaml.safe_load(DEFAULTS_FILE.read_text())


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _merge(base: dict, over: dict, where: str = "", warn=_stderr) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            warn(f"config: unknown key {where}{k!r} (ignored; check the spelling against soundfilm/defaults.yaml)")
            continue
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}{k}.", warn)
        else:
            out[k] = v
    return out


def config_path(root: Path, explicit: str | None = None) -> Path:
    """--config if given, else ROOT/config.yaml, else the main config.yaml next to ascii_film.py."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    for p in (root / "config.yaml", HERE / "config.yaml"):
        if p.exists():
            return p
    return DEFAULTS_FILE


def load(root: Path, explicit: str | None = None, warn=_stderr) -> tuple[dict, Path]:
    path = config_path(root, explicit)
    return parse(path.read_text(), warn), path


def parse(text: str, warn=_stderr) -> dict:
    """config.yaml text -> the full config (defaults filled in), checked. Raises SystemExit if it's unusable."""
    try:
        user = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"config: not valid YAML: {e}")
    if not isinstance(user, dict):
        raise SystemExit("config: the file must be sections like `seed:`, `editing:`, `ascii:`")
    cfg = _merge(defaults(), user, warn=warn)
    try:
        _check(cfg)
    except (KeyError, TypeError, IndexError) as e:
        raise SystemExit(f"config: a value has the wrong shape ({e!r}); compare with soundfilm/defaults.yaml")
    return cfg


def _check(cfg: dict) -> None:
    fps, sr = cfg["video"]["fps"], cfg["analysis"]["sample_rate"]
    if sr % fps:
        raise SystemExit(f"config: analysis.sample_rate ({sr}) must be a multiple of video.fps ({fps})")
    bands = cfg["ascii"]["charset"]["bands"]
    if not bands or any(len(set(b["chars"])) < 2 for b in bands):
        raise SystemExit("config: every ascii.charset.bands entry needs at least 2 different characters")
    p = cfg["editing"]["pools"]
    if p["dark_below"] > p["bright_above"]:
        raise SystemExit("config: editing.pools.dark_below must be <= bright_above")
    b = cfg["background"]
    for key, ok in (("show_in", ("silence", "black")), ("style", ("ascii", "plain")), ("clock", ("film", "gaps"))):
        if b[key] not in ok:
            raise SystemExit(f"config: background.{key} must be {' or '.join(ok)}, not {b[key]!r}")
    if str(b["chars"]).strip() and len(set(str(b["chars"]))) < 2:
        raise SystemExit('config: background.chars needs at least 2 different characters (or "" for the photos\' set)')
    if not 0 <= float(cfg["editing"]["between_onsets"]["sustain"]) <= 1:
        raise SystemExit("config: editing.between_onsets.sustain must be between 0 and 1")
    t = cfg["title"]
    corners = ("top-right", "top-left", "bottom-right", "bottom-left", "centre")
    if t["corner"] not in corners:
        raise SystemExit(f"config: title.corner must be one of {', '.join(corners)}, not {t['corner']!r}")
    if not 0 < float(t["seconds"]) <= 60:
        raise SystemExit("config: title.seconds must be more than 0 and at most 60")
    if not 0.005 <= float(t["size"]) <= 0.3 or not 0 <= float(t["margin"]) <= 0.4:
        raise SystemExit("config: title.size must be 0.005-0.3 and title.margin 0-0.4 (shares of the frame height)")
    c = cfg["clips"]
    if c["clock"] not in ("shots", "film", "random"):
        raise SystemExit(f"config: clips.clock must be shots, film or random, not {c['clock']!r}")
    if not 0 <= float(c["speed"]) <= 8:
        raise SystemExit(f"config: clips.speed must be between 0 and 8, not {c['speed']!r}")


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- editing values in place
# The dashboard's sliders change single values in config.yaml. Rewriting the file through a YAML dumper
# would throw away every comment, so this edits just the value on its line and leaves everything else
# (comments, spacing, order) alone. Each edit is checked by parsing the result.

_KEY_LINE = re.compile(r"^(?P<ind>[ \t]*)(?P<key>[A-Za-z_][\w-]*):(?P<rest>.*)$")
_ITEM_LINE = re.compile(r"^(?P<ind>[ \t]*)- ")


def get(cfg: dict, path: str):
    """get(cfg, 'editing.pools.bright_above'); list items by index: 'ascii.charset.bands.0.chars'."""
    node = cfg
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def yaml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{round(v, 4):.4f}".rstrip("0")
        return s + "0" if s.endswith(".") else s
    return json.dumps(str(v), ensure_ascii=False)              # a JSON string is a valid YAML string


def _split_comment(rest: str) -> tuple[str, str]:
    """' 0.62    # note' -> (' 0.62    ', '# note'), ignoring # inside quotes."""
    q = None
    for i, ch in enumerate(rest):
        if q:
            if ch == q and not (q == '"' and rest[i - 1] == "\\"):
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            return rest[:i], rest[i:]
    return rest, ""


def _with_value(old_val: str, new: str) -> str:
    """Put `new` where the old value was, keeping the comment column if it fits."""
    lead = old_val[:len(old_val) - len(old_val.lstrip())] or " "
    width = len(old_val)
    body = lead + new
    return body + " " * (width - len(body)) if len(body) < width else body + (" " if old_val.strip() else "")


def _flow_set(line: str, key: str, new: str) -> str | None:
    """Set key in a one-line flow mapping like `- {name: glassy, above: 0.70, chars: " .:'"}`."""
    pat = re.compile(r"(\b" + re.escape(key) + r":\s*)(\"(?:[^\"\\]|\\.)*\"|'(?:[^']|'')*'|[^,}]+?)(\s*[,}])")
    out, n = pat.subn(lambda m: m.group(1) + new + m.group(3), line, count=1)
    return out if n else None


def _set_one(lines: list[str], path: list[str], value) -> None:
    new = yaml_value(value)
    stack: list[tuple[int, str]] = []                          # (indent, key) of the enclosing mappings
    items: dict[tuple, int] = {}                               # list index counters per parent path
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ITEM_LINE.match(line)
        if m:                                                  # "- ..." list item under the last key
            ind = len(m.group("ind"))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            parent = tuple(k for _, k in stack)
            idx = items.get(parent, -1) + 1
            items[parent] = idx
            here = list(parent) + [str(idx)]
            if path[:len(here)] == here and len(path) == len(here) + 1:
                done = _flow_set(line, path[-1], new)
                if done is None:
                    raise SystemExit(f"config: can't find {'.'.join(path)} in its line; edit it in config.yaml")
                lines[i] = done
                return
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        ind, key = len(m.group("ind")), m.group("key")
        while stack and stack[-1][0] >= ind:
            stack.pop()
        here = [k for _, k in stack] + [key]
        if here == path:
            val, comment = _split_comment(m.group("rest"))
            lines[i] = f"{m.group('ind')}{key}:{_with_value(val, new)}{comment}".rstrip()
            return
        stack.append((ind, key))
    _insert(lines, path, new)


def _insert(lines: list[str], path: list[str], new: str) -> None:
    """The key isn't in the file (it was deleted and fell back to the default): add it under its parent."""
    if any(p.isdigit() for p in path):
        raise SystemExit(f"config: {'.'.join(path)} isn't in config.yaml; add it there by hand")
    stack: list[tuple[int, str]] = []
    best = None                                                # (line index, indent) of the deepest parent found
    for i, line in enumerate(lines):
        m = _KEY_LINE.match(line) if line.strip() and not line.lstrip().startswith("#") else None
        if not m:
            continue
        ind, key = len(m.group("ind")), m.group("key")
        while stack and stack[-1][0] >= ind:
            stack.pop()
        here = [k for _, k in stack] + [key]
        if path[:len(here)] == here and (best is None or len(here) > len(best[2])):
            best = (i, ind, here)
        stack.append((ind, key))
    if best is None:
        depth, at, ind = 0, len(lines), -2
    else:
        at, ind, here = best
        depth = len(here)
        j = at + 1                                             # after the parent's last child
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > ind):
            j += 1
        at = j
        while at > best[0] + 1 and not lines[at - 1].strip():
            at -= 1
    new_lines = []
    for d, k in enumerate(path[depth:]):
        pad = " " * (ind + 2 * (d + 1))
        last = depth + d == len(path) - 1
        new_lines.append(f"{pad}{k}: {new}" if last else f"{pad}{k}:")
    if depth == 0 and at > 0 and lines[at - 1].strip():
        new_lines.insert(0, "")                                # a new top-level section: a blank line before it
    lines[at:at] = new_lines


def set_values(text: str, changes: dict) -> tuple[str, dict]:
    """Apply {'editing.pools.bright_above': 0.7, ...} to config.yaml text. Returns (new text, full config).
    Raises SystemExit if the result doesn't parse, or doesn't hold exactly the values asked for."""
    lines = text.splitlines()
    for path, value in changes.items():
        _set_one(lines, path.split("."), value)
    out = "\n".join(lines) + "\n"
    cfg = parse(out, warn=lambda *_: None)
    for path, value in changes.items():
        got = get(cfg, path)
        same = abs(float(got) - float(value)) < 1e-4 if isinstance(value, float) and not isinstance(got, bool) \
            else got == value
        if not same:
            raise SystemExit(f"config: setting {path} to {value!r} came out as {got!r}; edit it in config.yaml")
    return out, cfg
