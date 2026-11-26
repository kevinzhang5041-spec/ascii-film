#!/usr/bin/env bash
# One-time setup: checks Python 3 and ffmpeg, creates .venv/ here and installs the packages into it.
# Safe to re-run. Your system Python is never modified.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
info() { printf '  ..   %s\n' "$*"; }
die()  { printf '  \033[31mfail\033[0m %s\n' "$*" >&2; exit 1; }

echo "ascii-film setup"

PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || die "python3 not found. Install it with: brew install python"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
  || die "Python 3.9+ needed (found $("$PY" -V 2>&1))"
ok "python3 $("$PY" -c 'import platform; print(platform.python_version())') ($PY)"

if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  if command -v brew >/dev/null; then
    info "installing ffmpeg with Homebrew (a few minutes)..."
    brew install ffmpeg
  else
    die "ffmpeg not found and Homebrew isn't installed. See https://brew.sh then run: brew install ffmpeg"
  fi
fi
ok "$(ffmpeg -version | head -1 | cut -d' ' -f1-3)"

command -v sips >/dev/null && ok "sips (macOS built-in, converts .heic)" || info "sips not found: .heic photos won't convert"

if [[ ! -x .venv/bin/python ]]; then
  info "creating virtual environment in .venv/"
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
.venv/bin/python -c 'import cv2, numpy, PIL; print(f"  \033[32mok\033[0m   opencv {cv2.__version__}, numpy {numpy.__version__}, pillow {PIL.__version__} (in .venv)")'

mkdir -p input output previews final
ok "folders: input/ output/ previews/ final/"
echo "Done. Drop clips into input/ and run ./ascii (or ./ascii dashboard)."
