"""soundfilm: a microsound track drives the editing and the ASCII rendering of a Brakhage-style film.

  analysis.py  step 1  per-frame audio features -> analysis.json
  score.py     step 2  mapping rules -> one decision per frame (the "score")
  frames.py    step 3  draws one frame: image -> crop/layer/colour -> ASCII (ascii_film) -> grain/jitter
  render.py    step 3  parallel render to H.264 + the original audio
  debug.py     overlay for --debug
  plot.py      analysis.png
  verify.py    sync / duration / determinism checks
  testgen.py   synthetic test track and placeholder images
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # the ascii-film folder, which holds ascii_film.py
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
