"""./film: sound makes image. Run ./film --help, or see SOUNDFILM.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import HERE

COMMANDS = ("render", "analyze", "plot", "verify", "stills", "testgen", "selftest", "dashboard", "sort")


def _root_cfg(a):
    from . import config
    root = Path(a.root).expanduser().resolve()
    cfg, path = config.load(root, a.config)
    return root, cfg, path


def _latest(root, cfg) -> Path:
    vids = sorted((root / cfg["paths"]["renders"]).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        raise SystemExit("No renders yet: run ./film first")
    return vids[-1]


def _plot(root, cfg, an, span=None, log=print):
    from . import plot
    return plot.for_root(root, cfg, an, span, log)


def cmd_render(a):
    import ascii_film as af
    from . import render
    root, cfg, cpath = _root_cfg(a)
    print(f"config: {cpath}")
    meta = render.render(root, cfg, seed=a.seed, debug_=a.debug, preview=a.preview, out=a.out, jobs=a.jobs,
                         reanalyze=a.reanalyze, progress=af.cli_progress())
    from . import analysis
    _plot(root, cfg, analysis.load_or_analyze(root, cfg))
    if a.verify:
        from . import verify
        rep = verify.verify(Path(a.out) if a.out else root / cfg["paths"]["renders"] / meta["video"], root, cfg)
        print(verify.summary(rep))
        return 0 if rep["passed"] else 1
    return 0


def cmd_analyze(a):
    from . import analysis
    root, cfg, _ = _root_cfg(a)
    an = analysis.load_or_analyze(root, cfg, force=a.force)
    print(f"{root / cfg['paths']['analysis']}: {an.duration:.2f} s, {an.n} frames, {len(an.onsets)} onsets")
    _plot(root, cfg, an)


def cmd_plot(a):
    from . import analysis
    root, cfg, _ = _root_cfg(a)
    _plot(root, cfg, analysis.load_or_analyze(root, cfg), tuple(a.span) if a.span else None)


def cmd_verify(a):
    from . import verify
    root, cfg, _ = _root_cfg(a)
    video = Path(a.video) if a.video else _latest(root, cfg)
    rep = verify.verify(video, root, cfg)
    print(verify.summary(rep))
    return 0 if rep["passed"] else 1


def cmd_stills(a):
    from . import verify
    root, cfg, _ = _root_cfg(a)
    video = Path(a.video) if a.video else _latest(root, cfg)
    print(f"wrote {verify.stills(video)}")


def cmd_testgen(a):
    from . import testgen
    testgen.generate(Path(a.root).expanduser().resolve(), force=a.force)


def cmd_sort(a):
    from . import sortimages
    root, cfg, _ = _root_cfg(a)
    sortimages.sort_photos([Path(x) for x in a.sources], root, cfg, a.texture_share, a.max_side, a.dry_run)


def cmd_dashboard(a):
    from . import dashboard
    dashboard.serve(Path(a.root), a.config, a.port, open_browser=not a.no_open)


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def cmd_selftest(a):
    """Synthetic assets -> analysis + plot -> full render, debug render, preview -> verify all -> stills
    -> determinism (same seed twice = identical file, other seed = different film)."""
    import ascii_film as af
    from . import analysis, render, testgen, verify
    root = Path(a.root).expanduser().resolve()
    if not (root / "audio" / "track.wav").exists():
        testgen.generate(root)
    root, cfg, cpath = _root_cfg(a)
    print(f"selftest in {root} with {cpath}")
    an = analysis.load_or_analyze(root, cfg, force=True)
    _plot(root, cfg, an)
    results, t0 = {}, time.time()
    runs = [("full", dict()), ("debug", dict(debug_=True)), ("preview", dict(preview=(13.0, 17.0)))]
    for name, kw in runs:
        meta = render.render(root, cfg, progress=af.cli_progress(f"{name:8s}"), **kw)
        video = root / cfg["paths"]["renders"] / meta["video"]
        rep = verify.verify(video, root, cfg)
        print(verify.summary(rep))
        results[name] = (video, rep)
        print(f"wrote {verify.stills(video)}")
    # determinism
    renders = root / cfg["paths"]["renders"]
    p = (13.0, 17.0)
    again = render.render(root, cfg, preview=p, out=renders / "determinism_a.mp4", log=lambda *_: None)
    other = render.render(root, cfg, preview=p, seed=cfg["seed"] + 1, out=renders / "determinism_b.mp4",
                          log=lambda *_: None)
    same = _sha1(results["preview"][0]) == _sha1(renders / again["video"])
    edl = lambda v: (renders / v).with_name(Path(v).stem + ".edl.csv").read_text()
    differs = edl(other["video"]) != edl(results["preview"][0].name)
    print(f"determinism: seed {cfg['seed']} rendered twice -> {'identical file (sha1)' if same else 'DIFFERENT'}; "
          f"seed {cfg['seed'] + 1} -> {'a different edit' if differs else 'THE SAME edit'}")
    ok = all(r["passed"] for _, r in results.values()) and same and differs
    print(f"selftest {'PASSED' if ok else 'FAILED'} in {time.time() - t0:.0f} s")
    return 0 if ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # `./film --preview 5 9` means `./film render --preview 5 9`; `./film --root x verify` works too
    i = next((j for j, x in enumerate(argv) if x in COMMANDS), None)
    if argv not in (["-h"], ["--help"]):
        argv = ["render"] + argv if i is None else [argv[i]] + argv[:i] + argv[i + 1:]

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=str(HERE), help="project folder with audio/ and images/ (default: %(default)s)")
    common.add_argument("--config", help="config file (default: ROOT/config.yaml, else the main config.yaml)")
    ap = argparse.ArgumentParser(prog="film", description="Sound makes image: a soundtrack drives the editing "
                                 "and ASCII rendering of a film. See SOUNDFILM.md.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", parents=[common], help="render the film (default command)")
    r.add_argument("--preview", nargs=2, type=float, metavar=("START", "END"),
                   help="render only START..END seconds, fast and low-res")
    r.add_argument("--debug", action="store_true", help="overlay frame number, onsets, features and decisions")
    r.add_argument("--seed", type=int, help="override the config seed")
    r.add_argument("--jobs", type=int, help="parallel render processes (default 6 on this Mac)")
    r.add_argument("--out", help="output .mp4 (default renders/film_seed<N>.mp4 or renders/preview_...)")
    r.add_argument("--reanalyze", action="store_true", help="ignore the cached analysis.json")
    r.add_argument("--verify", action="store_true", help="run the sync checks on the result")
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("analyze", parents=[common], help="analyse the track -> analysis.json + analysis.png")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("plot", parents=[common], help="analysis.png (optionally zoomed to START END seconds)")
    s.add_argument("span", nargs="*", type=float, metavar="SECONDS")
    s.set_defaults(fn=cmd_plot)
    s = sub.add_parser("verify", parents=[common], help="sync / duration checks on a render (default: newest)")
    s.add_argument("video", nargs="?")
    s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("stills", parents=[common], help="export telling frames + a contact sheet from a render")
    s.add_argument("video", nargs="?")
    s.set_defaults(fn=cmd_stills)
    s = sub.add_parser("testgen", parents=[common], help="write a synthetic track + placeholder images into ROOT")
    s.add_argument("--force", action="store_true", help="overwrite test assets made earlier")
    s.set_defaults(fn=cmd_testgen)
    s = sub.add_parser("selftest", parents=[common], help="testgen + render + verify everything in ROOT")
    s.set_defaults(fn=cmd_selftest)
    s = sub.add_parser("sort", parents=[common], help="score photos and copy them into images/bright|texture|dark")
    s.add_argument("sources", nargs="+", help="folders or files of photos (never modified)")
    s.add_argument("--texture-share", type=float, default=1 / 3, help="share that goes to texture (default 1/3)")
    s.add_argument("--max-side", type=int, default=2560, help="long side of the copies in pixels (default 2560)")
    s.add_argument("--dry-run", action="store_true", help="score and write the report/contact sheet, copy nothing")
    s.set_defaults(fn=cmd_sort)
    s = sub.add_parser("dashboard", parents=[common], help="the browser dashboard (http://127.0.0.1:8778)")
    s.add_argument("--port", type=int, default=8778)
    s.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    s.set_defaults(fn=cmd_dashboard)

    a = ap.parse_args(argv)
    if a.cmd == "plot" and a.span and len(a.span) != 2:
        ap.error("plot takes START END")
    if a.cmd in ("testgen", "selftest") and a.root == str(HERE):
        a.root = str(HERE / "testrun")                   # never write test assets over the real audio/ images/
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
