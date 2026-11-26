"""analysis.png: waveform, onsets, RMS, centroid, flatness over time, with the config's thresholds and
(when images are available) what the score did with them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .analysis import decode_audio
from .debug import POOL_RGB

KIND_RGB = {"cut": (1.0, 1.0, 1.0), "hold": (0.6, 0.6, 0.6), "decay": (0.3, 0.3, 0.3), "black": (0, 0, 0),
            "silence": (0.25, 0, 0)}


def plot(an, cfg: dict, out_png: Path, sc=None, span=None, events=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fps, n = an.fps, an.n
    t = np.arange(n) / fps
    lo, hi = span if span else (0.0, an.duration)
    ed, asc = cfg["editing"], cfg["ascii"]
    pool_c = {k: tuple(c / 255 for c in v) for k, v in POOL_RGB.items()}

    rows = 6 if sc is not None else 5
    ratios = [1.3, 1.1, 1, 1, 1] + ([0.35] if sc is not None else [])
    fig, ax = plt.subplots(rows, 1, figsize=(20, 13), sharex=True, gridspec_kw=dict(height_ratios=ratios))
    fig.patch.set_facecolor("white")

    # waveform (min/max envelope at 400 points per second)
    sr = 8000
    y = decode_audio(an.track, sr)
    step = sr // 400
    m = len(y) // step
    yy = y[:m * step].reshape(m, step)
    tw = (np.arange(m) + 0.5) * step / sr
    ax[0].fill_between(tw, yy.min(1), yy.max(1), color="0.25", lw=0)
    ax[0].set_ylabel("waveform")
    ax[0].set_ylim(-1.05, 1.05)

    of, os_ = an.onset_frames, an.strength
    for a in ax[:5]:
        for f in of:
            a.axvline(f / fps, color="tab:red", alpha=0.18, lw=0.8)
    if events:
        et = [e["time"] for e in events]
        ax[0].plot(et, np.full(len(et), 0.97), "v", color="tab:green", ms=5, label="test-track events (truth)")
        ax[0].legend(loc="upper right", fontsize=8)

    # onset envelope + detected onsets (height = strength, colour = pool chosen)
    ax[1].plot(t, an.f["onset_env"], color="0.45", lw=0.8, label="onset envelope")
    cols = [pool_c[s["pool"]] for s in sc.shots] if sc is not None else "tab:red"
    ax[1].scatter(of / fps, os_, c=cols, s=22, zorder=3, edgecolors="k", linewidths=0.4,
                  label="onset (snapped), height = strength")
    fl = ed["flash"]
    if fl["enabled"]:
        ax[1].axhline(fl["strength_above"], color="tab:orange", ls="--", lw=0.9, label="flash threshold")
    ax[1].set_ylabel("onsets")
    ax[1].set_ylim(0, 1.15)
    ax[1].legend(loc="lower center", fontsize=8, framealpha=0.7, ncol=3)

    # RMS -> silence and columns
    ax[2].step(t, an.f["rms_norm"], where="post", color="tab:blue", lw=0.9, label="RMS (0-1 over dB range)")
    sil = np.clip(1 + ed["silence"]["threshold_db"] / cfg["analysis"]["rms_range_db"], 0, 1)
    ax[2].axhline(sil, color="tab:red", ls="--", lw=0.9, label=f"silence < {ed['silence']['threshold_db']} dB")
    ax[2].set_ylabel("RMS")
    ax[2].set_ylim(0, 1.05)
    if sc is not None:
        a2 = ax[2].twinx()
        a2.step(t, [r["cols"] for r in sc.rows], where="post", color="tab:purple", lw=0.9, alpha=0.8)
        a2.set_ylabel("columns", color="tab:purple")
    ax[2].legend(loc="upper left", fontsize=8)

    # centroid -> pools and charsets
    pl = ed["pools"]
    ax[3].axhspan(pl["bright_above"], 1.05, color=pool_c["bright"], alpha=0.15)
    ax[3].axhspan(pl["dark_below"], pl["bright_above"], color=pool_c["texture"], alpha=0.12)
    ax[3].axhspan(0, pl["dark_below"], color=pool_c["dark"], alpha=0.15)
    ax[3].step(t, an.f["centroid"], where="post", color="k", lw=0.9, label="spectral centroid")
    for b in asc["charset"]["bands"]:
        if b["above"] > 0:
            ax[3].axhline(b["above"], color="0.4", ls=":", lw=0.8)
        ax[3].text(lo + 0.002 * (hi - lo), b["above"] + 0.01, f"chars: {b['name']}", fontsize=7, color="0.3")
    ax[3].set_ylabel("centroid")
    ax[3].set_ylim(0, 1.05)
    ax[3].legend(loc="upper right", fontsize=8, title="bands: bright / texture / dark pool", title_fontsize=7)

    # flatness -> colour mix
    ax[4].step(t, an.f["flatness"], where="post", color="tab:pink", lw=0.9, label="spectral flatness (noisy = 1)")
    if sc is not None:
        ax[4].step(t, [r["mix"] for r in sc.rows], where="post", color="tab:green", lw=0.9, alpha=0.8,
                   label="colour mix (1 = full colour)")
    ax[4].set_ylabel("flatness")
    ax[4].set_ylim(0, 1.05)
    ax[4].legend(loc="upper right", fontsize=8)

    if sc is not None:
        strip = np.array([[KIND_RGB[r["kind"]] if r["kind"] != "cut" else pool_c[r["pool"]] for r in sc.rows]])
        ax[5].imshow(strip, aspect="auto", extent=(0, n / fps, 0, 1), interpolation="nearest")
        ax[5].set_yticks([])
        ax[5].set_ylabel("edit", rotation=0, ha="right", va="center")

    ax[-1].set_xlabel("seconds")
    ax[-1].set_xlim(lo, hi)
    n_cuts = len(sc.shots) if sc is not None else len(of)
    fig.suptitle(f"{Path(an.track).name}: {an.duration:.2f} s, {n} frames at {fps} fps, {len(of)} onsets, "
                 f"{n_cuts} cuts", fontsize=12)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def events_for(an):
    """The test track's ground truth (audio/track_events.json), if there is one."""
    p = an.track.with_name(an.track.stem + "_events.json")
    return json.loads(p.read_text())["events"] if p.exists() else None


def for_root(root: Path, cfg: dict, an, span=None, log=print) -> Path:
    """ROOT/analysis.png (or analysis_A-Bs.png for a span): the sound, plus the edit when there are images."""
    from . import score
    sc = None
    try:
        from .sets import sections_for
        pools = score.gather_pools(root, cfg, root / ".work")
        sc = score.build_score(an, cfg, pools, cfg["seed"], sections=sections_for(root, cfg, pools, an.fps, an.n))
    except SystemExit:
        pass                                            # no images yet: plot the sound only
    name = "analysis.png" if not span else f"analysis_{span[0]:g}-{span[1]:g}s.png"
    out = plot(an, cfg, Path(root) / name, sc, span, events_for(an))
    log(f"wrote {out}")
    return out
