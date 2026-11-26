# soundfilm: sound makes image

The soundtrack drives both the editing and the ASCII rendering of the film, frame-accurately at 24 fps.
Every onset in the track becomes a hard cut on exactly that frame. Loudness, spectral centroid and flatness then pick
the image pool, the grid resolution, the glyphs and the colour. The ASCII drawing is `ascii_film.py`'s renderer.

## Setup (once)

```bash
cd ~/ascii-film
./film setup
```

This creates `.venv/` here if needed and adds librosa, PyYAML and matplotlib to it. Nothing outside this folder changes.

## Your film

Put your files here:

| path | what |
|---|---|
| `audio/track.wav` | the soundtrack (`.mp3` / `.m4a` / `.aif` / `.flac` work too) |
| `images/bright/` | scans for high, glassy sounds (jpg / png / heic) |
| `images/texture/` | scans for the middle of the spectrum |
| `images/dark/` | scans for low, heavy sounds |

Then:

```bash
./film analyze                  # analysis.json + analysis.png (look at this first)
./film --preview 30 45          # 0:30-0:45, 640x360, fast
./film --preview 30 45 --debug  # same, with the why-overlay
./film                          # the full film, 1920x1080 -> renders/film_seed1.mp4
./film --debug                  # the full film with the overlay
./film --seed 7                 # a different edit from the same rules
./film verify                   # sync checks on the newest render
./film stills                   # a contact sheet of telling frames from the newest render
./film plot 30 45               # analysis plot zoomed to 0:30-0:45
```

A preview shows exactly the frames the full render will have there, only smaller. The edit is computed for the whole track first.

### Choosing photos by hand: the sorter

In the dashboard, click **Sort photos…** (or go to http://127.0.0.1:8778/sort).

- **Bringing photos in:** they land in `images/inbox/`, which the film doesn't use. You can bring them in three ways:
  - **Import Photos selection:** select photos in the Photos app first, then click this.
  - **Import album…:** a whole Photos album.
  - **Add files…:** pick files, or copy them into `images/inbox/` in Finder.

  Videos are skipped, and a photo you imported before isn't imported again. The first time, macOS may ask whether the dashboard can control Photos; click OK.
- **Sorting:** each photo shows the automatic **suggestion** (→ bright / texture / dark), which you can ignore. The thumbnails show the 16:9 crop the film uses. *whole* shows the full photo.
  - Click selects, ⇧-click selects a range, ⌘-click adds to the selection, ⌘A selects all.
  - **1** bright · **2** texture · **3** dark · **S** take the suggestion · **X** reject · **0** back to the inbox.
  - Space (or a double-click) opens one photo large, with what the film crops away dimmed. There, 1/2/3/S/X move it and go on to the next, and ← → step through. That's the quickest way through an inbox.
- **How often photos repeat:** the strip above the grid shows, for each pool:
  - how many photos it has, how many cuts draw from it, and how many times each photo appears on average and at most;
  - unused photos, and the shortest gap before a photo comes back;
  - roughly how many photos it would need for about 5 appearances each.

  Green means few repeats, amber several, red a lot. Every photo in a pool gets a ×N badge showing how many cuts it has in the film. Choose *most used first* to find the worst repeaters, and click a pool's card to open it in that order. The large view also tells you when the photo first appears and how long it's on screen. All of this reflects the next render with your saved settings. Tune's live numbers show the same figures for unsaved changes.
- **Where the files go:** sorting moves the file between `images/inbox`, `bright`, `texture`, `dark` and `rejected`. Nothing is deleted, and a name that's taken gets `-2`.
- **Cropping and rotating by hand:** in the large view press **C** (or *✂ Crop / rotate…*), or press C on a selected thumbnail in the grid. A 16:9 frame appears over the photo.
  - Drag inside it to move it, drag a corner to resize, scroll to zoom. Arrow keys nudge it (⇧ for bigger steps), and + / − change its size.
  - **⟲ / ⟳** (or **[** **]**) turn the photo 90°, which is handy for fitting more of a portrait photo into the frame. The **tilt** slider (or **,** **.**, with ⇧ for 5°) turns it up to ±45°, and the corners that opens up are black (empty space in the film).
  - **Enter** saves, **R** resets (no rotation, centre of the photo), **Esc** cancels.

  The crop is saved in `images/crops.json`, and the photo file itself is never changed. Cropped photos get a ✂ in the grid, their thumbnails show your crop, and the photo panel beside the player outlines it. The film then uses your crop instead of the centre of the photo. Cropped photos skip the random crop/zoom, so they're shown exactly as you framed them; to change that, untick **keep my crops exact** in Tune → Treatments.

### Sorting photos automatically

```bash
~/ascii-film/film sort ~/Desktop/scans
```

This scores every photo in the folder or folders you name and copies each into `images/bright`, `images/texture` or `images/dark`. It judges the photo by what the film will show of it: the 16:9 centre crop, auto-levelled:

- **texture:** the most detailed third (leaves, gravel, busy scenes).
- **bright / dark:** the rest, split at their median brightness.

The copies are JPEGs up to 2560 px with the EXIF removed, so no GPS location travels with them. Your originals aren't touched, and a name that's already in a pool is skipped. Open `images_sorted.png` to see the result. `images/sort_report.csv` lists each photo's scores. To overrule a choice, move the copy to another pool folder.

- `--dry-run` writes only the report and the contact sheet.
- `--texture-share 0.5` sends half the photos to texture.
- `--root DIR` sorts into another project folder.

## Dashboard

```bash
~/ascii-film/film dashboard
```

This opens http://127.0.0.1:8778 in your browser. Stop it with Ctrl-C in that terminal. One page holds everything:

- **Inputs:** click *Add…* on the track, bright, texture or dark box to open the macOS file chooser (Cmd-click or Cmd-A picks several images). The files are copied in and your originals stay where they are. In Safari or Chrome you can also drag files from Finder onto a box. The Claude app's browser pane doesn't accept drags from Finder, so use *Add…* there. To swap files:
  - **Track:** click *Replace…* on the track box, or drop a new track on it.
  - **Photo:** click a thumbnail, then *Replace…*. The new photo goes into that pool with its own name.
  - **Same name:** adding a photo whose name is already in the pool offers to replace it.

  The page always asks first. The old file is moved to `.work/replaced/<date-time>/` in the project folder, not deleted, so you can move it back. After a new track arrives, the dashboard re-analyses it automatically.
- **Photo panel:** *Show photo* opens a panel beside the player with the real photo behind the frame on screen. It follows the video frame by frame and outlines the part of the photo the film shows. Fading and black frames dim it. Renders made before this panel existed show the photo, but their outline only marks the centre crop.
- **Player:** set the preview range, then *Render preview* or *Render full film*. The debug overlay, the seed, and a sync check after rendering are optional. A finished render opens in the player.
- **Analysis:** *Analyse* draws `analysis.png`. *Plot preview range* zooms the plot to the preview range.
- **Renders:** play, verify, export stills and open the contact sheet for any render. *sync report* shows the verify results.
- **Tune:** sliders, switches and colour pickers for every rule: cuts and timing, photo pools, resolution, character sets, colour, brightness and the treatments. Hover over a label for what it does. Before you save, the right side shows the effect on the whole film: cuts, flashes, the cuts per pool, how much is black, character sets, columns and colour. Each changed number also shows what it was. It also draws the first cuts from the preview start with the new settings. Changed rows turn amber, ↺ puts one back to its default, and *Discard* drops all unsaved changes. *Save* writes only the changed values into `config.yaml`, leaving comments and layout alone, and keeps the old version. *Save & render preview* saves, then renders the preview range. Onset sensitivity re-analyses the track, so its numbers take a second or two.
- **config.yaml:** edit the rules right on the page. *Save* checks the file first and refuses one that wouldn't load. Every save copies the previous version to `.work/config-history/`. *Save & render preview* saves and then re-renders the current range. ⌘S saves.

Jobs run one at a time, and the Activity panel shows their progress and output. `--root testrun` points the dashboard at the test folder.

## Tuning

Every rule and threshold is in **`config.yaml`**, with a comment on each line. Edit it and re-render. The analysis is cached in `analysis.json`; it's only recomputed when the track or the `analysis:` block changes. To see how a setting moved things, look at `analysis.png` (it draws the thresholds) or at a `--debug` preview.

| rule | config |
|---|---|
| onset → hard cut, no repeat within N cuts | `editing.no_repeat_within` |
| onset strength → hold 1-4 frames | `editing.hold`. Strength mixes spectral jump and loudness: `analysis.onset_loudness_weight` |
| between onsets: hold, or decay to black | `editing.between_onsets` |
| silence → pure black | `editing.silence.threshold_db` |
| centroid → bright / texture / dark pool | `editing.pools` |
| very strong onset → 1-frame invert flash | `editing.flash` |
| RMS → 15-200 columns, smoothed | `ascii.resolution` |
| centroid → glyph ramp | `ascii.charset.bands` (any characters; ramps are auto-sorted by ink) |
| flatness → mono / colour | `ascii.color` |
| crop/zoom, char jitter, gate weave, grain, colour shift, layering, exposure | `treatments.*` (each `enabled` + `intensity`) |
| more / fewer onsets detected | `analysis.onset_delta` (lower = more), `analysis.onset_wait_ms` |

Defaults live in `soundfilm/defaults.yaml`. A key you delete from `config.yaml` falls back to its default. A misspelt key prints a warning.

## What a render leaves in `renders/`

- `film_seed1.mp4`: H.264 1920x1080 24 fps (x264 `veryfast`, CRF 18) plus the original audio (AAC 320k)
- `film_seed1.edl.csv`: one line per frame: cut / hold / decay / black / silence, the image, and the reason
- `film_seed1.json`: seed, frame range, sizes, config digest
- `film_seed1.fp.npy`: a thumbnail of every frame as it went into the encoder (used by `verify`)
- `film_seed1.verify.json`, `film_seed1_contact.png`, `film_seed1_stills/`: written by `verify` and `stills`

Re-rendering with the same seed overwrites that seed's render. Copy or rename a version you want to keep.

## Test run

`testrun/` holds a synthetic 20 s track and 90 placeholder images. The track has silence, clicks from weak to strong, glassy pings, low tones, noise bursts, a dense glitch passage and a crescendo. `audio/track_events.json` records when every sound really starts. Re-run everything with:

```bash
./film selftest
```

That analyses the track, renders the full film, a `--debug` film and a 13-17 s preview, and verifies all three. It also exports stills and checks determinism: the same seed twice must give an identical file, and another seed a different edit. It takes about 15 s. To regenerate the test assets: `./film testgen --force` (it only overwrites files it made). Any command takes `--root testrun` to work on the test folder instead of your real one.

### What `verify` checks

- **plan**: every cut in the EDL sits on an onset frame, and every onset frame has a cut
- **frames**: the decoded video matches, frame by frame, what the renderer sent to the encoder. So nothing was dropped, duplicated or shifted.
- **audio**: the muxed soundtrack is cross-correlated with the original at points from start to end. It passes if it's within 1 ms everywhere, which also rules out drift. A delay of 10 ms shows up as 480 samples and fails. Onsets re-detected from the AAC are listed for information. In a dense track a few land on a neighbouring peak, which is the detector, not timing.
- **blind**: from pixels and sound only: for each isolated onset, does the biggest burst of new light happen on that exact frame?
- **duration**: the video and audio streams match the track (or the preview range)
- **truth** (test track only): detected onsets vs the times the sounds were actually placed at

## Code

`soundfilm/`: `analysis.py` (librosa features, `analysis.json`), `score.py` (the mapping rules, one decision per frame), `frames.py` (draws a frame: crop, layer, colour, exposure, then `ascii_film`, then grain and jitter), `render.py` (parallel segments → x264 → mux), `debug.py` (overlay), `plot.py`, `verify.py`, `testgen.py`, `dashboard.py` + `web/index.html` (`./film dashboard`), `__main__.py` (the `./film` command).

Speed: about 190 frames/s at 1080p on an Apple M5 Pro, so a 3-minute film takes about 25 s.
