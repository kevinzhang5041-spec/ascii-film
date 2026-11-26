# ascii-film

Turns iPhone clips and photos into ASCII-art video, keeping the original audio, then joins them into one film.

> **Sound-driven film:** `./film` turns a soundtrack plus folders of scans into a fully automatic ASCII film, cut
> frame-accurately to the audio. See [SOUNDFILM.md](SOUNDFILM.md).

## Get it

macOS only (it uses Homebrew's ffmpeg, `sips` for HEIC photos and VideoToolbox for decoding). It needs Python 3.

```bash
git clone https://github.com/kevinzhang5041-spec/ascii-film.git ~/ascii-film
cd ~/ascii-film
./ascii setup
./film setup
```

Then `./ascii dashboard` for the clip-to-ASCII studio, or `./film dashboard` for the sound-driven film ([SOUNDFILM.md](SOUNDFILM.md)). `./film selftest` makes a synthetic test track and photos, then renders and verifies a test film.

Everything lives in this folder:

| folder      | what goes there                                                        |
|-------------|------------------------------------------------------------------------|
| `input/`    | your `.MOV` clips and `.HEIC` / `.JPG` photos (never modified or deleted) |
| `output/`   | one rendered `.mp4` per input file                                     |
| `previews/` | contact sheets                                                         |
| `final/`    | the assembled film, `film.mp4`                                         |

## First time

```bash
cd ~/ascii-film
./ascii setup
```

Checks for Python 3 and ffmpeg (installs ffmpeg with Homebrew if it's missing) and puts the Python packages in `.venv/` here. Your system Python isn't touched.

## The dashboard (easiest)

```bash
./ascii dashboard
```

This opens http://127.0.0.1:8777 in your browser. From there you can:

- **Add clips:** drag `.MOV` / `.HEIC` files onto the page, or copy them into `input/` in Finder. If a file with the same name is already there, the upload is refused; nothing in `input/` gets overwritten.
- **Preview:** pick a clip and scrub the slider. The frame re-renders as you change the look. *Original* shows the source frame.
- **Look:** pick a preset, or edit columns, colour, characters, tint, background, contrast and invert. Columns, colour and characters accept schedules like `0%:20,100%:160`. Use **Save as preset…** to store a look in `presets.json`.
- **Render:** *Render all* renders anything not rendered yet. Tick *include already rendered* to redo everything, or use *Re-render* on one clip. A clip rendered with a different look than the current one is marked **rendered · other look**.
- **Contact sheet:** grabs one frame every N seconds from every clip and tiles them into one image.
- **Film:** set the length (default `3:00`) and fade, then *Assemble film*. The timeline shows how the clips add up and what gets trimmed.

Stop the dashboard with Ctrl-C in the terminal.

## The same things from the terminal

Render everything new in `input/` with a preset:

```bash
./ascii --preset focus
```

Same, but override part of the preset:

```bash
./ascii --preset focus --charset detailed --contrast 1.2
```

Re-render one clip (outputs that already exist are skipped unless you pass `--force`):

```bash
./ascii IMG_0042.MOV --preset dissolve --force
```

Contact sheet, one frame every 10 seconds of every clip, saved to `previews/` and opened:

```bash
./ascii --preview 10 --preset focus
```

Assemble `output/` in filename order into an exact 3:00 film with 1-second fades, saved to `final/film.mp4`:

```bash
./ascii assemble --length 3:00 --fade 1
```

List the saved presets:

```bash
./ascii presets
```

Render a single test frame at 12.5 s with the standalone script:

```bash
.venv/bin/python ascii_film.py input/IMG_0042.MOV previews/test.png --preview 12.5 --cols 120 --color color
```

### Style options

Each option takes one value, or `time:value` keyframes separated by commas. Times are in seconds (`90:`) or a percentage of the clip (`50%:`, handy in presets). Numbers and colours fade smoothly between keyframes. Characters and invert switch at each keyframe.

| option       | example                              | meaning                                                   |
|--------------|--------------------------------------|-----------------------------------------------------------|
| `--cols`     | `120`, `"0:20,180:160"`              | characters across the frame                               |
| `--color`    | `mono`, `color`, `0.5`, `"0%:mono,100%:color"` | mono → full colour blend                        |
| `--charset`  | `standard`, `detailed`, `blocks`, `dots`, `minimal`, `binary`, `lines`, or a literal like `" .oO@"` | glyphs, sparse → dense |
| `--tint`     | `#ffffff`, `green`, `amber`, `moss`  | glyph colour in mono                                      |
| `--bg`       | `#000000`, `paper`, `night`          | background                                                |
| `--contrast` | `1.0`, `1.3`                         | luminance contrast (1 = unchanged)                        |
| `--invert`   | flag, or `"0:false,90:true"`         | dense glyphs in dark areas (use with a light `--bg`)      |

Output settings: `--fps 30`, `--size 1920` (long edge in pixels; portrait clips stay portrait), `--photo-seconds 5` (how long a photo is held).

### Presets

`presets.json` ships with `focus` (20 → 160 columns, mono → colour), `dissolve` (the reverse), `canopy`, `moss`, `terminal`, `paper` and `blocks`. Add your own in the dashboard, or by editing the file:

```json
"dusk": { "description": "warm, coarse", "cols": "90", "color": "0.6", "tint": "amber", "charset": "blocks" }
```

## Notes

- **Order:** the film joins `output/` in filename order. iPhone names (`IMG_0001…`) already sort by shooting order. To reorder, rename the files in `output/`, e.g. add `01_` prefixes.
- **Live Photos:** `IMG_1234.HEIC` and `IMG_1234.MOV` become `IMG_1234-heic.mp4` and `IMG_1234-mov.mp4`, so neither overwrites the other.
- **HDR:** iPhone HDR (HLG / Dolby Vision) clips are converted to standard range before rendering.
- **Mixed orientation:** in the assembled film, portrait clips are pillarboxed inside the most common frame size. Use `./ascii assemble --size 1080x1920` for a vertical film.
- **Too little footage:** if `output/` holds less than the length you ask for, assembly warns you and the film runs to the end of the footage.
- **Speed:** each clip is split into 6 time segments that decode (in hardware), render and encode in parallel, joined frame-exactly. On an M5 Pro, 1080p30 iPhone footage renders at about 330–350 fps, so 3 minutes of clips take about 16 s. `--jobs N` sets the number of segments. Clips shorter than about 10 s render as one piece.
