A command line tool written in Swift for Macintosh for creating anaglyph images for viewing with inexpensive red/cyan 3D glasses from side-by-side stereoscopic images.

## Usage

```
anaglyphtool [options] <image1> [image2] ...
```

Each input image is expected to be a side-by-side (SBS) stereo pair. Output files are written next to the input (or to `--output`) and named `<input>-anaglyph.<ext>`.

### Options

| Option | Description |
|---|---|
| `-o, --output <dir>` | Output directory (default: same as input) |
| `-m, --mode <mode>` | Anaglyph mode: `simple`, `optimized`, `dubois`, `grayscale` (default: `simple`) |
| `-n, --name` | Use mode-based naming: appends `-anaglyph-<mode>` instead of `-anaglyph` |
| `-q, --quality <0-1>` | JPEG compression quality (default: 0.9) |
| `-a, --auto` | Auto-detect the offset and vertical alignment from the stereo pair |
| `-f, --fast` | Fast mode for auto-detection (lower analysis resolution) |
| `--offset <pixels>` | Manual convergence offset (negative = closer convergence) |
| `-v, --verbose` | Show detailed processing information |
| `-h, --help` | Show help |

A bare signed number is accepted as shorthand for `--offset`: `-100` means `--offset -100`, and `+40` means `--offset 40`. An unsigned number (e.g. `100`) is treated as a filename, so positive offsets need the `+` prefix or the explicit `--offset` flag.

### Modes

- `simple` — Basic red/cyan channel separation (fast, good depth)
- `optimized` — Optimized matrices for better depth perception
- `dubois` — Dubois method for better color preservation
- `grayscale` — Grayscale anaglyph (reduces color rivalry)

### Convergence offset

The offset shifts the left and right views toward each other to control where the scene converges. Negative values bring the convergence point closer, which places nearer subjects at screen depth:

| Offset | Use for |
|---|---|
| `0` | No adjustment (default) |
| `-20` to `-40` | Distant subjects |
| `-40` to `-60` | General scenes |
| `-80` to `-100` | Close subjects |
| beyond `-100` | Very close / macro subjects |

Alternatively, `-a` block-matches the two views and sets the offset so the nearest content sits at screen depth. Everything else then appears behind the screen, which is the most comfortable arrangement for red/cyan glasses: crossed (in-front) parallax is where ghosting is most visible. Matching runs on a reduced copy of the image, so it takes a fraction of a second even for 10,000-pixel-wide pairs, and flat or repetitive areas (sky, backdrops, foliage) are ignored rather than allowed to skew the result. Any vertical misalignment between the two eyes is measured and corrected at the same time. Add `-v` to see the measured near and far disparity, the depth range as a fraction of the width, and a histogram of the depth distribution.

### Examples

```sh
anaglyphtool photo.jpg                 # Basic conversion
anaglyphtool -a photo.jpg              # Auto-detect offset
anaglyphtool -a -f photo.jpg           # Fast auto-detect
anaglyphtool -100 -m dubois photo.jpg  # Manual offset for a close subject
anaglyphtool -m dubois -a -f *.jpg     # Best color, fast auto, batch
```

Supported formats: JPEG, PNG, TIFF, HEIC, BMP

## Building

Open `anaglyphtool.xcodeproj` in Xcode and build, or from the command line:

```sh
xcodebuild -project anaglyphtool.xcodeproj -scheme anaglyphtool -configuration Release build
```
