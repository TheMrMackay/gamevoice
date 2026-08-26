"""Generate the GameVoice application icon.

The mark is a speech bubble - dialogue - with a level meter inside it, for the
voice reading that dialogue aloud. Both halves of what the app does, in one
shape.

Everything is drawn at 8x and downsampled, because Pillow's shape drawing has no
anti-aliasing of its own and a 16 px icon drawn directly looks chewed. Elements
are sized as fractions of the canvas so the same code produces every resolution.

Two colourways: indigo when idle, green while listening, so the tray icon says
which it is at a glance.

    python tools/make_icon.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

# Windows picks the nearest size from an .ico, so shipping the small ones as
# real renders beats letting it downscale a 256 and blur the bars.
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SUPERSAMPLE = 8

THEMES = {
    "idle": ("#4338CA", "#7C3AED"),      # indigo -> violet
    "active": ("#047857", "#10B981"),    # deep green -> emerald
}

BUBBLE = "#FFFFFF"
# The bars sit in the bubble, so they take the darker end of the background
# gradient rather than a fourth colour.
BAR_ALPHA = 255


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i: i + 2], 16) for i in (0, 2, 4))


def _gradient(size: int, top: str, bottom: str) -> Image.Image:
    """A vertical gradient, built once as an array rather than per row."""
    start, end = np.array(_rgb(top), np.float32), np.array(_rgb(bottom), np.float32)
    # Diagonal reads better than straight vertical on a rounded square.
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :]
    blend = np.clip((ys * 0.75 + xs * 0.25), 0.0, 1.0)[:, :, None]
    pixels = start[None, None, :] * (1 - blend) + end[None, None, :] * blend
    return Image.fromarray(pixels.astype(np.uint8), "RGB")


def _rounded_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255
    )
    return mask


def render(size: int, theme: str = "idle") -> Image.Image:
    """One icon at one size, fully anti-aliased."""
    top, bottom = THEMES[theme]
    canvas = size * SUPERSAMPLE
    s = float(canvas)

    plate = _gradient(canvas, top, bottom)
    plate.putalpha(_rounded_mask(canvas, radius=s * 0.225))

    layer = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # --- speech bubble ---------------------------------------------------
    left, right = s * 0.155, s * 0.845
    top_y, bottom_y = s * 0.185, s * 0.665
    draw.rounded_rectangle(
        (left, top_y, right, bottom_y), radius=s * 0.115, fill=BUBBLE
    )
    # Tail, drawn as a triangle tucked under the body so the join is invisible.
    draw.polygon(
        [
            (s * 0.285, bottom_y - s * 0.02),
            (s * 0.475, bottom_y - s * 0.02),
            (s * 0.325, s * 0.845),
        ],
        fill=BUBBLE,
    )

    # --- level meter inside the bubble -----------------------------------
    # Heights rise and fall so it reads as sound rather than as a bar chart.
    # Five bars turn to mush in a 16 px tray icon, so small sizes get three
    # thicker ones instead - the shape survives, the detail is what goes.
    if size <= 20:
        heights = (0.55, 1.0, 0.42)
        bar_fraction, gap_fraction = 0.150, 0.105
    elif size <= 32:
        heights = (0.50, 0.85, 1.0, 0.45)
        bar_fraction, gap_fraction = 0.115, 0.075
    else:
        heights = (0.42, 0.72, 1.0, 0.62, 0.30)
        bar_fraction, gap_fraction = 0.088, 0.055

    bar_colour = _rgb(top) + (BAR_ALPHA,)
    span = right - left
    bar_width = span * bar_fraction
    gap = span * gap_fraction
    total = len(heights) * bar_width + (len(heights) - 1) * gap
    start_x = left + (span - total) / 2.0
    centre_y = (top_y + bottom_y) / 2.0
    tallest = (bottom_y - top_y) * 0.56

    for index, factor in enumerate(heights):
        x0 = start_x + index * (bar_width + gap)
        half = tallest * factor / 2.0
        draw.rounded_rectangle(
            (x0, centre_y - half, x0 + bar_width, centre_y + half),
            radius=bar_width / 2.0,
            fill=bar_colour,
        )

    plate.alpha_composite(layer)
    return plate.resize((size, size), Image.LANCZOS)


def build() -> list[Path]:
    ASSETS.mkdir(exist_ok=True)
    written: list[Path] = []

    for theme in THEMES:
        frames = [render(size, theme) for size in ICO_SIZES]
        largest = frames[-1]

        png = ASSETS / f"icon-{theme}.png"
        largest.save(png)
        written.append(png)

        ico = ASSETS / ("icon.ico" if theme == "idle" else f"icon-{theme}.ico")
        # append_images embeds these exact renders. Without it Pillow would
        # resize the 256 down to every size, throwing away the simplified
        # small-size artwork that is the whole reason for rendering each one.
        largest.save(
            ico,
            format="ICO",
            sizes=[(s, s) for s in ICO_SIZES],
            append_images=frames[:-1],
        )
        written.append(ico)

    written.append(_preview())
    return written


def _preview() -> Path:
    """A sheet showing every size on light and dark, to check legibility."""
    sizes = (16, 24, 32, 48, 64, 128)
    pad, gap = 24, 20
    width = pad * 2 + sum(sizes) + gap * (len(sizes) - 1)
    row = max(sizes) + pad * 2
    sheet = Image.new("RGB", (width, row * 2), "#FFFFFF")
    sheet.paste(Image.new("RGB", (width, row), "#1B1D22"), (0, row))

    for row_index, theme in enumerate(("idle", "active")):
        x = pad
        for size in sizes:
            icon = render(size, theme)
            y = row_index * row + pad + (max(sizes) - size) // 2
            sheet.paste(icon, (x, y), icon)
            x += size + gap

    path = ASSETS / "icon-preview.png"
    sheet.save(path)
    return path


if __name__ == "__main__":
    for path in build():
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
