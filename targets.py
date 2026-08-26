"""Target images for stimulus and simulation.

A target is a float array in [0, 1], shape (GRID_H, GRID_W): the luminance the
flickering cursor takes on at each scan position. Text targets echo the paper's
"NO" / "NO CAMERAS" demonstrations.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


def text_target(text="NO", grid_w=None, grid_h=None):
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    # Render large, then downsample to the scan grid.
    W, H = grid_w * 16, grid_h * 16
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    size = H
    font = None
    while size > 8:
        try:
            font = ImageFont.truetype("arialbd.ttf", size)
        except OSError:
            font = ImageFont.load_default()
            break
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= W * 0.9 and bbox[3] - bbox[1] <= H * 0.9:
            break
        size = int(size * 0.9)
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (W - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (H - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    small = img.resize((grid_w, grid_h), Image.LANCZOS)
    return np.asarray(small, dtype=float) / 255.0


def image_target(path, grid_w=None, grid_h=None):
    grid_w = grid_w or config.GRID_W
    grid_h = grid_h or config.GRID_H
    img = Image.open(path).convert("L").resize((grid_w, grid_h), Image.LANCZOS)
    return np.asarray(img, dtype=float) / 255.0


def load_target(spec, grid_w=None, grid_h=None):
    """spec: 'text:NO CAMERAS' or a path to an image file."""
    if spec.startswith("text:"):
        return text_target(spec[5:], grid_w, grid_h)
    return image_target(spec, grid_w, grid_h)
