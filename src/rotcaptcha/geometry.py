"""Circular-disc rotation geometry, matching real rotation-captcha layout.

Real captchas (Baidu-style, as in the yixiaowang2001 eval set) are 152x152 with
the puzzle content inside an inscribed circle on a white background. The label is
the *clockwise* rotation applied to the content (0-360); solving means rotating
back counter-clockwise by that amount.

Key invariant: if we rotate a square image about its center and then keep only the
inscribed disc (radius = side/2), every disc pixel maps from valid image content
under any rotation (rotation preserves distance-to-center), so corners never leak
background into the disc. That lets us rotate first, mask second, with no bleed.
"""

from __future__ import annotations

from PIL import Image

DEFAULT_SIZE = 152  # match real Baidu captchas
DEFAULT_BG = (255, 255, 255)  # white background outside the disc


def center_square(img: Image.Image) -> Image.Image:
    """Center-crop to the largest centered square."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _circle_mask(size: int) -> Image.Image:
    """L-mode mask: 255 inside the inscribed circle, 0 outside (antialiased)."""
    # Render at 4x and downsample for a smooth edge.
    scale = 4
    big = Image.new("L", (size * scale, size * scale), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(big).ellipse((0, 0, size * scale - 1, size * scale - 1), fill=255)
    return big.resize((size, size), Image.LANCZOS)


# Cache masks by size (they are pure functions of size).
_MASK_CACHE: dict[int, Image.Image] = {}


def circle_mask(size: int) -> Image.Image:
    m = _MASK_CACHE.get(size)
    if m is None:
        m = _circle_mask(size)
        _MASK_CACHE[size] = m
    return m


def make_disc_sample(
    img: Image.Image,
    angle_cw: float,
    size: int = DEFAULT_SIZE,
    bg=DEFAULT_BG,
) -> Image.Image:
    """Turn a base image into a captcha-like disc rotated clockwise by `angle_cw`.

    Steps: center-square -> resize -> rotate clockwise -> mask to inscribed disc
    over a solid background. Output is RGB `size`x`size`.
    """
    img = center_square(img).convert("RGB").resize((size, size), Image.BICUBIC)
    # PIL rotates counter-clockwise for positive angles; negate for clockwise.
    rotated = img.rotate(-angle_cw, resample=Image.BICUBIC, expand=False)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(rotated, (0, 0), circle_mask(size))
    return canvas


def restore(img: Image.Image, angle_cw: float) -> Image.Image:
    """Undo a clockwise rotation of `angle_cw` (for visualizing predictions)."""
    return img.rotate(angle_cw, resample=Image.BICUBIC, expand=False)
