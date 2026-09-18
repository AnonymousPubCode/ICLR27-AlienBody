"""Pixel art sprites for AlienBody renderer.

All sprites are hardcoded numpy arrays — zero external dependencies.
Inspired by Crafter/MiniGrid aesthetic: clean 16x16 pixel art.

Sprite palette:
  T = transparent (will be composited over background)
  Each sprite is a 16x16 grid of color indices → mapped to RGB at render time.
"""
from __future__ import annotations
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Transparent marker
# ═══════════════════════════════════════════════════════════════════
T = -1  # transparent pixel

# ═══════════════════════════════════════════════════════════════════
# Agent Sprites (16x16, 4 directions)
# Astronaut design: helmet (white dome), visor (blue), body (grey), legs
# ═══════════════════════════════════════════════════════════════════

# Agent sprites fill more of the 16x16 tile for better visibility.
# Larger head, wider body, directional visor clearly shows facing.

_AGENT_UP = [
    [T,T,T,T,9,9,9,9,9,9,9,9,T,T,T,T],  # helmet rim
    [T,T,T,9,1,1,1,1,1,1,1,1,9,T,T,T],  # visor top (blue = facing up)
    [T,T,9,1,1,1,1,1,1,1,1,1,1,9,T,T],  # visor wide
    [T,T,9,1,1,1,1,1,1,1,1,1,1,9,T,T],  # visor
    [T,T,9,0,0,0,0,0,0,0,0,0,0,9,T,T],  # helmet bottom (white)
    [T,T,T,9,9,0,0,0,0,0,0,9,9,T,T,T],  # chin
    [T,T,5,5,5,5,5,5,5,5,5,5,5,5,T,T],  # shoulders (grey suit)
    [T,T,5,5,0,0,0,0,0,0,0,0,5,5,T,T],  # torso
    [T,5,5,5,0,0,2,2,2,2,0,0,5,5,5,T],  # torso + red badge
    [T,5,5,5,0,0,2,2,2,2,0,0,5,5,5,T],  # torso
    [T,T,5,5,0,0,0,0,0,0,0,0,5,5,T,T],  # waist
    [T,T,T,5,5,5,5,5,5,5,5,5,5,T,T,T],  # belt
    [T,T,T,5,5,5,T,T,T,T,5,5,5,T,T,T],  # legs
    [T,T,T,5,5,5,T,T,T,T,5,5,5,T,T,T],  # legs
    [T,T,T,9,9,9,T,T,T,T,9,9,9,T,T,T],  # boots
    [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T],
]

_AGENT_RIGHT = [
    [T,T,T,T,9,9,9,9,9,9,9,T,T,T,T,T],
    [T,T,T,9,0,0,0,0,0,1,1,9,T,T,T,T],  # visor right (blue)
    [T,T,9,0,0,0,0,0,1,1,1,1,9,T,T,T],  # visor right wide
    [T,T,9,0,0,0,0,0,1,1,1,1,9,T,T,T],  # visor
    [T,T,9,0,0,0,0,0,0,0,0,0,9,T,T,T],  # helmet
    [T,T,T,9,9,0,0,0,0,0,9,9,T,T,T,T],
    [T,T,5,5,5,5,5,5,5,5,5,5,T,T,T,T],  # shoulders
    [T,T,5,5,0,0,0,0,0,0,5,5,5,5,T,T],  # torso + arm right
    [T,T,5,5,0,0,2,2,0,0,5,5,5,5,T,T],  # badge
    [T,T,5,5,0,0,2,2,0,0,5,5,5,5,T,T],
    [T,T,5,5,0,0,0,0,0,0,5,5,T,T,T,T],
    [T,T,T,5,5,5,5,5,5,5,5,T,T,T,T,T],
    [T,T,T,5,5,5,T,T,5,5,5,T,T,T,T,T],
    [T,T,T,5,5,5,T,T,5,5,5,T,T,T,T,T],
    [T,T,T,9,9,9,T,T,9,9,9,T,T,T,T,T],
    [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T],
]

_AGENT_DOWN = [
    [T,T,T,T,9,9,9,9,9,9,9,9,T,T,T,T],  # helmet rim
    [T,T,T,9,0,0,0,0,0,0,0,0,9,T,T,T],  # helmet top (white)
    [T,T,9,0,0,0,0,0,0,0,0,0,0,9,T,T],
    [T,T,9,0,0,8,8,0,0,8,8,0,0,9,T,T],  # face: eyes (light blue)
    [T,T,9,0,0,0,0,2,2,0,0,0,0,9,T,T],  # mouth (red)
    [T,T,T,9,9,0,0,0,0,0,0,9,9,T,T,T],  # chin
    [T,T,5,5,5,5,5,5,5,5,5,5,5,5,T,T],  # shoulders
    [T,T,5,5,0,0,0,0,0,0,0,0,5,5,T,T],  # torso
    [T,5,5,5,0,0,2,2,2,2,0,0,5,5,5,T],  # badge
    [T,5,5,5,0,0,2,2,2,2,0,0,5,5,5,T],
    [T,T,5,5,0,0,0,0,0,0,0,0,5,5,T,T],
    [T,T,T,5,5,5,5,5,5,5,5,5,5,T,T,T],  # belt
    [T,T,T,5,5,5,T,T,T,T,5,5,5,T,T,T],  # legs
    [T,T,T,5,5,5,T,T,T,T,5,5,5,T,T,T],
    [T,T,T,9,9,9,T,T,T,T,9,9,9,T,T,T],  # boots
    [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T],
]

_AGENT_LEFT = [
    [T,T,T,T,T,9,9,9,9,9,9,9,T,T,T,T],
    [T,T,T,T,9,1,1,0,0,0,0,0,9,T,T,T],  # visor left (blue)
    [T,T,T,9,1,1,1,1,0,0,0,0,0,9,T,T],  # visor left wide
    [T,T,T,9,1,1,1,1,0,0,0,0,0,9,T,T],
    [T,T,T,9,0,0,0,0,0,0,0,0,0,9,T,T],
    [T,T,T,T,9,9,0,0,0,0,0,9,9,T,T,T],
    [T,T,T,T,5,5,5,5,5,5,5,5,5,5,T,T],
    [T,T,5,5,5,5,0,0,0,0,0,0,5,5,T,T],  # arm left
    [T,T,5,5,5,5,0,0,2,2,0,0,5,5,T,T],
    [T,T,5,5,5,5,0,0,2,2,0,0,5,5,T,T],
    [T,T,T,T,5,5,0,0,0,0,0,0,5,5,T,T],
    [T,T,T,T,T,5,5,5,5,5,5,5,5,T,T,T],
    [T,T,T,T,T,5,5,5,T,T,5,5,5,T,T,T],
    [T,T,T,T,T,5,5,5,T,T,5,5,5,T,T,T],
    [T,T,T,T,T,9,9,9,T,T,9,9,9,T,T,T],
    [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T,T],
]

# ═══════════════════════════════════════════════════════════════════
# Target Sprite (16x16) — glowing star/flag
# ═══════════════════════════════════════════════════════════════════

_TARGET = [
    [T,T,T,T,T,T,T,4,4,T,T,T,T,T,T,T],  # star tip
    [T,T,T,T,T,T,4,4,4,4,T,T,T,T,T,T],
    [T,T,T,T,T,4,4,7,7,4,4,T,T,T,T,T],  # inner glow (orange)
    [T,T,T,T,4,4,7,7,7,7,4,4,T,T,T,T],
    [4,4,4,4,4,4,7,7,7,7,4,4,4,4,4,4],  # star horizontal bar
    [T,4,4,4,4,7,7,7,7,7,7,4,4,4,4,T],
    [T,T,4,4,4,7,7,4,4,7,7,4,4,4,T,T],
    [T,T,T,4,4,7,4,4,4,4,7,4,4,T,T,T],  # star center
    [T,T,T,4,4,7,4,4,4,4,7,4,4,T,T,T],
    [T,T,4,4,4,7,7,4,4,7,7,4,4,4,T,T],
    [T,4,4,4,4,7,7,7,7,7,7,4,4,4,4,T],
    [4,4,4,4,4,4,7,7,7,7,4,4,4,4,4,4],  # star horizontal bar bottom
    [T,T,T,T,4,4,7,7,7,7,4,4,T,T,T,T],
    [T,T,T,T,T,4,4,7,7,4,4,T,T,T,T,T],
    [T,T,T,T,T,T,4,4,4,4,T,T,T,T,T,T],
    [T,T,T,T,T,T,T,4,4,T,T,T,T,T,T,T],  # star bottom tip
]

# ═══════════════════════════════════════════════════════════════════
# Obstacle Sprite (16x16) — brick wall
# ═══════════════════════════════════════════════════════════════════

_W = 9  # brick dark (maroon)
_G = 5  # mortar (grey)
_OBSTACLE = [
    [_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W],
    [_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W],
    [_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W],
    [_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G],  # mortar line
    [_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W],
    [_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W],
    [_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W,_W,_G,_W,_W],
    [_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G],
    [_W,_W,_G,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W],
    [_W,_W,_G,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W],
    [_W,_W,_G,_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_W],
    [_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G],
    [_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_G,_W,_W,_W],
    [_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_G,_W,_W,_W],
    [_W,_W,_W,_W,_W,_W,_G,_W,_W,_W,_W,_W,_G,_W,_W,_W],
    [_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G,_G],
]

# ═══════════════════════════════════════════════════════════════════
# Floor Tile patterns (16x16) — subtle checker for depth
# ═══════════════════════════════════════════════════════════════════

# Two variants for checkerboard pattern
# "Light" tile: color + subtle inner highlight
# "Dark" tile: color + subtle inner shadow

# ═══════════════════════════════════════════════════════════════════
# Color mapping: sprite index → RGB
# ═══════════════════════════════════════════════════════════════════

# Extended palette for sprite rendering (ARC colors + extra shades)
SPRITE_PALETTE = {
    0: (245, 245, 240),  # white/cream (helmet, torso highlight)
    1: (0, 116, 217),    # blue (visor)
    2: (255, 65, 54),     # red (badge/accent)
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow (star/target)
    5: (120, 130, 140),   # grey (suit)
    6: (240, 18, 190),    # magenta
    7: (255, 160, 50),    # orange (glow)
    8: (127, 219, 255),   # light blue (eyes)
    9: (70, 50, 50),      # dark brown (boots, bricks)
}

# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

AGENT_SPRITES = {
    0: np.array(_AGENT_UP, dtype=np.int8),     # up
    1: np.array(_AGENT_RIGHT, dtype=np.int8),  # right
    2: np.array(_AGENT_DOWN, dtype=np.int8),   # down
    3: np.array(_AGENT_LEFT, dtype=np.int8),   # left
}

TARGET_SPRITE = np.array(_TARGET, dtype=np.int8)
OBSTACLE_SPRITE = np.array(_OBSTACLE, dtype=np.int8)

SPRITE_SIZE = 16  # all sprites are 16x16


def sprite_to_rgb(sprite: np.ndarray, palette: dict = None) -> tuple:
    """Convert a sprite index array to (RGB, alpha) using vectorized numpy.

    Returns:
        rgb: (H, W, 3) uint8
        alpha: (H, W) uint8  (255 = opaque, 0 = transparent)
    """
    if palette is None:
        palette = SPRITE_PALETTE

    # Build a lookup table: index -> (R, G, B)
    max_idx = max(max(palette.keys()), 9) + 1
    lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)  # +1 for safety
    for idx, color in palette.items():
        lut[idx] = color

    # Mask: transparent where sprite == T (-1)
    opaque = sprite >= 0  # bool array (H, W)
    safe_idx = np.clip(sprite, 0, max_idx)  # replace -1 with 0 for indexing

    rgb = lut[safe_idx]  # (H, W, 3) — vectorized lookup
    alpha = (opaque.astype(np.uint8) * 255)  # (H, W)
    return rgb, alpha


def blit_sprite(
    canvas: np.ndarray,
    sprite: np.ndarray,
    y: int, x: int,
    cell_px: int,
    palette: dict = None,
):
    """Composite a 16x16 sprite onto canvas at (y, x), scaled to cell_px.

    Fully vectorized — no Python pixel loops.
    """
    rgb, alpha = sprite_to_rgb(sprite, palette)

    # Scale sprite to cell size using nearest neighbor (vectorized)
    if cell_px != SPRITE_SIZE:
        idx = np.arange(cell_px)
        src = np.minimum((idx * SPRITE_SIZE // cell_px), SPRITE_SIZE - 1)
        rgb = rgb[np.ix_(src, src)]      # (cell_px, cell_px, 3)
        alpha = alpha[np.ix_(src, src)]  # (cell_px, cell_px)

    h, w = rgb.shape[:2]
    ch, cw = canvas.shape[:2]

    # Clip to canvas bounds
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + h, ch), min(x + w, cw)
    sy0, sx0 = y0 - y, x0 - x
    sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)

    if y1 <= y0 or x1 <= x0:
        return

    # Alpha composite: only overwrite opaque pixels (vectorized)
    region = canvas[y0:y1, x0:x1]
    mask = alpha[sy0:sy1, sx0:sx1] > 0  # (H, W) bool
    mask3 = mask[:, :, np.newaxis]  # broadcast to (H, W, 1)
    canvas[y0:y1, x0:x1] = np.where(mask3, rgb[sy0:sy1, sx0:sx1], region)
