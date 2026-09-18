"""Dual renderer: sprite-based PNG image (for VLMs) and ASCII text (for LLMs).

Image renderer uses pixel art sprites (16x16) for a game-like aesthetic.
Inspired by Crafter/MiniGrid: clean tiles, directional agent, glowing target.
Text renderer produces structured ASCII with coordinate annotations.

Rendering quality is tuned for VLM consumption:
  - 512x512 default (32px/cell for 16x16 grid)
  - Sprite-based agent (astronaut with 4 facing directions)
  - Brick-wall obstacles, star target, textured floor tiles
  - UI overlay: phase indicator, step counter
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

from alienbody.env.grid import (
    EnvConfig, GridState, Position,
    ARC_COLORS, DIRECTION_NAMES, DIRECTION_DELTAS,
)
from alienbody.env.sprites import (
    AGENT_SPRITES, TARGET_SPRITE, OBSTACLE_SPRITE,
    SPRITE_PALETTE, blit_sprite, SPRITE_SIZE,
)

# ── Constants ──────────────────────────────────────────────────────

CELL_PX = 32

# Floor tile colors (enhanced ARC palette for game-like look)
FLOOR_COLORS = {
    0: (228, 228, 220),  # warm off-white floor
    1: (30, 100, 200),   # blue (deeper for tiles)
    2: (220, 55, 45),    # red
    3: (40, 180, 55),    # green
    4: (240, 200, 20),   # yellow
    5: (160, 160, 165),  # grey
    6: (220, 25, 170),   # magenta
    7: (240, 120, 30),   # orange
    8: (110, 200, 240),  # light blue
    9: (120, 20, 35),    # maroon
}

# Slightly lighter variant for checkerboard
FLOOR_COLORS_LIGHT = {
    k: tuple(min(255, c + 12) for c in v) for k, v in FLOOR_COLORS.items()
}

GRID_LINE_COLOR = (195, 195, 190)
UI_BG_COLOR = (35, 35, 45)       # dark header/footer
UI_TEXT_COLOR = (220, 220, 215)
UI_BAR_HEIGHT = 28  # pixels for header/footer bar

# ── Minimal Bitmap Font (5x7, uppercase + digits + punctuation) ──

_FONT_5X7 = {
    'A': [0x70,0x88,0x88,0xF8,0x88,0x88,0x88],
    'B': [0xF0,0x88,0x88,0xF0,0x88,0x88,0xF0],
    'C': [0x70,0x88,0x80,0x80,0x80,0x88,0x70],
    'D': [0xF0,0x88,0x88,0x88,0x88,0x88,0xF0],
    'E': [0xF8,0x80,0x80,0xF0,0x80,0x80,0xF8],
    'F': [0xF8,0x80,0x80,0xF0,0x80,0x80,0x80],
    'G': [0x70,0x88,0x80,0xB8,0x88,0x88,0x70],
    'H': [0x88,0x88,0x88,0xF8,0x88,0x88,0x88],
    'I': [0x70,0x20,0x20,0x20,0x20,0x20,0x70],
    'J': [0x38,0x10,0x10,0x10,0x10,0x90,0x60],
    'K': [0x88,0x90,0xA0,0xC0,0xA0,0x90,0x88],
    'L': [0x80,0x80,0x80,0x80,0x80,0x80,0xF8],
    'M': [0x88,0xD8,0xA8,0x88,0x88,0x88,0x88],
    'N': [0x88,0xC8,0xA8,0x98,0x88,0x88,0x88],
    'O': [0x70,0x88,0x88,0x88,0x88,0x88,0x70],
    'P': [0xF0,0x88,0x88,0xF0,0x80,0x80,0x80],
    'Q': [0x70,0x88,0x88,0x88,0xA8,0x90,0x68],
    'R': [0xF0,0x88,0x88,0xF0,0xA0,0x90,0x88],
    'S': [0x70,0x88,0x80,0x70,0x08,0x88,0x70],
    'T': [0xF8,0x20,0x20,0x20,0x20,0x20,0x20],
    'U': [0x88,0x88,0x88,0x88,0x88,0x88,0x70],
    'V': [0x88,0x88,0x88,0x88,0x50,0x50,0x20],
    'W': [0x88,0x88,0x88,0x88,0xA8,0xD8,0x88],
    'X': [0x88,0x88,0x50,0x20,0x50,0x88,0x88],
    'Y': [0x88,0x88,0x50,0x20,0x20,0x20,0x20],
    'Z': [0xF8,0x08,0x10,0x20,0x40,0x80,0xF8],
    '0': [0x70,0x88,0x98,0xA8,0xC8,0x88,0x70],
    '1': [0x20,0x60,0x20,0x20,0x20,0x20,0x70],
    '2': [0x70,0x88,0x08,0x10,0x20,0x40,0xF8],
    '3': [0xF8,0x10,0x20,0x10,0x08,0x88,0x70],
    '4': [0x10,0x30,0x50,0x90,0xF8,0x10,0x10],
    '5': [0xF8,0x80,0xF0,0x08,0x08,0x88,0x70],
    '6': [0x30,0x40,0x80,0xF0,0x88,0x88,0x70],
    '7': [0xF8,0x08,0x10,0x20,0x40,0x40,0x40],
    '8': [0x70,0x88,0x88,0x70,0x88,0x88,0x70],
    '9': [0x70,0x88,0x88,0x78,0x08,0x10,0x60],
    ' ': [0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    ':': [0x00,0x00,0x20,0x00,0x20,0x00,0x00],
    '/': [0x08,0x08,0x10,0x20,0x40,0x80,0x80],
    '|': [0x20,0x20,0x20,0x20,0x20,0x20,0x20],
    '.': [0x00,0x00,0x00,0x00,0x00,0x00,0x20],
}


def _draw_text_simple(
    img: np.ndarray, text: str,
    x: int, y: int,
    color: tuple = (255, 255, 255),
    scale: int = 1,
):
    """Draw text using minimal 5x7 bitmap font. Pure numpy, no dependencies."""
    cx = x
    for ch in text.upper():
        glyph = _FONT_5X7.get(ch)
        if glyph is None:
            cx += 4 * scale  # unknown char = space
            continue
        for row_i, row_byte in enumerate(glyph):
            for col_i in range(5):
                if row_byte & (0x80 >> col_i):
                    py = y + row_i * scale
                    px = cx + col_i * scale
                    if 0 <= py < img.shape[0] - scale + 1 and 0 <= px < img.shape[1] - scale + 1:
                        img[py:py + scale, px:px + scale] = color
        cx += 6 * scale  # 5px char + 1px gap


# Legacy constants (kept for classic renderer)
RENDER_COLORS = dict(ARC_COLORS)
RENDER_COLORS[0] = (235, 235, 235)
OBSTACLE_COLOR = (60, 60, 60)
OBSTACLE_X_COLOR = (100, 100, 100)
TARGET_BORDER_COLOR = (0, 220, 180)
AGENT_FILL_COLOR = (255, 255, 255)
AGENT_DIR_COLOR = (220, 50, 50)


# ── Sprite-Based Image Renderer ──────────────────────────────────

def render_image(
    state: GridState,
    config: EnvConfig,
    show_target: bool = False,
    cell_px: int = CELL_PX,
    style: str = "sprite",
) -> np.ndarray:
    """Render grid as RGB numpy array (H, W, 3), uint8.

    Args:
        style: "sprite" (game-like pixel art) or "classic" (plain color blocks)
    """
    if style == "classic":
        return _render_classic(state, config, show_target, cell_px)
    return _render_sprite(state, config, show_target, cell_px)


def _render_sprite(
    state: GridState,
    config: EnvConfig,
    show_target: bool,
    cell_px: int,
) -> np.ndarray:
    """Game-like sprite renderer with UI overlay."""
    gs = config.grid_size
    grid_h = gs * cell_px
    grid_w = gs * cell_px
    # Total image includes header + grid + footer
    header_h = UI_BAR_HEIGHT
    footer_h = UI_BAR_HEIGHT
    img_h = header_h + grid_h + footer_h
    img_w = grid_w
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # -- Header bar --
    img[0:header_h, :] = UI_BG_COLOR
    if not show_target:
        phase_text = "PHASE 1: CALIBRATION"
    else:
        # Use shorter text on small grids
        phase_text = "PHASE 2: GO TO TARGET" if grid_w < 400 else "PHASE 2: NAVIGATE TO TARGET"
    _draw_text_simple(img, phase_text, x=6, y=4, color=UI_TEXT_COLOR, scale=2)

    # -- Footer bar --
    img[header_h + grid_h:, :] = UI_BG_COLOR
    step_num = state.phase1_steps + state.phase2_steps
    # Always expose facing direction and internal color state so that
    # perception (reading the observable state) is never the bottleneck;
    # the calibration challenge is *what each action does*, not seeing the state.
    # (n_actions is omitted here — it is already given in the text prompt — to
    #  guarantee the state cues fit within the footer width.)
    dir_name = DIRECTION_NAMES.get(state.agent_dir, "?").upper()
    if grid_w < 400:
        footer_text = f"S{step_num} | F:{dir_name[0]} | C:{state.agent_color}"
    else:
        max_txt = f"/{config.max_total_steps}" if hasattr(config, 'max_total_steps') else ""
        footer_text = f"Step {step_num}{max_txt} | Facing: {dir_name} | Color: {state.agent_color}"
    _draw_text_simple(img, footer_text, x=6, y=header_h + grid_h + 4, color=UI_TEXT_COLOR, scale=2)

    # Offset grid drawing by header height
    grid_region = img[header_h:header_h + grid_h, 0:grid_w]

    obstacles = set()
    if hasattr(config, 'obstacles') and config.obstacles:
        obstacles = {(r, c) for r, c in config.obstacles}

    # 1. Draw floor tiles (checkerboard pattern for depth)
    for r in range(gs):
        for c in range(gs):
            y0, x0 = r * cell_px, c * cell_px
            color_idx = config.cell_colors[r][c]

            if (r, c) in obstacles:
                blit_sprite(grid_region, OBSTACLE_SPRITE, y0, x0, cell_px)
            else:
                is_light = (r + c) % 2 == 0
                palette = FLOOR_COLORS_LIGHT if is_light else FLOOR_COLORS
                base_color = palette.get(color_idx, (200, 200, 200))
                grid_region[y0:y0 + cell_px, x0:x0 + cell_px] = base_color

                if color_idx != 0:
                    border = max(1, cell_px // 16)
                    darker = tuple(max(0, c - 30) for c in base_color)
                    grid_region[y0:y0 + border, x0:x0 + cell_px] = darker
                    grid_region[y0 + cell_px - border:y0 + cell_px, x0:x0 + cell_px] = darker
                    grid_region[y0:y0 + cell_px, x0:x0 + border] = darker
                    grid_region[y0:y0 + cell_px, x0 + cell_px - border:x0 + cell_px] = darker

    # 2. Draw grid lines (subtle, 1px)
    for i in range(gs + 1):
        pos = min(i * cell_px, grid_h - 1)
        grid_region[pos, :] = GRID_LINE_COLOR
        pos = min(i * cell_px, grid_w - 1)
        grid_region[:, pos] = GRID_LINE_COLOR

    # 3. Draw target (star sprite)
    if show_target:
        tr, tc = config.target_pos
        ty, tx = tr * cell_px, tc * cell_px
        blit_sprite(grid_region, TARGET_SPRITE, ty, tx, cell_px)

    # 4. Draw agent (astronaut sprite, direction-aware)
    ar, ac = state.agent_pos.row, state.agent_pos.col
    ay, ax = ar * cell_px, ac * cell_px
    agent_sprite = AGENT_SPRITES[state.agent_dir]

    # If agent has a color (Type D), tint the suit
    if state.agent_color > 0:
        palette = dict(SPRITE_PALETTE)
        suit_rgb = ARC_COLORS.get(state.agent_color, (120, 130, 140))
        palette[5] = suit_rgb  # replace grey suit with agent color
        blit_sprite(grid_region, agent_sprite, ay, ax, cell_px, palette)
    else:
        blit_sprite(grid_region, agent_sprite, ay, ax, cell_px)

    return img


# ── Classic Renderer (legacy, for comparison) ────────────────────

def _render_classic(
    state: GridState,
    config: EnvConfig,
    show_target: bool,
    cell_px: int,
) -> np.ndarray:
    """Original flat-color renderer (kept as fallback)."""
    gs = config.grid_size
    img_size = gs * cell_px
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    obstacles = set()
    if hasattr(config, 'obstacles') and config.obstacles:
        obstacles = {(r, c) for r, c in config.obstacles}

    for r in range(gs):
        for c in range(gs):
            r0, r1 = r * cell_px, (r + 1) * cell_px
            c0, c1 = c * cell_px, (c + 1) * cell_px
            if (r, c) in obstacles:
                img[r0:r1, c0:c1] = OBSTACLE_COLOR
                _draw_x_pattern(img, r0, c0, cell_px, OBSTACLE_X_COLOR)
            else:
                color_idx = config.cell_colors[r][c]
                color = RENDER_COLORS.get(color_idx, (200, 200, 200))
                img[r0:r1, c0:c1] = color

    for i in range(gs + 1):
        pos = i * cell_px
        if pos < img_size:
            img[pos, :] = (180, 180, 180)
            img[:, pos] = (180, 180, 180)

    if show_target:
        tr, tc = config.target_pos
        _draw_target_classic(img, tr, tc, cell_px)

    ar, ac = state.agent_pos.row, state.agent_pos.col
    _draw_agent_classic(img, ar, ac, cell_px, state.agent_dir, state.agent_color)

    return img


def _draw_x_pattern(img: np.ndarray, r0: int, c0: int, size: int, color: tuple):
    """Draw an X pattern inside a cell (for obstacles)."""
    for i in range(size):
        # Main diagonal
        if r0 + i < img.shape[0] and c0 + i < img.shape[1]:
            img[r0 + i, c0 + i] = color
        # Anti-diagonal
        if r0 + i < img.shape[0] and c0 + size - 1 - i < img.shape[1]:
            img[r0 + i, c0 + size - 1 - i] = color


def _draw_target_classic(img: np.ndarray, row: int, col: int, cell_px: int):
    """Draw target as thick bright border around cell."""
    r0, r1 = row * cell_px, (row + 1) * cell_px
    c0, c1 = col * cell_px, (col + 1) * cell_px
    bw = max(2, cell_px // 10)  # border width scales with cell size

    for b in range(bw):
        rr0, rr1 = r0 + b, r1 - b
        cc0, cc1 = c0 + b, c1 - b
        if rr0 < rr1 and cc0 < cc1:
            img[rr0, cc0:cc1] = TARGET_BORDER_COLOR
            img[rr1 - 1, cc0:cc1] = TARGET_BORDER_COLOR
            img[rr0:rr1, cc0] = TARGET_BORDER_COLOR
            img[rr0:rr1, cc1 - 1] = TARGET_BORDER_COLOR


def _draw_agent_classic(img: np.ndarray, row: int, col: int, cell_px: int,
                direction: int, agent_color: int):
    """Draw agent as white circle with red directional triangle."""
    r0, r1 = row * cell_px, (row + 1) * cell_px
    c0, c1 = col * cell_px, (col + 1) * cell_px
    margin = max(3, cell_px // 6)

    # Fill agent cell with white (always visible)
    img[r0 + margin:r1 - margin, c0 + margin:c1 - margin] = AGENT_FILL_COLOR

    # Draw directional triangle (red) taking ~40% of cell on the facing edge
    cx = (c0 + c1) // 2
    cy = (r0 + r1) // 2
    tri_size = max(3, cell_px // 4)

    if direction == 0:      # up
        _fill_triangle(img, (r0 + margin, cx), tri_size, "up", AGENT_DIR_COLOR)
    elif direction == 1:    # right
        _fill_triangle(img, (cy, c1 - margin - 1), tri_size, "right", AGENT_DIR_COLOR)
    elif direction == 2:    # down
        _fill_triangle(img, (r1 - margin - 1, cx), tri_size, "down", AGENT_DIR_COLOR)
    elif direction == 3:    # left
        _fill_triangle(img, (cy, c0 + margin), tri_size, "left", AGENT_DIR_COLOR)

    # Inner color dot (shows agent_color for Type D)
    if agent_color > 0:
        inner_color = ARC_COLORS.get(agent_color, (200, 200, 200))
        dot_r = max(2, cell_px // 8)
        img[cy - dot_r:cy + dot_r, cx - dot_r:cx + dot_r] = inner_color


def _fill_triangle(img: np.ndarray, tip: tuple, size: int, direction: str, color: tuple):
    """Draw a filled triangle pointing in the given direction."""
    ty, tx = tip
    h, w = img.shape[:2]

    for i in range(size):
        # Width of triangle at this row expands linearly
        half_w = i
        if direction == "up":
            row = ty + i
            for dc in range(-half_w, half_w + 1):
                c = tx + dc
                if 0 <= row < h and 0 <= c < w:
                    img[row, c] = color
        elif direction == "down":
            row = ty - i
            for dc in range(-half_w, half_w + 1):
                c = tx + dc
                if 0 <= row < h and 0 <= c < w:
                    img[row, c] = color
        elif direction == "right":
            col = tx - i
            for dr in range(-half_w, half_w + 1):
                r = ty + dr
                if 0 <= r < h and 0 <= col < w:
                    img[r, col] = color
        elif direction == "left":
            col = tx + i
            for dr in range(-half_w, half_w + 1):
                r = ty + dr
                if 0 <= r < h and 0 <= col < w:
                    img[r, col] = color


def image_to_png_bytes(img: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes (for saving or sending to VLM API)."""
    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        buf = io.BytesIO()
        np.save(buf, img)
        return buf.getvalue()


def save_image(img: np.ndarray, path: str, scale: int = 1):
    """Save grid image to file, optionally scaled up."""
    from PIL import Image
    pil_img = Image.fromarray(img)
    if scale > 1:
        new_size = (img.shape[1] * scale, img.shape[0] * scale)
        pil_img = pil_img.resize(new_size, Image.NEAREST)
    pil_img.save(path)


def render_episode_gif(
    config: 'EnvConfig',
    actions: list[int],
    path: str,
    cell_px: int = CELL_PX,
    scale: int = 2,
    frame_duration: int = 300,
    phase1_budget: Optional[int] = None,
    style: str = "sprite",
):
    """Render a full episode as animated GIF.

    Args:
        config: Environment config.
        actions: List of action indices (the full episode trajectory).
        path: Output file path (.gif).
        scale: Upscale factor (2 = double resolution for crisp pixels).
        frame_duration: Milliseconds per frame.
        phase1_budget: Override phase 1 budget (default: config.phase1_budget).
        style: "sprite" or "classic".
    """
    from PIL import Image as PILImage
    from alienbody.env.core import AlienBodyEnv

    if phase1_budget is None:
        phase1_budget = config.phase1_budget

    env = AlienBodyEnv(config, render_mode="image")
    obs, info = env.reset()

    frames = []

    # Capture initial frame
    img = render_image(env.state, config, show_target=False, cell_px=cell_px, style=style)
    frames.append(img)

    for step_i, action in enumerate(actions):
        # Check for phase transition
        show_target = (info.get("phase", 1) == 2)

        obs, reward, terminated, truncated, info = env.step(action)
        show_target = (info.get("phase", 1) == 2)
        img = render_image(env.state, config, show_target=show_target, cell_px=cell_px, style=style)
        frames.append(img)

        if terminated or truncated:
            break

    # Convert to PIL and save
    pil_frames = []
    for frame in frames:
        pil = PILImage.fromarray(frame)
        if scale > 1:
            pil = pil.resize((frame.shape[1] * scale, frame.shape[0] * scale), PILImage.NEAREST)
        pil_frames.append(pil)

    # Last frame stays longer
    durations = [frame_duration] * len(pil_frames)
    if len(durations) > 1:
        durations[-1] = frame_duration * 3  # hold final frame

    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    return len(frames)


# ── Text Renderer ──────────────────────────────────────────────────

COLOR_SYMBOLS = {
    0: ".",   # background → dot
    1: "B",   # blue
    2: "R",   # red
    3: "G",   # green
    4: "Y",   # yellow
    5: "#",   # grey
    6: "M",   # magenta
    7: "O",   # orange
    8: "b",   # light blue
    9: "m",   # maroon
}

# Human-readable names for each color value
COLOR_NAMES = {
    0: "empty", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "lt_blue", 9: "maroon",
}

# Brightness ordering: higher color index = brighter
# (used by "move_toward_brightest" which picks the 4-neighbor with max color value)
BRIGHTEST_ORDER = sorted(COLOR_SYMBOLS.keys(), key=lambda k: k)  # 0 < 1 < ... < 9


def _build_color_legend(config) -> str:
    """Build a compact color legend for text-mode observations.

    Shows symbol-to-value mapping and brightness ordering, so LLMs can
    interpret relational actions (move_toward_brightest, flee_same_color, etc.)
    without relying on visual color perception.
    """
    # Only show colors that actually appear in this grid
    used_colors = set()
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            used_colors.add(config.cell_colors[r][c])

    # Symbol map: "[.]bg(0) [B]blue(1) ..."
    symbol_parts = []
    for val in sorted(used_colors):
        sym = COLOR_SYMBOLS.get(val, "?")
        name = COLOR_NAMES.get(val, f"c{val}")
        symbol_parts.append("[{}]={}({})".format(sym, name, val))
    symbol_map = " ".join(symbol_parts)

    # Brightness order (sorted by value, higher = brighter)
    used_sorted = sorted(used_colors)
    bright_parts = []
    for val in used_sorted:
        sym = COLOR_SYMBOLS.get(val, "?")
        bright_parts.append("{}({})".format(sym, val))
    brightness = " < ".join(bright_parts)
    if brightness:
        brightness += "  (higher value = brighter)"

    legend = "Color key: {}\nBrightness: {}".format(symbol_map, brightness)
    return legend

AGENT_ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}


def render_text(
    state: GridState,
    config: EnvConfig,
    show_target: bool = False,
) -> str:
    """Render grid as structured text for LLMs.

    Includes: header (grid info, agent state, target), coordinate grid,
    color legend, and action listing.
    """
    gs = config.grid_size
    obstacles = set()
    if hasattr(config, 'obstacles') and config.obstacles:
        obstacles = {(r, c) for r, c in config.obstacles}

    lines = []

    # Header
    lines.append(f"Grid: {gs}x{gs}")
    lines.append(f"Agent: ({state.agent_pos.row},{state.agent_pos.col})"
                 f" facing {DIRECTION_NAMES[state.agent_dir]}"
                 f" color={state.agent_color}")
    if show_target:
        lines.append(f"Target: ({config.target_pos[0]},{config.target_pos[1]})")
    lines.append(f"Actions available: {', '.join(f'Action {i}' for i in range(config.n_actions))}")
    lines.append("")

    # Column headers
    col_header = "    " + " ".join(f"{c:2d}" for c in range(gs))
    lines.append(col_header)
    lines.append("    " + "---" * gs)

    # Grid rows
    for r in range(gs):
        row_chars = []
        for c in range(gs):
            if r == state.agent_pos.row and c == state.agent_pos.col:
                row_chars.append(f" {AGENT_ARROWS[state.agent_dir]}")
            elif show_target and r == config.target_pos[0] and c == config.target_pos[1]:
                row_chars.append(" T")
            elif (r, c) in obstacles:
                row_chars.append(" X")
            else:
                sym = COLOR_SYMBOLS.get(config.cell_colors[r][c], "?")
                row_chars.append(f" {sym}")
        lines.append(f"{r:2d} |{''.join(row_chars)}")

    # Color legend
    lines.append("")
    lines.append(_build_color_legend(config))

    return "\n".join(lines)


def render_feedback_text(record: dict) -> str:
    """Compact single-line feedback for LLM context windows."""
    parts = [f"Step {record['step']}: Action {record['action']}"]
    if record["position_changed"]:
        parts.append(f"pos ({record['prev_pos'][0]},{record['prev_pos'][1]})"
                     f"→({record['new_pos'][0]},{record['new_pos'][1]})")
    else:
        parts.append("pos unchanged")
    if record["direction_changed"]:
        parts.append(f"dir→{DIRECTION_NAMES[record['new_dir']]}")
    if record["color_changed"]:
        parts.append(f"color {record['prev_color']}→{record['new_color']}")
    return " | ".join(parts)
