# AlienBody Demo — Art Direction (v1)

Top-down **grid calibration game** for the AlienBody paper demo.
Not cyber-HUD chrome; readable sprites over a calm floor.

## Style lock
- Top-down / slight 3/4, soft pixel-inspired but clean (not noisy 8-bit)
- Palette: cool stone floors, cyan agent accent, teal goal, warm-dark obstacles
- Transparent PNG sprites; square canvas; centered subject; no baked ground under characters
- Readable at ~48–96px on screen (source 256×256 or 512×512)

## Asset map (program layers)

| id | path | layer | use |
|----|------|-------|-----|
| `floor_light` | `environment/floor_light.png` | tile | even cells |
| `floor_dark` | `environment/floor_dark.png` | tile | odd cells |
| `obstacle` | `environment/obstacle.png` | sprite | blocked cells |
| `target` | `environment/target.png` | sprite | Phase-2 goal |
| `agent_n` | `player/agent_n.png` | sprite | facing north |
| `agent_e` | `player/agent_e.png` | sprite | facing east |
| `agent_s` | `player/agent_s.png` | sprite | facing south |
| `agent_w` | `player/agent_w.png` | sprite | facing west |

## Interaction upgrades (code)
- Sprite-based agent with facing from `agent.dir`
- Soft move lerp (keep)
- Target pulse using sprite + glow
- Phase-1 vs Phase-2 HUD tint already exists — keep
- Action buttons: keep anonymous labels; add press feedback
- Link to original: `/static/v0/index.html` optional note in footer

## Non-goals (v1)
- Full walk-cycle animation sheets
- Named-vs-anonymous dual view
- LLM live play
