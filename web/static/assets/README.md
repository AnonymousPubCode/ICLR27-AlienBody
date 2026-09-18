# AlienBody demo assets (v1)

Snake_case paths following Desktop/Game conventions.

```
assets/
  ART_DIRECTION.md
  raw/                 # GPT originals (keep)
  game/
    player/
      agent_n.png
      agent_e.png
      agent_s.png
      agent_w.png
    environment/
      floor_light.png
      floor_dark.png
      map_bg.png
      obstacle.png
      target.png
```

Generated 2026-09-08 via danqing `gpt_4o_image` (~122 credits × 4).
Post: black→alpha, 2×2 agent crop, E/W swapped to match facing, floor tiles picked from 4×4 sheet.

Classic procedural demo: `/static/v0/index.html`
