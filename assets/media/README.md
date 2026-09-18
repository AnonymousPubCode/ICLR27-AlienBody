# Media gallery

Curated clips and stills for the README / project page.
Full generation pipeline lives in the paper workspace (`scripts/make_demos.py`); this folder is a **reviewer-facing subset**.

## Layout

```
media/
  alienbody/   # GIF (+ a few MP4) of AlienBody episodes
  games/       # Kirby / Crafter stills (matched steps)
```

## AlienBody clips

**Compare rows** (`*_compare_oracle_systematic_random.gif`): same env, three agents.
**VLM replays** (`*_qwen3.5-4b.gif`): frames rebuilt from logged trajectories (no re-query).

| File | Notes |
|------|-------|
| `family{1–6}_test_000_compare_*.gif` | Headline side-by-sides |
| `family1_test_004_qwen3.5-4b.gif` | Open-VLM success (F1) |
| `family4_test_000_qwen3.5-4b.gif` | Open-VLM struggle (F4) |

## Games

| File | Condition |
|------|-----------|
| `kirby_named.png` / `kirby_anonymous.png` | step 299 |
| `crafter_named.png` / `crafter_anonymous.png` | step 150 |

## Size policy

GIFs preferred in README (GitHub inline). MP4 kept for `<video>` on `website/`.
Do not dump the full `results/demos/` tree here (~100+ files).
