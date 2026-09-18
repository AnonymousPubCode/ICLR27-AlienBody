# Deploying AlienBody (web + bench)

This document is a **plan + starter configs** for hosting. Pushing to hosting providers is optional for review; the GitHub anonymous repo already contains code + train/dev/test.

## Recommended stack

| Asset | Recommended host | Why |
|-------|------------------|-----|
| **Interactive web demo** | [Hugging Face Spaces](https://huggingface.co/spaces) (Docker) | Free tier, WebSocket-friendly, academic audience |
| **Benchmark JSON** | Same GitHub repo **or** HF Dataset | GitHub is enough for 720 envs (~few MB); HF helps discovery |
| **Static landing page** | GitHub Pages | Optional marketing page linking to Space + paper |

Avoid GitHub Pages for the game itself: the demo is **FastAPI + WebSocket**, not a static site.

---

## A. Local (already works)

```bash
pip install -r requirements.txt
python web/app.py
# open http://localhost:8000
```

---

## B. Hugging Face Space (Docker) — recommended for public demo

1. Create a Space → SDK **Docker** → public.
2. Copy these files to the Space root (or point the Space at this repo subdirectory later):
   - `Dockerfile` (this folder ships a Space-oriented Dockerfile below in-repo as `web/Dockerfile`)
   - entire `alienbody/`, `web/`, `data/envs/` (or download on start)
3. Hardware: CPU is fine for the algorithmic agents + human play.
4. Space URL becomes your anonymous demo link for the paper footnote.

Minimal `web/Dockerfile` (also mirrored under `web/`):

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alienbody alienbody
COPY web web
COPY data data
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> HF Spaces expect port **7860**. If `web/app.py` hardcodes 8000, launch with uvicorn as above (preferred) instead of `python web/app.py`.

### Secrets on Spaces

Only needed if you later add live VLM play: set `OPENAI_API_KEY` etc. in Space secrets. Default demo (Random / Oracle / Systematic) needs **no keys**.

---

## C. Alternatives for web

| Provider | Notes |
|----------|--------|
| **Render / Railway / Fly.io** | Same Docker image; easy custom domain |
| **Modal / Cloud Run** | Good if you add GPU VLM backends later |
| **Static export** | Not applicable without rewriting to client-only sim |

---

## D. Publishing the bench

**Option 1 — GitHub only (status quo)**  
Reviewers clone this repo; `data/envs/{train,dev,test}` is included. Secret split stays withheld until acceptance / leaderboard policy.

**Option 2 — Hugging Face Dataset** (nice for discovery)

```text
AnonymousPubCode/AlienBody   # dataset card
  train/*.json  or  family{k}/{split}/*.json
```

Load example:

```python
from datasets import load_dataset
# after you upload
# ds = load_dataset("AnonymousPubCode/AlienBody")
```

Keep a `dataset_card.md` with: task definition, splits, license (MIT), paper link (anonymous), and “secret split withheld”.

**Option 3 — Zenodo DOI**  
For long-term archival after acceptance (camera-ready).

---

## E. Paper footnote (suggested)

Once the Space (or repo) is live:

> Code, environments, and an interactive demo: `https://github.com/AnonymousPubCode/ICLR27-AlienBody` (anonymous). Public release upon acceptance.

Add the Space URL when deployed:

> Live demo: `https://huggingface.co/spaces/<anon>/<space>`.

---

## F. Checklist before advertising a public demo

- [ ] No API keys in image / env defaults  
- [ ] No author names / internal paths  
- [ ] Secret split not downloadable  
- [ ] Rate-limit WebSocket if exposing LLM agents  
- [ ] README badges point to working URLs  

Rebuild the anonymous package after edits:

```bash
cd code && python scripts/build_anonymous_release.py
```
