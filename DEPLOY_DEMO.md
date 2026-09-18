# Deploy AlienBody demo (get a public URL)

## Why not Hugging Face Spaces (free)?

As of 2026, **Docker / Gradio Spaces on free CPU require Hugging Face PRO**.
Creating a Docker Space on the free tier returned HTTP **402 Payment Required**.

If you subscribe to [HF Pro](https://huggingface.co/pro), set `HF_SPACE_ID` (e.g. `YourOrg/AlienBody`) and run:

```bash
python deploy_hf_space.py
```

Then open the Space URL printed by the script.

---

## Recommended free path: Render

Gives a URL like `https://alienbody-demo.onrender.com`.

### Steps (≈5 minutes, once)

1. Sign up at [https://render.com](https://render.com) (GitHub login is fine).
2. **New → Blueprint** → connect repo  
   `AnonymousPubCode/ICLR27-AlienBody`  
   (or push this folder to any GitHub repo you control).
3. Confirm it picks up `render.yaml` + root `Dockerfile`.
4. Create → wait for first Docker build (5–10 min).
5. Open the service URL on the dashboard.

### Notes

- Free tier **spins down** after idle; first load after sleep can take ~30–60s.
- No API keys needed (human play + Oracle/Systematic/…).
- Demo generates episodes on the fly (no need to ship all JSON envs).

### Push Dockerfile updates to the public GitHub first

The GitHub anonymous repo still has the older root `Dockerfile` / README.
After you OK it, we should push:

- `Dockerfile` (Space/Render-oriented, port 7860)
- `requirements-space.txt`
- `render.yaml`

---

## Local preview (no public URL)

```bash
pip install -r requirements-space.txt
python web/app.py
# http://localhost:8000
```
