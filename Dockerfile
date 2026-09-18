FROM python:3.11-slim

WORKDIR /app

# Lean install for the interactive demo (no LLM/training stacks).
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY alienbody ./alienbody
COPY web ./web
# Demo generates episodes on the fly; bench JSON optional.
RUN mkdir -p data

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

# HF Spaces expects the app on port 7860.
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
