FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alienbody ./alienbody
COPY web ./web
COPY data ./data

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT}
