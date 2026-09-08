FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STYLEFORGE_DATA_DIR=/app/data \
    STYLEFORGE_UI_HOST=0.0.0.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app
COPY app_ui.py /app/app_ui.py
COPY scripts /app/scripts

EXPOSE 8000 7860
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
