FROM  ghcr.io/astral-sh/uv:python3.14-trixie-slim

COPY . /app

WORKDIR /app

RUN uv sync --locked
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgobject-2.0-0 \
    gcc \
    g++ \
    libgomp1 \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libffi-dev \
    libjpeg-dev \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 7860

CMD ["uv", "run", "app.py"]