FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    pulseaudio \
    libpulse-dev \
    libportaudio2 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# PulseAudio refuses to run as root — create a test user
RUN useradd -m -s /bin/bash testuser

WORKDIR /home/testuser/app
COPY . .
RUN rm -rf .venv && chown -R testuser:testuser /home/testuser/app

USER testuser

# Pre-warm uv cache + install all deps
RUN uv sync

CMD ["bash", "-c", "\
    echo '=== Starting PulseAudio ===' && \
    pulseaudio --start --exit-idle-time=-1 && \
    pactl load-module module-null-sink sink_name=test_loopback && \
    echo '=== PulseAudio ready (null-sink loaded) ===' && \
    echo '' && \
    echo '=== Running pytest ===' && \
    uv run pytest tests/ -v --tb=short && \
    echo '' && \
    echo '=== Running loopback smoke test ===' && \
    uv run python scripts/linux-smoke/smoke_test.py \
"]
