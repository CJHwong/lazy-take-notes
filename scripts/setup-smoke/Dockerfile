FROM debian:bookworm

ARG LTN_TAG=v0.5.1
ARG LTN_ARCH=arm64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libasound2 \
    pulseaudio \
    pulseaudio-utils \
    python3 \
    tar \
    && rm -rf /var/lib/apt/lists/*

# setup.sh installs from a served release, so the tarball path runs against a
# real asset. Pass a dry-run tag and arch with --build-arg to test another one.
RUN mkdir -p /artifacts \
    && curl -fL -o /artifacts/lazy-take-notes-$LTN_ARCH-linux.tar.gz \
        https://github.com/CJHwong/lazy-take-notes/releases/download/$LTN_TAG/lazy-take-notes-$LTN_ARCH-linux.tar.gz \
    && curl -fL -o /artifacts/lazy-take-notes-$LTN_ARCH-linux.tar.gz.sha256 \
        https://github.com/CJHwong/lazy-take-notes/releases/download/$LTN_TAG/lazy-take-notes-$LTN_ARCH-linux.tar.gz.sha256

RUN mkdir -p /models \
    && curl -fL -o /models/ggml-base.en.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

COPY setup.sh /src/setup.sh
COPY scripts/setup-smoke/run_tests.sh /src/run_tests.sh
COPY scripts/linux-smoke/speech.wav /src/speech.wav
COPY scripts/linux-smoke/speech.txt /src/speech.txt
COPY scripts/linux-smoke/words_check.py /src/words_check.py

RUN useradd -m testuser && mkdir -p /srv && chown testuser:testuser /srv

USER testuser
WORKDIR /home/testuser
ENV LTN_TAG=${LTN_TAG}
CMD ["bash", "/src/run_tests.sh"]
