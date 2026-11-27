FROM rust:1-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    pulseaudio \
    pulseaudio-utils \
    libasound2-dev \
    cmake \
    clang \
    libclang-dev \
    ffmpeg \
    curl \
    python3 \
    && rm -rf /var/lib/apt/lists/*

COPY . /src
WORKDIR /src/rust
RUN rustup toolchain install
# whisper.cpp defaults to -march=native, whose ARM FP16 flags GCC 12 rejects
# in this image. The release takes the same baseline.
ENV GGML_NATIVE=OFF
RUN cargo build --release --locked
RUN install -m755 /src/rust/target/release/lazy-take-notes /usr/local/bin/

RUN mkdir -p /models \
    && curl -fL -o /models/ggml-base.en.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

RUN useradd -m testuser
COPY scripts/linux-smoke/run_smoke.sh /usr/local/bin/
USER testuser
CMD ["/usr/local/bin/run_smoke.sh"]
