FROM rust:1-bookworm

# The window packages line from the Phase 3 spike. The build asks for more
# packages; add them here.
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
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libvulkan-dev \
    libfontconfig-dev \
    libssl-dev \
    mesa-vulkan-drivers \
    xvfb \
    fonts-dejavu-core \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

COPY . /src
WORKDIR /src/rust
RUN rustup toolchain install
# whisper.cpp defaults to -march=native, whose ARM FP16 flags GCC 12 rejects
# in this image. The release takes the same baseline.
ENV GGML_NATIVE=OFF
RUN cargo build --release --locked --features gui --bin lazy-take-notes-gui

# Installed after the build, so a test-only package does not invalidate the
# compile layer.
RUN apt-get update && apt-get install -y --no-install-recommends xauth x11-utils \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash", "/src/scripts/gui-smoke/run_spike.sh"]
