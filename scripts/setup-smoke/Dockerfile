FROM debian:bookworm

# Base system deps for Homebrew + building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    file \
    git \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Homebrew requires a non-root user, but needs /home/linuxbrew owned correctly
RUN useradd -m -s /bin/bash testuser \
    && mkdir -p /home/linuxbrew/.linuxbrew \
    && chown -R testuser:testuser /home/linuxbrew

USER testuser
WORKDIR /home/testuser

# Install Homebrew (linuxbrew)
RUN NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:$PATH"

# Pre-install uv via brew (saves time in the actual setup.sh run)
RUN brew install uv

# Copy just the setup script and test runner
COPY --chown=testuser:testuser setup.sh /home/testuser/setup.sh
COPY --chown=testuser:testuser scripts/setup-smoke/run_tests.sh /home/testuser/run_tests.sh

CMD ["bash", "/home/testuser/run_tests.sh"]
