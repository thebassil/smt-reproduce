FROM python:3.9-slim

# Install solver binaries (for ground-truth reproduction only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Z3 4.15.4
RUN wget -q https://github.com/Z3Prover/z3/releases/download/z3-4.15.4/z3-4.15.4-x64-glibc-2.35.zip \
    -O /tmp/z3.zip && \
    unzip -q /tmp/z3.zip -d /opt && \
    ln -s /opt/z3-4.15.4-x64-glibc-2.35/bin/z3 /usr/local/bin/z3 && \
    rm /tmp/z3.zip

# CVC5 (latest stable)
RUN wget -q https://github.com/cvc5/cvc5/releases/latest/download/cvc5-Linux-x86_64-static.zip \
    -O /tmp/cvc5.zip && \
    unzip -q /tmp/cvc5.zip -d /opt && \
    find /opt -name 'cvc5' -type f -executable -exec ln -s {} /usr/local/bin/cvc5 \; && \
    rm /tmp/cvc5.zip

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[train]"

# Copy pipeline and configs
COPY pipeline/ pipeline/
COPY conf/ conf/
COPY shared_folds_k5.json .

# Copy system and ablation code
COPY systems/ systems/
COPY ablations/ ablations/

# Copy pre-computed results
COPY final_data/ final_data/

# Data is mounted at runtime: -v ./data:/app/data
ENV SMT_DB_PATH=/app/data/results.sqlite
ENV SMT_Z3=/usr/local/bin/z3
ENV SMT_CVC5=/usr/local/bin/cvc5

ENTRYPOINT ["smt"]
CMD ["--help"]
