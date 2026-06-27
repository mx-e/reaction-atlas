FROM --platform=linux/amd64 python:3.11-slim

# CPU worker — handles CREST conformer search, PBE0 DFT barrier refinement,
# and ML energy evaluation of CREST conformers.

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl xz-utils git \
    libgomp1 libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# xTB 6.7.1 (pre-built static binary) — used by CREST jobs
RUN curl -fSL -o /tmp/xtb.tar.xz \
        https://github.com/grimme-lab/xtb/releases/download/v6.7.1/xtb-6.7.1-linux-x86_64.tar.xz && \
    tar xf /tmp/xtb.tar.xz -C /opt && \
    ln -s /opt/xtb-6.7.1/bin/xtb /usr/local/bin/xtb && \
    rm /tmp/xtb.tar.xz

# CREST 3.0.2 (pre-built static binary)
RUN curl -fSL -o /tmp/crest.tar.xz \
        https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz && \
    mkdir -p /opt/crest && \
    tar xf /tmp/crest.tar.xz -C /opt/crest && \
    find /opt/crest -name "crest" -type f -exec ln -sf {} /usr/local/bin/crest \; && \
    rm /tmp/crest.tar.xz

# Verify CREST stack
RUN xtb --version 2>&1 | head -1 && crest --version 2>&1 | head -3

# Python deps:
#  - sqlalchemy/psycopg2-binary/loguru/numpy: shared (CREST + DFT)
#  - pyscf: PBE0 DFT for the DFT job kind
#  - ase:   geometry helpers used by dft_runner
RUN pip install --no-cache-dir \
        sqlalchemy psycopg2-binary loguru numpy scipy \
        pyscf ase && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir git+https://github.com/mx-e/md-et.git

# Verify imports
RUN python3 -c "from pyscf import gto, dft; print('pyscf OK')" && \
    python3 -c "import torch; print(f'torch {torch.__version__} OK (CPU)')"

# Cache md-et model weights at build time
RUN python3 -c "from md_et import load_calculator; load_calculator(variant='12l')" || \
    echo "Warning: Could not pre-cache md-et weights (will download at runtime)"

WORKDIR /app
ENV PYTHONPATH=/app
ENV XTBPATH=/opt/xtb-6.7.1/share/xtb

COPY packages/db /app/packages/db
# Cloud SQL Auth Proxy (for multi-region DB connectivity)
ADD https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.14.3/cloud-sql-proxy.linux.amd64 /cloud-sql-proxy
RUN chmod +x /cloud-sql-proxy

COPY packages/cpu-worker/worker.py /app/worker.py
COPY packages/cpu-worker/dft_runner.py /app/dft_runner.py
COPY packages/cpu-worker/rmsd_match.py /app/rmsd_match.py

COPY packages/cpu-worker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
