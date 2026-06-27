FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    graphviz graphviz-dev gcc g++ pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Ensure pip is available for Python 3.11
RUN python3 -m pip install --upgrade pip || python -m ensurepip --upgrade

# Cloud SQL Auth Proxy (for multi-region DB connectivity)
ADD https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.14.3/cloud-sql-proxy.linux.amd64 /cloud-sql-proxy
RUN chmod +x /cloud-sql-proxy

WORKDIR /app
ENV PYTHONPATH=/app

# Install Python dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY packages/worker/requirements.txt .
RUN grep -v '^md-et' requirements.txt > requirements_no_mdet.txt && \
    pip install --no-cache-dir -r requirements_no_mdet.txt && \
    pip install --no-cache-dir git+https://github.com/mx-e/md-et.git progressbar tenacity psycopg2-binary


# Cache md-et model weights at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from md_et import load_calculator; load_calculator(variant='12l')" || \
    echo "Warning: Could not pre-cache md-et weights (will download at runtime)"

# Copy shared DB package
COPY packages/db /app/packages/db

# Copy kinetics package (snapshot, sampler — used by PES conc gate and kinetic sampling)
COPY packages/kinetics /app/packages/kinetics

# Copy MoreRed (noise schedules, diffusion sampling)
COPY packages/MoreRed_src /app/MoreRed/src

# Copy TS diffusion model weights
COPY packages/worker/models/ /app/models/

# Copy data files (start molecule, fragments)
COPY data/ /app/data/

# Copy worker code
COPY packages/worker/worker.py /app/worker.py
COPY packages/worker/lib/ /app/lib/
COPY packages/worker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
