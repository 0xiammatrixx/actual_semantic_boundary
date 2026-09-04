# Pin the CPU-only PyTorch build and numpy for reproducibility.
# torch 2.2.2 is the last release with official macOS x86_64 wheels, so the
# exact same version is used for local reproduction and in this Linux image.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
        torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy==1.26.4

COPY scaffold/ /app/scaffold/

WORKDIR /app/scaffold

# Single-threaded math for deterministic, reproducible results.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["python", "-m", "boundary_nl2sql.run"]
