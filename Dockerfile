FROM python:3.10.12-slim-buster

RUN pip install -U pip --no-cache-dir
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev\
    git\
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# WORKDIR /workspace
# COPY . .
# COPY ../requirements.txt .

# RUN pip install -r requirements.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu