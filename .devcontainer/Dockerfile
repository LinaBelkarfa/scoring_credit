FROM python:3.11-slim

# Optimisations standards pour la prod (pas de fichiers .pyc, logs en temps réel)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation de Git et de libgomp1 (indispensable pour LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copier et installer le requirements.txt depuis la racine du projet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt