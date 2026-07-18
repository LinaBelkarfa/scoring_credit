FROM python:3.11-slim

# Optimisations standards pour la prod (pas de fichiers .pyc, logs en temps réel)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation de Git à l'intérieur du conteneur
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace