# Fake News Detector

A production-ready machine learning system for detecting misinformation using ensemble learning.

## Features
- Multi-Model Ensemble (DistilBERT + Gradient Boosting + Random Forest)
- Real-time Predictions via REST API
- Web Interface for article analysis
- Comprehensive test suite
- Docker deployment ready
- Monitoring with Prometheus & Grafana

## Quick Start

### Docker (Simplest)
\\\ash
docker-compose up
\\\

### Local Development
\\\ash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/models/train_ensemble.py
\\\

## API Endpoints

- POST /api/predict - Single article prediction
- POST /api/batch-predict - Multiple articles
- GET /api/health - Health check
- GET /api/status - Application status

## Documentation

- Quick Start: QUICKSTART.md
- Full Setup: docs/SETUP.md
- Project Overview: PROJECT_SUMMARY.md

## Tech Stack

- Backend: Flask, PyTorch, scikit-learn
- Database: PostgreSQL, Redis
- Monitoring: Prometheus, Grafana
- DevOps: Docker, Docker Compose, GitHub Actions

---
Created: September 2024
