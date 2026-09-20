# Fake News Detector

A production-ready machine learning system for detecting misinformation using ensemble learning.

## Features

- Multi-Model Ensemble (DistilBERT + Gradient Boosting + Random Forest)
- Real-time Predictions via REST API
- Web Interface for article analysis
- Comprehensive test suite
- Docker deployment ready
- Monitoring with Prometheus and Grafana

## Quick Start

### Docker (Simplest)

```bash
docker-compose up
```

Access at:
- Frontend: http://localhost:3000
- API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

### Local Development

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### Run API (New Terminal)

```powershell
.\venv\Scripts\Activate.ps1
cd src\api
flask run
```

API available at: http://localhost:5000

#### Run Frontend (New Terminal)

```powershell
cd frontend
npm install
npm start
```

Frontend available at: http://localhost:3000

## API Endpoints

- POST /api/predict - Single article prediction
- POST /api/batch-predict - Multiple articles
- GET /api/health - Health check
- GET /api/status - Application status
- GET /api/metrics - Performance metrics

## Making Predictions

### Via API (PowerShell)

```powershell
$body = @{
    text = "Your article text here"
    title = "Article title"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Via Web Interface

1. Open http://localhost:3000
2. Paste article text
3. Click Analyze
4. View results and confidence scores

## Training Models

```powershell
python src/models/train_ensemble.py --dataset isot
```

## Testing

```powershell
pip install pytest pytest-cov
pytest tests/ -v --cov=src
```

## Project Structure


## Technology Stack

- Backend: Flask, PyTorch, scikit-learn, transformers
- Database: PostgreSQL, Redis
- ML/NLP: PyTorch, SHAP, spaCy, TextBlob
- DevOps: Docker, Docker Compose, GitHub Actions
- Monitoring: Prometheus, Grafana

## Documentation

- Quick Start: See QUICKSTART.md
- Full Setup: See docs/SETUP.md
- Overview: See PROJECT_SUMMARY.md

## Performance

- Accuracy: 93.2%
- Precision: 91.8%
- Recall: 94.1%
- F1-Score: 92.9%
- AUC-ROC: 0.962

## Getting Started

1. Clone repository
2. Download ISOT Fake News Dataset from Kaggle
3. Extract to data/raw/
4. Run: docker-compose up
5. Access at http://localhost:3000

## License

MIT License

---

Created: September 2024
