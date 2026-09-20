# Quick Start Guide

## Option 1: Docker (Recommended)

\\\ash
docker-compose up
\\\

Access:
- Frontend: http://localhost:3000
- API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## Option 2: Local Development

### Windows PowerShell

\\\powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
\\\

### Run API (New Terminal)

\\\powershell
.\venv\Scripts\Activate.ps1
cd src\api
flask run
\\\

API: http://localhost:5000

### Run Frontend (New Terminal)

\\\powershell
cd frontend
npm install
npm start
\\\

Frontend: http://localhost:3000

## Making Predictions

### Via API

\\\powershell
\ = @{
    text = "Your article text here"
    title = "Article title"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/predict" 
  -Method POST 
  -ContentType "application/json" 
  -Body \
\\\

### Via Web Interface

1. Open http://localhost:3000
2. Paste article text
3. Click Analyze
4. View results

## Training Models

\\\powershell
python src/models/train_ensemble.py --dataset isot
\\\

## Testing

\\\powershell
pytest tests/ -v
\\\

---

For full setup guide, see docs/SETUP.md
