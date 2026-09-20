# Quick Start Guide

Get the fake news detector up and running in 5 minutes.

## Option 1: Docker (Recommended - Easiest)

```bash
docker-compose up
```

Then access:
- Frontend: http://localhost:3000
- API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

## Option 2: Local Development

### 1. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run API (Terminal 1)

```powershell
.\venv\Scripts\Activate.ps1
cd src\api
flask run
```

API at: http://localhost:5000

### 4. Run Frontend (Terminal 2)

```powershell
cd frontend
npm install
npm start
```

Frontend at: http://localhost:3000

## Making Predictions

### Test API (PowerShell)

```powershell
$body = @{
    text = "This is a test article about politics"
    title = "Test Article"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

### Check Health

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/api/health"
```

## Training Models

```powershell
python src/models/train_ensemble.py --dataset isot
```

## Running Tests

```powershell
pytest tests/ -v
```

## Troubleshooting

### Port Already in Use

Change port in src/api/app.py or use Docker

### Database Connection Error

Make sure PostgreSQL is running (Docker handles this)

### Out of Memory

Reduce batch_size in configs/config.yaml

## Next Steps

1. Download ISOT dataset from Kaggle
2. Place in data/raw/
3. Train models: python src/models/train_ensemble.py
4. Deploy to cloud (see docs/SETUP.md)

---

For full setup: See docs/SETUP.md
