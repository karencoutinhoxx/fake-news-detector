# Setup and Deployment Guide

## Local Development Setup

### 1. Clone Repository
```bash
git clone https://github.com/karencoutinhoxx/fake-news-detector.git
cd fake-news-detector
```

### 2. Backend Setup
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run API
```powershell
cd src\api
flask run
```

### 4. Run Frontend
```powershell
cd frontend
npm install
npm start
```

## Docker Setup

### Start All Services
```bash
docker-compose up
```

Services available at:
- Frontend: http://localhost:3000
- API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

### Stop Services
```bash
docker-compose down
```

## Training Models

```powershell
python src/models/train_ensemble.py --dataset isot
```

## Testing

```powershell
pytest tests/ -v --cov=src
```

---

For more details, see README.md and QUICKSTART.md
