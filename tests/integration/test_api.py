import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestAPIHealthCheck:
    def test_health_check(self, client):
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'

class TestAPIPrediction:
    def test_prediction_missing_text(self, client):
        response = client.post('/api/predict',
                             data=json.dumps({}),
                             content_type='application/json')
        assert response.status_code == 400

    def test_prediction_valid_input(self, client):
        payload = {
            'text': 'This is a test article',
            'title': 'Test Article'
        }
        response = client.post('/api/predict',
                             data=json.dumps(payload),
                             content_type='application/json')
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
