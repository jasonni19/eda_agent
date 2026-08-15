from fastapi.testclient import TestClient

from eda_agent.api import app


client = TestClient(app)


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_non_csv_before_agent_call():
    response = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert response.json()["detail"] == "Please upload a .csv file."


def test_rejects_empty_csv_before_agent_call():
    response = client.post("/analyze", files={"file": ("empty.csv", b"", "text/csv")})
    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded CSV is empty."
