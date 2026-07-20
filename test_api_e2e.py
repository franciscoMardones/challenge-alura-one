"""Quick E2E test for API endpoints."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

print("Testing /chat with out-of-scope query...")
response = client.post(
    "/chat",
    json={"message": "¿Cuál es el clima hoy?", "session_id": "test"},
)
print(f"Status: {response.status_code}")
data = response.json()
print(f"Answer: {data['answer'][:100]}...")
print(f"Sources: {data['sources']}")
assert response.status_code == 200
assert "fallback" not in data  # ChatResponse model doesn't have fallback field
assert len(data["sources"]) == 0  # Should be empty for fallback
print("✓ Out-of-scope test passed\n")

print("Testing /chat with in-scope query...")
response = client.post(
    "/chat",
    json={"message": "¿Qué es BimBam Buy?", "session_id": "test"},
)
print(f"Status: {response.status_code}")
data = response.json()
print(f"Answer: {data['answer'][:150]}...")
print(f"Sources: {data['sources']}")
assert response.status_code == 200
assert len(data["sources"]) > 0  # Should have sources
print("✓ In-scope test passed\n")

print("All E2E tests passed.")