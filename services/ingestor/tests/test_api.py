from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import JobRecord

API_KEY = "test-key-that-is-at-least-24-characters"


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        ingestor_api_key=API_KEY,
        web_ui_username="admin",
        web_ui_password="test-web-password-123456789",
        web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
        anythingllm_auto_sync=False,
    )
    return TestClient(create_app(settings))


def test_health_does_not_require_key(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_jobs_require_api_key(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.get("/jobs")
        authenticated = client.get("/jobs", headers={"X-API-Key": API_KEY})
    assert response.status_code == 401
    assert authenticated.status_code == 200


def test_rejects_non_youtube_before_starting_job(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        response = client.post(
            "/jobs",
            headers={"X-API-Key": API_KEY},
            json={"url": "https://example.com/video"},
        )
    assert response.status_code == 422


def test_web_ui_requires_login_and_uses_secure_session_cookie(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        page = client.get("/ui")
        denied = client.post(
            "/ui/api/jobs/batch",
            json={"urls": ["https://youtu.be/dQw4w9WgXcQ"]},
        )
        wrong = client.post(
            "/ui/api/login",
            json={"username": "admin", "password": "wrong-password"},
        )
        login = client.post(
            "/ui/api/login",
            json={"username": "admin", "password": "test-web-password-123456789"},
        )
        session = client.get("/ui/api/session")

    assert page.status_code == 200
    assert "批量添加视频知识" in page.text
    assert denied.status_code == 401
    assert wrong.status_code == 401
    assert login.status_code == 200
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert session.json() == {"authenticated": True, "username": "admin"}


def test_web_ui_batch_deduplicates_input_urls(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password="test-web-password-123456789",
            web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
            anythingllm_auto_sync=False,
        )
    )
    job = JobRecord(
        id="a" * 32,
        url="https://youtu.be/dQw4w9WgXcQ",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    app.state.manager.submit = AsyncMock(return_value=(job, False))

    with TestClient(app) as client:
        client.post(
            "/ui/api/login",
            json={"username": "admin", "password": "test-web-password-123456789"},
        )
        response = client.post(
            "/ui/api/jobs/batch",
            json={
                "urls": [
                    "https://youtu.be/dQw4w9WgXcQ",
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://example.com/not-youtube",
                ]
            },
        )

    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["accepted"] is True
    assert items[1]["input_duplicate"] is True
    assert items[1]["job_id"] == items[0]["job_id"]
    assert items[2]["accepted"] is False
    assert app.state.manager.submit.await_count == 1
