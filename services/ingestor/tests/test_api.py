import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import respx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import JobRecord

API_KEY = "test-key-that-is-at-least-24-characters"
INITIAL_WEB_PASSWORD = "test-web-password-123456789"
USER_WEB_PASSWORD = "memorable-password-123"


def build_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        data_dir=tmp_path,
        ingestor_api_key=API_KEY,
        web_ui_username="admin",
        web_ui_password=INITIAL_WEB_PASSWORD,
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
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        session = client.get("/ui/api/session")
        blocked_until_changed = client.get("/ui/api/jobs")
        password_change = client.post(
            "/ui/api/password",
            json={"new_password": USER_WEB_PASSWORD},
        )
        ready_session = client.get("/ui/api/session")
        client.post("/ui/api/logout")
        old_password = client.post(
            "/ui/api/login",
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        new_password = client.post(
            "/ui/api/login",
            json={"username": "admin", "password": USER_WEB_PASSWORD},
        )

    assert page.status_code == 200
    assert "批量添加视频知识" in page.text
    assert "请先设置新密码" in page.text
    assert "新建知识库" in page.text
    assert denied.status_code == 401
    assert wrong.status_code == 401
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert session.json() == {
        "authenticated": True,
        "must_change_password": True,
        "username": "admin",
    }
    assert blocked_until_changed.status_code == 403
    assert password_change.status_code == 200
    assert password_change.json()["must_change_password"] is False
    assert ready_session.json() == {
        "authenticated": True,
        "must_change_password": False,
        "username": "admin",
    }
    credential_file = tmp_path / "auth" / "web-credentials.json"
    assert credential_file.is_file()
    assert USER_WEB_PASSWORD not in credential_file.read_text(encoding="utf-8")
    assert old_password.status_code == 401
    assert new_password.status_code == 200
    assert new_password.json()["must_change_password"] is False


@respx.mock
def test_web_ui_lists_and_creates_arbitrary_workspaces(tmp_path: Path) -> None:
    base_url = "http://anythingllm.test/api"
    settings = Settings(
        data_dir=tmp_path,
        ingestor_api_key=API_KEY,
        web_ui_username="admin",
        web_ui_password=INITIAL_WEB_PASSWORD,
        web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
        anythingllm_base_url=base_url,
        anythingllm_api_key="anythingllm-test-key",
        anythingllm_auto_sync=False,
    )
    list_route = respx.get(f"{base_url}/v1/workspaces").mock(
        return_value=httpx.Response(
            200,
            json={
                "workspaces": [
                    {"id": 2, "name": "任意知识库", "slug": "custom-library"},
                    {"id": 1, "name": "另一个库", "slug": "another-library"},
                ]
            },
        )
    )
    create_route = respx.post(f"{base_url}/v1/workspace/new").mock(
        return_value=httpx.Response(
            200,
            json={
                "workspace": {"id": 3, "name": "用户自己命名", "slug": "user-created"},
                "message": "Workspace created",
            },
        )
    )

    with TestClient(create_app(settings)) as client:
        client.post(
            "/ui/api/login",
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        client.post("/ui/api/password", json={"new_password": USER_WEB_PASSWORD})
        listed = client.get("/ui/api/workspaces")
        created = client.post("/ui/api/workspaces", json={"name": "用户自己命名"})

    assert listed.status_code == 200
    assert listed.json() == {
        "workspaces": [
            {"id": 2, "name": "任意知识库", "slug": "custom-library"},
            {"id": 1, "name": "另一个库", "slug": "another-library"},
        ]
    }
    assert created.status_code == 200
    assert created.json()["workspace"] == {
        "id": 3,
        "name": "用户自己命名",
        "slug": "user-created",
    }
    assert list_route.calls[0].request.headers["authorization"] == "Bearer anythingllm-test-key"
    assert json.loads(create_route.calls[0].request.content) == {"name": "用户自己命名"}


def test_web_password_can_be_changed_again_and_invalidates_other_sessions(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password=INITIAL_WEB_PASSWORD,
            web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
            anythingllm_auto_sync=False,
        )
    )
    with TestClient(app) as primary, TestClient(app) as secondary:
        primary.post(
            "/ui/api/login",
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        primary.post("/ui/api/password", json={"new_password": USER_WEB_PASSWORD})
        secondary.post(
            "/ui/api/login",
            json={"username": "admin", "password": USER_WEB_PASSWORD},
        )
        wrong_current = primary.post(
            "/ui/api/password",
            json={"current_password": "wrong-password", "new_password": "another-password-456"},
        )
        changed = primary.post(
            "/ui/api/password",
            json={
                "current_password": USER_WEB_PASSWORD,
                "new_password": "another-password-456",
            },
        )
        stale_session = secondary.get("/ui/api/session")

    assert wrong_current.status_code == 400
    assert changed.status_code == 200
    assert stale_session.json()["authenticated"] is False

    with build_client(tmp_path) as restarted:
        initial_after_restart = restarted.post(
            "/ui/api/login",
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        persisted_password = restarted.post(
            "/ui/api/login",
            json={"username": "admin", "password": "another-password-456"},
        )

    assert initial_after_restart.status_code == 401
    assert persisted_password.status_code == 200
    assert persisted_password.json()["must_change_password"] is False


def test_web_ui_batch_deduplicates_input_urls(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password=INITIAL_WEB_PASSWORD,
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
            json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
        )
        client.post("/ui/api/password", json={"new_password": USER_WEB_PASSWORD})
        missing_workspace = client.post(
            "/ui/api/jobs/batch",
            json={"urls": ["https://youtu.be/dQw4w9WgXcQ"]},
        )
        response = client.post(
            "/ui/api/jobs/batch",
            json={
                "urls": [
                    "https://youtu.be/dQw4w9WgXcQ",
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://example.com/not-youtube",
                ],
                "workspace_slug": "custom-library",
            },
        )

    assert missing_workspace.status_code == 422
    assert missing_workspace.json()["detail"] == "请选择目标知识库"
    assert response.status_code == 200
    items = response.json()["items"]
    assert items[0]["accepted"] is True
    assert items[1]["input_duplicate"] is True
    assert items[1]["job_id"] == items[0]["job_id"]
    assert items[0]["workspace_slug"] == "custom-library"
    assert items[2]["accepted"] is False
    assert app.state.manager.submit.await_count == 1
