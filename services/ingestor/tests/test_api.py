import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import respx
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import JobRecord, JobStatus, SyncStatus

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
                "category_path": "投资/虚拟币/BTC",
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
    assert items[0]["category_path"] == "投资/虚拟币/BTC"
    assert items[2]["accepted"] is False
    assert app.state.manager.submit.await_count == 1
    submitted_request = app.state.manager.submit.await_args.args[0]
    assert submitted_request.category_path == "投资/虚拟币/BTC"


def ready_web_client(app: object) -> TestClient:
    client = TestClient(app)
    client.__enter__()
    client.post(
        "/ui/api/login",
        json={"username": "admin", "password": INITIAL_WEB_PASSWORD},
    )
    client.post("/ui/api/password", json={"new_password": USER_WEB_PASSWORD})
    return client


@respx.mock
def test_directory_delete_is_recursive_and_requires_exact_confirmation(tmp_path: Path) -> None:
    base_url = "http://anythingllm.test/api"
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password=INITIAL_WEB_PASSWORD,
            web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
            anythingllm_base_url=base_url,
            anythingllm_api_key="anythingllm-test-key",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    jobs = [
        JobRecord(
            id="crypto-job",
            url="https://youtu.be/aaaaaaaaaaa",
            canonical_url="https://www.youtube.com/watch?v=aaaaaaaaaaa",
            workspace_slug="research",
            category_path="投资/虚拟币",
            status=JobStatus.completed,
            stage="completed",
            sync_status=SyncStatus.synced,
            anythingllm_document_location="custom-documents/crypto.json",
        ),
        JobRecord(
            id="stocks-job",
            url="https://youtu.be/bbbbbbbbbbb",
            canonical_url="https://www.youtube.com/watch?v=bbbbbbbbbbb",
            workspace_slug="research",
            category_path="投资/股票",
            status=JobStatus.failed,
            stage="failed",
            sync_status=SyncStatus.failed,
        ),
        JobRecord(
            id="python-job",
            url="https://youtu.be/ccccccccccc",
            canonical_url="https://www.youtube.com/watch?v=ccccccccccc",
            workspace_slug="research",
            category_path="软件开发/Python",
            status=JobStatus.completed,
            stage="completed",
        ),
    ]
    for job in jobs:
        manager.jobs[job.id] = job
        manager.storage.save(job)
        manager.catalog.register(job.workspace_slug, job.category_path)

    remove_embeddings = respx.post(
        f"{base_url}/v1/workspace/research/update-embeddings"
    ).mock(return_value=httpx.Response(200, json={"workspace": {"slug": "research"}}))
    purge_documents = respx.delete(f"{base_url}/v1/system/remove-documents").mock(
        return_value=httpx.Response(200, json={"success": True})
    )

    client = ready_web_client(app)
    try:
        listed = client.get("/ui/api/directories?workspace_slug=research")
        preview = client.post(
            "/ui/api/directories/delete-preview",
            json={"workspace_slug": "research", "path": "投资"},
        )
        rejected = client.request(
            "DELETE",
            "/ui/api/directories",
            json={
                "workspace_slug": "research",
                "path": "投资",
                "confirm_path": "软件开发",
            },
        )
        deleted = client.request(
            "DELETE",
            "/ui/api/directories",
            json={
                "workspace_slug": "research",
                "path": "投资",
                "confirm_path": "投资",
            },
        )
    finally:
        client.__exit__(None, None, None)

    assert listed.status_code == 200
    assert preview.json() == {
        "workspace_slug": "research",
        "path": "投资",
        "directories": 3,
        "jobs": 2,
        "synced_documents": 1,
        "active_jobs": 0,
    }
    assert rejected.status_code == 400
    assert deleted.status_code == 200
    assert deleted.json()["deleted_jobs"] == 2
    assert deleted.json()["deleted_directories"] == 3
    assert set(manager.jobs) == {"python-job"}
    assert manager.directory_paths("research") == ["软件开发", "软件开发/Python"]
    assert not (tmp_path / "jobs" / "crypto-job").exists()
    assert not (tmp_path / "jobs" / "stocks-job").exists()
    assert (tmp_path / "jobs" / "python-job").exists()
    assert remove_embeddings.call_count == 1
    assert purge_documents.call_count == 1
    assert json.loads(remove_embeddings.calls[0].request.content) == {
        "adds": [],
        "deletes": ["custom-documents/crypto.json"],
    }
    assert json.loads(purge_documents.calls[0].request.content) == {
        "names": ["custom-documents/crypto.json"]
    }


@respx.mock
def test_single_knowledge_delete_keeps_local_record_when_remote_purge_fails(
    tmp_path: Path,
) -> None:
    base_url = "http://anythingllm.test/api"
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password=INITIAL_WEB_PASSWORD,
            web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
            anythingllm_base_url=base_url,
            anythingllm_api_key="anythingllm-test-key",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    job = JobRecord(
        id="knowledge-job",
        url="https://youtu.be/ddddddddddd",
        canonical_url="https://www.youtube.com/watch?v=ddddddddddd",
        workspace_slug="research",
        category_path="投资/宏观",
        status=JobStatus.completed,
        stage="completed",
        anythingllm_document_location="custom-documents/macro.json",
    )
    manager.jobs[job.id] = job
    manager.storage.save(job)
    manager.catalog.register(job.workspace_slug, job.category_path)
    respx.post(f"{base_url}/v1/workspace/research/update-embeddings").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.delete(f"{base_url}/v1/system/remove-documents").mock(
        return_value=httpx.Response(500)
    )

    client = ready_web_client(app)
    try:
        mismatch = client.request(
            "DELETE",
            "/ui/api/jobs/knowledge-job",
            json={"confirm_job_id": "another-job"},
        )
        failed = client.request(
            "DELETE",
            "/ui/api/jobs/knowledge-job",
            json={"confirm_job_id": "knowledge-job"},
        )
    finally:
        client.__exit__(None, None, None)

    assert mismatch.status_code == 400
    assert failed.status_code == 409
    assert "AnythingLLM 删除知识失败" in failed.json()["detail"]
    assert manager.get("knowledge-job") is job
    assert (tmp_path / "jobs" / "knowledge-job" / "job.json").is_file()
