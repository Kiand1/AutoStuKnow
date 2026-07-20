import io
import json
import zipfile
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


def test_library_preview_and_downloads_require_web_login(tmp_path: Path) -> None:
    with build_client(tmp_path) as client:
        responses = [
            client.get("/ui/api/library?workspace_slug=research"),
            client.get("/ui/api/jobs/missing/content"),
            client.get("/ui/api/jobs/missing/download"),
            client.get("/ui/api/workspaces/research/download"),
            client.get(
                "/ui/api/directories/download",
                params={"workspace_slug": "research", "path": "投资"},
            ),
        ]
    assert [response.status_code for response in responses] == [401, 401, 401, 401, 401]


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


@respx.mock
def test_workspace_delete_requires_name_and_cleans_only_managed_source_documents(
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
            anythingllm_workspace_slug="research",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    jobs = [
        JobRecord(
            id="managed-knowledge",
            url="https://youtu.be/eeeeeeeeeee",
            canonical_url="https://www.youtube.com/watch?v=eeeeeeeeeee",
            workspace_slug="research",
            category_path="投资/虚拟币",
            status=JobStatus.completed,
            stage="completed",
            anythingllm_document_location="custom-documents/managed.json",
        ),
        JobRecord(
            id="failed-knowledge",
            url="https://youtu.be/fffffffffff",
            canonical_url="https://www.youtube.com/watch?v=fffffffffff",
            workspace_slug=None,
            category_path="",
            status=JobStatus.failed,
            stage="failed",
        ),
        JobRecord(
            id="other-workspace-knowledge",
            url="https://youtu.be/ggggggggggg",
            canonical_url="https://www.youtube.com/watch?v=ggggggggggg",
            workspace_slug="development",
            category_path="Python",
            status=JobStatus.completed,
            stage="completed",
        ),
    ]
    for job in jobs:
        manager.jobs[job.id] = job
        manager.storage.save(job)
        manager.catalog.register(job.workspace_slug, job.category_path)

    workspace_details = respx.get(f"{base_url}/v1/workspace/research").mock(
        return_value=httpx.Response(
            200,
            json={
                "workspace": [
                    {
                        "id": 7,
                        "name": "投资研究",
                        "slug": "research",
                        "documents": [
                            {"docpath": "custom-documents/managed.json"},
                            {"docpath": "custom-documents/manual.pdf.json"},
                            {"docpath": "custom-documents/shared.pdf.json"},
                        ],
                        "threads": [{"slug": "default"}],
                    }
                ]
            },
        )
    )
    remove_embeddings = respx.post(
        f"{base_url}/v1/workspace/research/update-embeddings"
    ).mock(return_value=httpx.Response(200, json={}))
    purge_documents = respx.delete(f"{base_url}/v1/system/remove-documents").mock(
        return_value=httpx.Response(200, json={"success": True})
    )
    delete_workspace = respx.delete(f"{base_url}/v1/workspace/research").mock(
        return_value=httpx.Response(200)
    )

    client = ready_web_client(app)
    try:
        preview = client.post(
            "/ui/api/workspaces/delete-preview",
            json={"workspace_slug": "research"},
        )
        rejected = client.request(
            "DELETE",
            "/ui/api/workspaces/research",
            json={"confirm_name": "另一个知识库"},
        )
        deleted = client.request(
            "DELETE",
            "/ui/api/workspaces/research",
            json={"confirm_name": "投资研究"},
        )
    finally:
        client.__exit__(None, None, None)

    assert preview.status_code == 200
    assert preview.json() == {
        "workspace": {
            "id": 7,
            "name": "投资研究",
            "slug": "research",
            "document_count": 3,
            "thread_count": 1,
        },
        "directories": 2,
        "managed_jobs": 2,
        "managed_documents": 1,
        "active_jobs": 0,
    }
    assert rejected.status_code == 409
    assert rejected.json()["detail"] == "确认名称与待删除知识库不一致"
    assert deleted.status_code == 200
    assert deleted.json() == {
        "workspace_slug": "research",
        "deleted": True,
        "workspace_name": "投资研究",
        "deleted_jobs": 2,
        "deleted_documents": 1,
        "deleted_directories": 2,
    }
    assert workspace_details.call_count == 3
    assert delete_workspace.call_count == 1
    assert set(manager.jobs) == {"other-workspace-knowledge"}
    assert manager.directory_paths("research") == []
    assert manager.directory_paths("development") == ["Python"]
    assert json.loads(remove_embeddings.calls[0].request.content) == {
        "adds": [],
        "deletes": ["custom-documents/managed.json"],
    }
    assert json.loads(purge_documents.calls[0].request.content) == {
        "names": ["custom-documents/managed.json"]
    }


@respx.mock
def test_knowledge_can_move_to_another_workspace_and_directory(tmp_path: Path) -> None:
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
            anythingllm_workspace_slug="research",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    job = JobRecord(
        id="move-knowledge",
        url="https://youtu.be/hhhhhhhhhhh",
        canonical_url="https://www.youtube.com/watch?v=hhhhhhhhhhh",
        workspace_slug="research",
        category_path="投资/虚拟币",
        status=JobStatus.completed,
        stage="completed",
        document_path="jobs/move-knowledge/document.md",
        sync_status=SyncStatus.synced,
        anythingllm_document_location="custom-documents/move.json",
        title="需要移动的知识",
    )
    manager.jobs[job.id] = job
    manager.storage.save(job)
    manager.catalog.create("research", "投资/虚拟币")
    manager.catalog.create("development", "Python/FastAPI")
    document = manager.storage.job_dir(job.id) / "document.md"
    document.write_text(
        "# 标题\n\n## 来源信息\n\n- 转录来源：YouTube 人工字幕\n- 知识目录：投资/虚拟币\n",
        encoding="utf-8",
    )

    target_detail_responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "workspace": [
                        {
                            "id": 2,
                            "name": "软件开发",
                            "slug": "development",
                            "documents": [],
                            "threads": [],
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "workspace": [
                        {
                            "slug": "development",
                            "documents": [{"docpath": "custom-documents/move.json"}],
                        }
                    ]
                },
            ),
        ]
    )
    target_details = respx.get(f"{base_url}/v1/workspace/development").mock(
        side_effect=lambda _: next(target_detail_responses)
    )
    target_update = respx.post(
        f"{base_url}/v1/workspace/development/update-embeddings"
    ).mock(return_value=httpx.Response(200, json={}))
    source_update = respx.post(
        f"{base_url}/v1/workspace/research/update-embeddings"
    ).mock(return_value=httpx.Response(200, json={}))
    source_details = respx.get(f"{base_url}/v1/workspace/research").mock(
        return_value=httpx.Response(
            200,
            json={"workspace": [{"slug": "research", "documents": []}]},
        )
    )

    client = ready_web_client(app)
    try:
        preview = client.get("/ui/api/jobs/move-knowledge/move-preview")
        moved = client.post(
            "/ui/api/jobs/move-knowledge/move",
            json={
                "target_workspace_slug": "development",
                "target_category_path": "Python/FastAPI",
            },
        )
    finally:
        client.__exit__(None, None, None)

    assert preview.status_code == 200
    assert preview.json()["workspace_slug"] == "research"
    assert moved.status_code == 200
    assert moved.json()["workspace_slug"] == "development"
    assert moved.json()["category_path"] == "Python/FastAPI"
    assert "- 知识目录：Python/FastAPI" in document.read_text(encoding="utf-8")
    assert "- 知识目录：投资/虚拟币" not in document.read_text(encoding="utf-8")
    assert target_details.call_count == 2
    assert target_update.call_count == 1
    assert source_update.call_count == 1
    assert source_details.call_count == 1
    assert json.loads(target_update.calls[0].request.content) == {
        "adds": ["custom-documents/move.json"],
        "deletes": [],
    }
    assert json.loads(source_update.calls[0].request.content) == {
        "adds": [],
        "deletes": ["custom-documents/move.json"],
    }


@respx.mock
def test_knowledge_can_move_between_directories_without_reembedding(
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
            anythingllm_workspace_slug="research",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    job = JobRecord(
        id="move-directory",
        url="https://youtu.be/jjjjjjjjjjj",
        canonical_url="https://www.youtube.com/watch?v=jjjjjjjjjjj",
        workspace_slug="research",
        category_path="投资/虚拟币",
        status=JobStatus.completed,
        stage="completed",
        document_path="jobs/move-directory/document.md",
        sync_status=SyncStatus.synced,
        anythingllm_document_location="custom-documents/directory.json",
        title="知识库内移动",
    )
    manager.jobs[job.id] = job
    manager.storage.save(job)
    manager.catalog.create("research", "投资/虚拟币")
    manager.catalog.create("research", "投资/宏观经济")
    document = manager.storage.job_dir(job.id) / "document.md"
    document.write_text(
        "# 标题\n\n- 转录来源：人工字幕\n- 知识目录：投资/虚拟币\n",
        encoding="utf-8",
    )
    workspace_details = respx.get(f"{base_url}/v1/workspace/research").mock(
        return_value=httpx.Response(
            200,
            json={
                "workspace": [
                    {
                        "id": 1,
                        "name": "投资研究",
                        "slug": "research",
                        "documents": [
                            {"docpath": "custom-documents/directory.json"}
                        ],
                    }
                ]
            },
        )
    )

    client = ready_web_client(app)
    try:
        moved = client.post(
            "/ui/api/jobs/move-directory/move",
            json={
                "target_workspace_slug": "research",
                "target_category_path": "投资/宏观经济",
            },
        )
    finally:
        client.__exit__(None, None, None)

    assert moved.status_code == 200
    assert moved.json()["workspace_slug"] == "research"
    assert moved.json()["category_path"] == "投资/宏观经济"
    assert workspace_details.call_count == 1
    assert "- 知识目录：投资/宏观经济" in document.read_text(encoding="utf-8")
    persisted = manager.storage.load_all()[job.id]
    assert persisted.workspace_slug == "research"
    assert persisted.category_path == "投资/宏观经济"


@respx.mock
def test_cross_workspace_move_rolls_back_target_when_source_removal_fails(
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
            anythingllm_workspace_slug="research",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    job = JobRecord(
        id="rollback-move",
        url="https://youtu.be/iiiiiiiiiii",
        canonical_url="https://www.youtube.com/watch?v=iiiiiiiiiii",
        workspace_slug="research",
        category_path="投资",
        status=JobStatus.completed,
        stage="completed",
        document_path="jobs/rollback-move/document.md",
        anythingllm_document_location="custom-documents/rollback.json",
    )
    manager.jobs[job.id] = job
    manager.storage.save(job)
    manager.catalog.create("development", "Python")
    document = manager.storage.job_dir(job.id) / "document.md"
    original = "# 标题\n\n- 转录来源：人工字幕\n- 知识目录：投资\n"
    document.write_text(original, encoding="utf-8")

    target_detail_responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "workspace": [
                        {
                            "id": 2,
                            "name": "软件开发",
                            "slug": "development",
                            "documents": [],
                        }
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "workspace": [
                        {
                            "slug": "development",
                            "documents": [{"docpath": "custom-documents/rollback.json"}],
                        }
                    ]
                },
            ),
        ]
    )
    respx.get(f"{base_url}/v1/workspace/development").mock(
        side_effect=lambda _: next(target_detail_responses)
    )
    target_update = respx.post(
        f"{base_url}/v1/workspace/development/update-embeddings"
    ).mock(
        side_effect=[
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
        ]
    )
    respx.post(f"{base_url}/v1/workspace/research/update-embeddings").mock(
        return_value=httpx.Response(500)
    )

    client = ready_web_client(app)
    try:
        failed = client.post(
            "/ui/api/jobs/rollback-move/move",
            json={
                "target_workspace_slug": "development",
                "target_category_path": "Python",
            },
        )
    finally:
        client.__exit__(None, None, None)

    assert failed.status_code == 409
    assert "从原知识库移除文档失败" in failed.json()["detail"]
    assert job.workspace_slug == "research"
    assert job.category_path == "投资"
    assert document.read_text(encoding="utf-8") == original
    assert target_update.call_count == 2
    assert json.loads(target_update.calls[1].request.content) == {
        "adds": [],
        "deletes": ["custom-documents/rollback.json"],
    }


def test_library_tree_content_and_archive_downloads(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            data_dir=tmp_path,
            ingestor_api_key=API_KEY,
            web_ui_username="admin",
            web_ui_password=INITIAL_WEB_PASSWORD,
            web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
            anythingllm_workspace_slug="research",
            anythingllm_auto_sync=False,
        )
    )
    manager = app.state.manager
    root_content = "# 根目录知识\n\n这是根目录正文。\n"
    nested_content = "# BTC 与 ETH\n\n## 摘要\n\n这是虚拟币分析正文。\n"
    jobs = [
        JobRecord(
            id="root-knowledge",
            url="https://youtu.be/kkkkkkkkkkk",
            canonical_url="https://www.youtube.com/watch?v=kkkkkkkkkkk",
            workspace_slug=None,
            category_path="",
            status=JobStatus.completed,
            stage="completed",
            title="根目录知识",
            document_path="jobs/root-knowledge/document.md",
        ),
        JobRecord(
            id="nested-knowledge",
            url="https://youtu.be/lllllllllll",
            canonical_url="https://www.youtube.com/watch?v=lllllllllll",
            workspace_slug="research",
            category_path="投资/虚拟币",
            status=JobStatus.completed,
            stage="completed",
            title="BTC/ETH:分析",
            uploader="研究频道",
            duration_seconds=321,
            transcript_source="youtube_manual",
            sync_status=SyncStatus.synced,
            document_path="jobs/nested-knowledge/document.md",
        ),
        JobRecord(
            id="missing-document",
            url="https://youtu.be/mmmmmmmmmmm",
            canonical_url="https://www.youtube.com/watch?v=mmmmmmmmmmm",
            workspace_slug="research",
            category_path="投资/股票",
            status=JobStatus.failed,
            stage="failed",
            title="尚未生成文档",
        ),
        JobRecord(
            id="other-workspace",
            url="https://youtu.be/nnnnnnnnnnn",
            canonical_url="https://www.youtube.com/watch?v=nnnnnnnnnnn",
            workspace_slug="development",
            category_path="Python",
            status=JobStatus.completed,
            stage="completed",
            title="不应出现在投资知识库",
        ),
    ]
    for job in jobs:
        manager.jobs[job.id] = job
        manager.storage.save(job)
        manager.catalog.register(manager.effective_workspace_slug(job), job.category_path)
    (manager.storage.job_dir("root-knowledge") / "document.md").write_text(
        root_content,
        encoding="utf-8",
        newline="\n",
    )
    (manager.storage.job_dir("nested-knowledge") / "document.md").write_text(
        nested_content,
        encoding="utf-8",
        newline="\n",
    )
    manager.catalog.create("research", "投资/股票")
    manager.catalog.create("research", "投资/宏观经济")

    client = ready_web_client(app)
    try:
        library = client.get("/ui/api/library?workspace_slug=research")
        content = client.get("/ui/api/jobs/nested-knowledge/content")
        single_download = client.get("/ui/api/jobs/nested-knowledge/download")
        workspace_download = client.get("/ui/api/workspaces/research/download")
        directory_download = client.get(
            "/ui/api/directories/download",
            params={"workspace_slug": "research", "path": "投资/虚拟币"},
        )
        unavailable = client.get("/ui/api/jobs/missing-document/content")
        missing_directory = client.get(
            "/ui/api/directories/download",
            params={"workspace_slug": "research", "path": "不存在"},
        )
        unsafe_directory = client.get(
            "/ui/api/directories/download",
            params={"workspace_slug": "research", "path": "../投资"},
        )
    finally:
        client.__exit__(None, None, None)

    assert library.status_code == 200
    payload = library.json()
    assert payload["workspace_slug"] == "research"
    assert payload["root"] == {
        "direct_jobs": 1,
        "total_jobs": 3,
        "downloadable_documents": 2,
        "total_bytes": len(root_content.encode()) + len(nested_content.encode()),
        "active_jobs": 0,
    }
    assert [document["id"] for document in payload["documents"]] == [
        "root-knowledge",
        "missing-document",
        "nested-knowledge",
    ]
    nested = next(
        document for document in payload["documents"] if document["id"] == "nested-knowledge"
    )
    assert nested["content_available"] is True
    assert nested["size_bytes"] == len(nested_content.encode())
    assert nested["uploader"] == "研究频道"
    assert next(
        document for document in payload["documents"] if document["id"] == "missing-document"
    )["content_available"] is False
    directory_details = {item["path"]: item for item in payload["directories"]}
    assert directory_details["投资"]["total_jobs"] == 2
    assert directory_details["投资"]["downloadable_documents"] == 1
    assert directory_details["投资/虚拟币"]["direct_jobs"] == 1
    assert directory_details["投资/宏观经济"]["total_jobs"] == 0

    assert content.status_code == 200
    assert content.headers["cache-control"] == "no-store"
    assert content.json()["content"] == nested_content
    assert content.json()["line_count"] == 5
    assert content.json()["category_path"] == "投资/虚拟币"
    assert single_download.status_code == 200
    assert single_download.content == nested_content.encode()
    assert single_download.headers["content-type"].startswith("text/markdown")
    assert "attachment" in single_download.headers["content-disposition"]

    assert workspace_download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(workspace_download.content)) as archive:
        workspace_names = set(archive.namelist())
        assert "research/" in workspace_names
        assert "research/投资/宏观经济/" in workspace_names
        assert "research/知识库信息.json" in workspace_names
        markdown_names = sorted(name for name in workspace_names if name.endswith(".md"))
        assert len(markdown_names) == 2
        assert any(name.startswith("research/投资/虚拟币/BTC_ETH_分析") for name in markdown_names)
        manifest = json.loads(archive.read("research/知识库信息.json"))
        assert {document["id"] for document in manifest["documents"]} == {
            "root-knowledge",
            "nested-knowledge",
        }
        assert manifest["scope"] == "root"

    assert directory_download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(directory_download.content)) as archive:
        directory_names = set(archive.namelist())
        directory_markdown = [name for name in directory_names if name.endswith(".md")]
        assert len(directory_markdown) == 1
        assert directory_markdown[0].startswith("research/投资/虚拟币/")
        directory_manifest = json.loads(archive.read("research/知识库信息.json"))
        assert [document["id"] for document in directory_manifest["documents"]] == [
            "nested-knowledge"
        ]
        assert directory_manifest["scope"] == "投资/虚拟币"

    assert unavailable.status_code == 404
    assert missing_directory.status_code == 404
    assert unsafe_directory.status_code == 422
    assert not list((tmp_path / "cache").glob(".knowledge-export-*.zip"))
