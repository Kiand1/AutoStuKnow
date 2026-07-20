import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.background import BackgroundTask

from .catalog import normalize_directory_path, path_is_within
from .config import Settings
from .exports import archive_download_name, build_knowledge_archive, knowledge_filename
from .fusion import FusionManager
from .mcp_server import McpAccessController, McpBearerAuthMiddleware, create_mcp_server
from .models import (
    BatchJobRequest,
    FusionRecord,
    JobRecord,
    JobRequest,
    JobSubmission,
    WebDirectoryDeleteRequest,
    WebDirectoryRequest,
    WebFusionDeleteRequest,
    WebFusionGenerateRequest,
    WebFusionPublishRequest,
    WebJobDeleteRequest,
    WebJobMoveRequest,
    WebLoginRequest,
    WebMcpSettingsRequest,
    WebPasswordChangeRequest,
    WebWorkspaceCreateRequest,
    WebWorkspaceDeleteRequest,
    WebWorkspaceTargetRequest,
)
from .pipeline import (
    JobManager,
    PipelineError,
    create_anythingllm_workspace,
    get_anythingllm_workspace,
    list_anythingllm_workspaces,
)
from .urls import canonicalize_youtube_url
from .web_auth import (
    SESSION_COOKIE,
    LoginThrottle,
    WebCredentialStore,
    WebSession,
    WebSessionSigner,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    manager = JobManager(resolved_settings)
    fusion_manager = FusionManager(resolved_settings, manager)
    mcp_access = McpAccessController(resolved_settings.data_dir, resolved_settings.mcp_enabled)
    mcp_server = create_mcp_server(resolved_settings, manager, fusion_manager)
    credential_store = WebCredentialStore(
        resolved_settings.data_dir,
        resolved_settings.web_ui_password,
    )
    session_signer = WebSessionSigner(
        resolved_settings.web_ui_session_secret,
        resolved_settings.web_ui_session_ttl_hours * 3600,
    )
    login_throttle = LoginThrottle()
    web_index = Path(__file__).with_name("web") / "index.html"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with mcp_server.session_manager.run():
            yield

    application = FastAPI(
        title="AutoStuKnow Ingestor",
        version="1.1.0",
        description=(
            "YouTube → Whisper → DeepSeek summary/fusion → AnythingLLM logical knowledge base"
        ),
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.manager = manager
    application.state.fusion_manager = fusion_manager
    application.state.credential_store = credential_store
    application.state.session_signer = session_signer
    application.state.mcp_server = mcp_server
    application.state.mcp_access = mcp_access

    def available_document(job: JobRecord) -> tuple[Path | None, int]:
        document_path = manager.document_file(job)
        if document_path is None:
            return None, 0
        try:
            return document_path, document_path.stat().st_size
        except OSError:
            return None, 0

    def fusion_payload(record: FusionRecord) -> dict[str, object]:
        versions = fusion_manager.versions(record.topic_id)
        return {
            **record.model_dump(mode="json"),
            "source_count": len(record.source_job_ids),
            "content_available": fusion_manager.document_file(record) is not None,
            "version_count": len(versions),
            "is_latest": bool(versions and versions[0].id == record.id),
        }

    def workspace_export_sources(workspace_slug: str) -> list[tuple[JobRecord, Path]]:
        sources: list[tuple[JobRecord, Path]] = []
        for job in manager.jobs_in_workspace(workspace_slug):
            document_path, _ = available_document(job)
            if document_path is not None:
                sources.append((job, document_path))
        return sources

    async def archive_response(
        workspace_slug: str,
        selected_path: str = "",
    ) -> FileResponse:
        directories = manager.directory_paths(workspace_slug)
        if selected_path and selected_path not in directories:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目录不存在")
        try:
            archive_path = await run_in_threadpool(
                build_knowledge_archive,
                resolved_settings.data_dir / "cache",
                workspace_slug,
                workspace_export_sources(workspace_slug),
                directories,
                selected_path,
            )
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"生成下载压缩包失败：{exc}",
            ) from exc
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=archive_download_name(workspace_slug, selected_path),
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        supplied = x_api_key or ""
        if not hmac.compare_digest(supplied, resolved_settings.ingestor_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的 X-API-Key",
            )

    def active_web_session(request: Request) -> WebSession | None:
        web_session = session_signer.verify(request.cookies.get(SESSION_COOKIE))
        if (
            web_session is None
            or web_session.username != resolved_settings.web_ui_username
            or web_session.credential_revision != credential_store.revision
        ):
            return None
        return web_session

    async def require_authenticated_web_session(request: Request) -> WebSession:
        web_session = active_web_session(request)
        if web_session is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
        return web_session

    async def require_web_session(request: Request) -> str:
        web_session = await require_authenticated_web_session(request)
        if credential_store.must_change_password:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="首次登录必须先修改初始密码",
            )
        return web_session.username

    def authenticated_response(credential_revision: int) -> JSONResponse:
        response = JSONResponse(
            {
                "authenticated": True,
                "must_change_password": credential_store.must_change_password,
                "username": resolved_settings.web_ui_username,
            }
        )
        response.set_cookie(
            key=SESSION_COOKIE,
            value=session_signer.create(
                resolved_settings.web_ui_username,
                credential_revision=credential_revision,
            ),
            max_age=resolved_settings.web_ui_session_ttl_hours * 3600,
            httponly=True,
            secure=resolved_settings.web_ui_secure_cookie,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @application.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    @application.get("/ui", include_in_schema=False, response_class=FileResponse)
    async def web_ui() -> FileResponse:
        return FileResponse(
            web_index,
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/ui/api/session", include_in_schema=False)
    async def web_session(request: Request) -> dict[str, str | bool]:
        current_session = active_web_session(request)
        authenticated = current_session is not None
        return {
            "authenticated": authenticated,
            "must_change_password": (
                credential_store.must_change_password if authenticated else False
            ),
            "username": current_session.username if current_session else "",
        }

    @application.post("/ui/api/login", include_in_schema=False)
    async def web_login(request: Request, credentials: WebLoginRequest) -> JSONResponse:
        client_id = request.client.host if request.client else "unknown"
        if not login_throttle.allow(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="登录尝试过多，请十分钟后再试",
            )
        username_matches = hmac.compare_digest(
            credentials.username.encode("utf-8"),
            resolved_settings.web_ui_username.encode("utf-8"),
        )
        password_matches = credential_store.verify(credentials.password)
        if not (username_matches and password_matches):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        login_throttle.clear(client_id)
        return authenticated_response(credential_store.revision)

    @application.post("/ui/api/password", include_in_schema=False)
    async def web_change_password(
        request: Request,
        password_change: WebPasswordChangeRequest,
    ) -> JSONResponse:
        await require_authenticated_web_session(request)
        changing_initial_password = credential_store.must_change_password
        if not changing_initial_password:
            current_password = password_change.current_password or ""
            if not credential_store.verify(current_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="当前密码错误",
                )
        if credential_store.verify(password_change.new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="新密码不能与当前密码相同",
            )
        revision = credential_store.set_password(password_change.new_password)
        return authenticated_response(revision)

    @application.get("/ui/api/workspaces", include_in_schema=False)
    async def web_workspaces(
        _: str = Depends(require_web_session),
    ) -> dict[str, list[dict[str, object]]]:
        try:
            workspaces = await list_anythingllm_workspaces(resolved_settings)
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        logical_by_slug: dict[str, dict[str, str]] = {}
        for logical in fusion_manager.list_logical_bases():
            logical_by_slug[logical.source_workspace_slug] = {
                "logical_kb_id": logical.id,
                "logical_role": "source",
                "logical_name": logical.name,
            }
            if logical.fusion_workspace_slug:
                logical_by_slug[logical.fusion_workspace_slug] = {
                    "logical_kb_id": logical.id,
                    "logical_role": "fusion",
                    "logical_name": logical.name,
                }
        return {
            "workspaces": [
                {**workspace, **logical_by_slug.get(str(workspace["slug"]), {})}
                for workspace in workspaces
            ]
        }

    @application.get("/ui/api/logical-knowledge-bases", include_in_schema=False)
    async def web_logical_knowledge_bases(
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        items = []
        for logical in fusion_manager.list_logical_bases():
            latest = fusion_manager.list_latest(logical.source_workspace_slug)
            items.append(
                {
                    **logical.model_dump(mode="json"),
                    "fusion_topic_count": len(latest),
                    "published_topic_count": sum(
                        item.status.value == "published" for item in latest
                    ),
                }
            )
        return {"logical_knowledge_bases": items, "count": len(items)}

    @application.get("/ui/api/fusions", include_in_schema=False)
    async def web_fusions(
        source_workspace_slug: str | None = Query(default=None, max_length=128),
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        records = fusion_manager.list_latest(
            source_workspace_slug.strip() if source_workspace_slug else None
        )
        return {"fusions": [fusion_payload(record) for record in records]}

    @application.post("/ui/api/fusions", include_in_schema=False)
    async def web_generate_fusion(
        fusion_request: WebFusionGenerateRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        try:
            record = await fusion_manager.generate(fusion_request)
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {"fusion": fusion_payload(record)}

    @application.get("/ui/api/fusions/{record_id}", include_in_schema=False)
    async def web_fusion_detail(
        record_id: str,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        record = fusion_manager.get(record_id)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="融合知识不存在")
        return {"fusion": fusion_payload(record)}

    @application.get("/ui/api/fusions/{record_id}/content", include_in_schema=False)
    async def web_fusion_content(
        record_id: str,
        _: str = Depends(require_web_session),
    ) -> Response:
        record = fusion_manager.get(record_id)
        document = fusion_manager.document_file(record) if record else None
        if record is None or document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="融合草稿不存在或尚未生成",
            )
        try:
            content = await run_in_threadpool(document.read_text, encoding="utf-8")
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"读取融合草稿失败：{exc}",
            ) from exc
        return Response(content=content, media_type="text/markdown; charset=utf-8")

    @application.get("/ui/api/fusion-topics/{topic_id}/versions", include_in_schema=False)
    async def web_fusion_versions(
        topic_id: str,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        versions = fusion_manager.versions(topic_id)
        if not versions:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="融合知识不存在")
        return {"versions": [fusion_payload(record) for record in versions]}

    @application.post("/ui/api/fusions/{record_id}/publish", include_in_schema=False)
    async def web_publish_fusion(
        record_id: str,
        publish_request: WebFusionPublishRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        try:
            record = await fusion_manager.publish(record_id, publish_request.confirm_title)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="融合知识不存在",
            ) from exc
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {"fusion": fusion_payload(record)}

    @application.delete("/ui/api/fusion-topics/{topic_id}", include_in_schema=False)
    async def web_delete_fusion_topic(
        topic_id: str,
        delete_request: WebFusionDeleteRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        try:
            deleted_versions = await fusion_manager.delete_topic(
                topic_id,
                delete_request.confirm_title,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="融合知识不存在",
            ) from exc
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {"deleted": True, "topic_id": topic_id, "deleted_versions": deleted_versions}

    @application.post("/ui/api/workspaces", include_in_schema=False)
    async def web_create_workspace(
        workspace_request: WebWorkspaceCreateRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, dict[str, object]]:
        try:
            workspace = await create_anythingllm_workspace(
                resolved_settings,
                workspace_request.name,
            )
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        return {"workspace": workspace}

    @application.post("/ui/api/workspaces/delete-preview", include_in_schema=False)
    async def web_workspace_delete_preview(
        workspace_request: WebWorkspaceTargetRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        try:
            workspace = await get_anythingllm_workspace(
                resolved_settings,
                workspace_request.workspace_slug,
            )
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        matched_jobs = manager.jobs_in_workspace(workspace_request.workspace_slug)
        return {
            "workspace": workspace,
            "directories": len(manager.directory_paths(workspace_request.workspace_slug)),
            "managed_jobs": len(matched_jobs),
            "managed_documents": sum(
                bool(job.anythingllm_document_location) for job in matched_jobs
            ),
            "active_jobs": sum(
                job.status.value in {"queued", "running"} for job in matched_jobs
            ),
        }

    @application.delete(
        "/ui/api/workspaces/{workspace_slug}",
        include_in_schema=False,
    )
    async def web_delete_workspace(
        workspace_slug: str,
        delete_request: WebWorkspaceDeleteRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, int | str | bool]:
        try:
            result = await manager.delete_workspace(
                workspace_slug.strip(),
                delete_request.confirm_name,
            )
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {"workspace_slug": workspace_slug, "deleted": True, **result}

    @application.get(
        "/ui/api/workspaces/{workspace_slug}/download",
        include_in_schema=False,
    )
    async def web_download_workspace(
        workspace_slug: str,
        _: str = Depends(require_web_session),
    ) -> FileResponse:
        normalized_slug = workspace_slug.strip()
        if not normalized_slug:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="知识库不能为空",
            )
        return await archive_response(normalized_slug)

    @application.get("/ui/api/directories", include_in_schema=False)
    async def web_directories(
        workspace_slug: str = Query(min_length=1, max_length=128),
        _: str = Depends(require_web_session),
    ) -> dict[str, list[dict[str, object]]]:
        directories: list[dict[str, object]] = []
        for path in manager.directory_paths(workspace_slug):
            matched_jobs = manager.jobs_in_directory(workspace_slug, path)
            directories.append(
                {
                    "path": path,
                    "depth": path.count("/") + 1,
                    "direct_jobs": sum(job.category_path == path for job in matched_jobs),
                    "total_jobs": len(matched_jobs),
                    "synced_documents": sum(
                        bool(job.anythingllm_document_location) for job in matched_jobs
                    ),
                    "active_jobs": sum(
                        job.status.value in {"queued", "running"} for job in matched_jobs
                    ),
                }
            )
        return {"directories": directories}

    @application.get("/ui/api/library", include_in_schema=False)
    async def web_library(
        workspace_slug: str = Query(min_length=1, max_length=128),
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        normalized_slug = workspace_slug.strip()
        if not normalized_slug:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="知识库不能为空",
            )
        jobs = sorted(
            manager.jobs_in_workspace(normalized_slug),
            key=lambda item: (item.category_path.casefold(), (item.title or "").casefold()),
        )
        documents: list[dict[str, object]] = []
        sizes: dict[str, int] = {}
        availability: dict[str, bool] = {}
        for job in jobs:
            document_path, size_bytes = available_document(job)
            sizes[job.id] = size_bytes
            availability[job.id] = document_path is not None
            documents.append(
                {
                    "id": job.id,
                    "title": job.title or job.canonical_url,
                    "category_path": job.category_path,
                    "status": job.status.value,
                    "stage": job.stage,
                    "source_url": job.canonical_url,
                    "uploader": job.uploader,
                    "duration_seconds": job.duration_seconds,
                    "transcript_source": job.transcript_source,
                    "sync_status": job.sync_status.value,
                    "updated_at": job.updated_at.isoformat(),
                    "content_available": availability[job.id],
                    "size_bytes": size_bytes,
                }
            )

        directories: list[dict[str, object]] = []
        for path in manager.directory_paths(normalized_slug):
            matched_jobs = [job for job in jobs if path_is_within(job.category_path, path)]
            direct_jobs = [job for job in matched_jobs if job.category_path == path]
            directories.append(
                {
                    "path": path,
                    "depth": path.count("/") + 1,
                    "direct_jobs": len(direct_jobs),
                    "total_jobs": len(matched_jobs),
                    "downloadable_documents": sum(
                        availability[job.id] for job in matched_jobs
                    ),
                    "total_bytes": sum(sizes[job.id] for job in matched_jobs),
                    "active_jobs": sum(
                        job.status.value in {"queued", "running"} for job in matched_jobs
                    ),
                }
            )
        return {
            "workspace_slug": normalized_slug,
            "root": {
                "direct_jobs": sum(not job.category_path for job in jobs),
                "total_jobs": len(jobs),
                "downloadable_documents": sum(availability.values()),
                "total_bytes": sum(sizes.values()),
                "active_jobs": sum(
                    job.status.value in {"queued", "running"} for job in jobs
                ),
            },
            "directories": directories,
            "documents": documents,
        }

    @application.get("/ui/api/directories/download", include_in_schema=False)
    async def web_download_directory(
        workspace_slug: str = Query(min_length=1, max_length=128),
        path: str = Query(min_length=1, max_length=512),
        _: str = Depends(require_web_session),
    ) -> FileResponse:
        try:
            normalized_path = normalize_directory_path(path)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        if not normalized_path:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="目录不能为空",
            )
        return await archive_response(workspace_slug.strip(), normalized_path)

    @application.post("/ui/api/directories", include_in_schema=False)
    async def web_create_directory(
        directory_request: WebDirectoryRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, str]:
        try:
            path = manager.create_directory(
                directory_request.workspace_slug,
                directory_request.path,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        return {"workspace_slug": directory_request.workspace_slug, "path": path}

    @application.post("/ui/api/directories/delete-preview", include_in_schema=False)
    async def web_directory_delete_preview(
        directory_request: WebDirectoryRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        matched_jobs = manager.jobs_in_directory(
            directory_request.workspace_slug,
            directory_request.path,
        )
        descendant_directories = [
            path
            for path in manager.directory_paths(directory_request.workspace_slug)
            if path == directory_request.path
            or path.startswith(f"{directory_request.path}/")
        ]
        if not descendant_directories:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目录不存在")
        return {
            "workspace_slug": directory_request.workspace_slug,
            "path": directory_request.path,
            "directories": len(descendant_directories),
            "jobs": len(matched_jobs),
            "synced_documents": sum(
                bool(job.anythingllm_document_location) for job in matched_jobs
            ),
            "active_jobs": sum(
                job.status.value in {"queued", "running"} for job in matched_jobs
            ),
        }

    @application.delete("/ui/api/directories", include_in_schema=False)
    async def web_delete_directory(
        directory_request: WebDirectoryDeleteRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, int | str]:
        if directory_request.confirm_path != directory_request.path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="确认目录与待删除目录不一致",
            )
        if directory_request.path not in manager.directory_paths(
            directory_request.workspace_slug
        ):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="目录不存在")
        try:
            result = await manager.delete_directory(
                directory_request.workspace_slug,
                directory_request.path,
            )
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {
            "workspace_slug": directory_request.workspace_slug,
            "path": directory_request.path,
            **result,
        }

    @application.post("/ui/api/logout", include_in_schema=False)
    async def web_logout() -> Response:
        response = Response(status_code=status.HTTP_204_NO_CONTENT)
        response.delete_cookie(SESSION_COOKIE, path="/", samesite="strict")
        response.headers["Cache-Control"] = "no-store"
        return response

    def mcp_settings_payload(request: Request) -> dict[str, str | bool]:
        endpoint = f"{str(request.base_url).rstrip('/')}/mcp"
        return {
            "enabled": mcp_access.enabled,
            "endpoint": endpoint,
            "authentication": "Bearer token",
            "dedicated_key": bool(resolved_settings.mcp_api_key),
        }

    @application.get("/ui/api/mcp", include_in_schema=False)
    async def web_mcp_settings(
        request: Request,
        _: str = Depends(require_web_session),
    ) -> dict[str, str | bool]:
        return mcp_settings_payload(request)

    @application.post("/ui/api/mcp", include_in_schema=False)
    async def web_update_mcp_settings(
        request: Request,
        update: WebMcpSettingsRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, str | bool]:
        mcp_access.set_enabled(update.enabled)
        return mcp_settings_payload(request)

    @application.post("/ui/api/jobs/batch", include_in_schema=False)
    async def web_batch_jobs(
        request: BatchJobRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, list[dict[str, object]]]:
        workspace_slug = (request.workspace_slug or "").strip()
        if not workspace_slug:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="请选择目标知识库",
            )
        items: list[dict[str, object]] = []
        accepted: dict[str, JobSubmission] = {}
        for raw_url in request.urls:
            submitted_url = raw_url.strip()
            try:
                canonical_url = canonicalize_youtube_url(submitted_url)
            except HTTPException as exc:
                items.append({"url": submitted_url, "accepted": False, "error": exc.detail})
                continue
            if canonical_url in accepted:
                previous = accepted[canonical_url]
                items.append(
                    {
                        "url": submitted_url,
                        "accepted": True,
                        "input_duplicate": True,
                        "workspace_slug": workspace_slug,
                        "category_path": request.category_path,
                        **previous.model_dump(mode="json"),
                    }
                )
                continue
            job, deduplicated = await manager.submit(
                JobRequest(
                    url=canonical_url,
                    language=request.language,
                    workspace_slug=workspace_slug,
                    category_path=request.category_path,
                    force=request.force,
                )
            )
            submission = JobSubmission(
                job_id=job.id,
                status=job.status,
                stage=job.stage,
                deduplicated=deduplicated,
                status_path=f"/jobs/{job.id}",
            )
            accepted[canonical_url] = submission
            items.append(
                {
                    "url": submitted_url,
                    "accepted": True,
                    "input_duplicate": False,
                    "workspace_slug": workspace_slug,
                    "category_path": request.category_path,
                    **submission.model_dump(mode="json"),
                }
            )
        return {"items": items}

    @application.get("/ui/api/jobs", include_in_schema=False)
    async def web_jobs(
        ids: str | None = Query(default=None, max_length=4_000),
        _: str = Depends(require_web_session),
    ) -> list[JobRecord]:
        if not ids:
            return manager.list_jobs(100)
        requested_ids = {value.strip() for value in ids.split(",") if value.strip()}
        return [job for job_id in requested_ids if (job := manager.get(job_id)) is not None]

    @application.get("/ui/api/jobs/{job_id}/content", include_in_schema=False)
    async def web_job_content(
        job_id: str,
        _: str = Depends(require_web_session),
    ) -> JSONResponse:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识不存在")
        document_path, size_bytes = available_document(job)
        if document_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="这条知识还没有可预览的 Markdown 文档",
            )
        try:
            content = await run_in_threadpool(document_path.read_text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"读取知识内容失败：{exc}",
            ) from exc
        return JSONResponse(
            {
                "id": job.id,
                "title": job.title or job.canonical_url,
                "workspace_slug": manager.effective_workspace_slug(job),
                "category_path": job.category_path,
                "source_url": job.canonical_url,
                "updated_at": job.updated_at.isoformat(),
                "size_bytes": size_bytes,
                "line_count": len(content.splitlines()),
                "character_count": len(content),
                "content": content,
            },
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/ui/api/jobs/{job_id}/download", include_in_schema=False)
    async def web_download_job(
        job_id: str,
        _: str = Depends(require_web_session),
    ) -> FileResponse:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识不存在")
        document_path, _ = available_document(job)
        if document_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="这条知识还没有可下载的 Markdown 文档",
            )
        return FileResponse(
            document_path,
            media_type="text/markdown; charset=utf-8",
            filename=knowledge_filename(job),
            headers={"Cache-Control": "no-store"},
        )

    @application.get("/ui/api/jobs/{job_id}/delete-preview", include_in_schema=False)
    async def web_job_delete_preview(
        job_id: str,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识不存在")
        return {
            "job_id": job.id,
            "title": job.title or job.canonical_url,
            "workspace_slug": job.workspace_slug,
            "category_path": job.category_path,
            "synced": bool(job.anythingllm_document_location),
            "active": job.status.value in {"queued", "running"},
        }

    @application.get("/ui/api/jobs/{job_id}/move-preview", include_in_schema=False)
    async def web_job_move_preview(
        job_id: str,
        _: str = Depends(require_web_session),
    ) -> dict[str, object]:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="知识不存在")
        return {
            "job_id": job.id,
            "title": job.title or job.canonical_url,
            "workspace_slug": manager.effective_workspace_slug(job),
            "category_path": job.category_path,
            "synced": bool(job.anythingllm_document_location),
            "active": job.status.value in {"queued", "running"},
        }

    @application.post("/ui/api/jobs/{job_id}/move", include_in_schema=False)
    async def web_move_job(
        job_id: str,
        move_request: WebJobMoveRequest,
        _: str = Depends(require_web_session),
    ) -> JobRecord:
        try:
            return await manager.move_job(
                job_id,
                move_request.target_workspace_slug,
                move_request.target_category_path,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识不存在",
            ) from exc
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

    @application.delete("/ui/api/jobs/{job_id}", include_in_schema=False)
    async def web_delete_job(
        job_id: str,
        delete_request: WebJobDeleteRequest,
        _: str = Depends(require_web_session),
    ) -> dict[str, str | bool]:
        if not hmac.compare_digest(delete_request.confirm_job_id, job_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="删除确认信息不匹配",
            )
        try:
            deleted = await manager.delete_job(job_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="知识不存在",
            ) from exc
        except PipelineError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return {
            "job_id": deleted.id,
            "deleted": True,
            "anythingllm_deleted": bool(deleted.anythingllm_document_location),
        }

    @application.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def readiness() -> dict[str, str]:
        results: dict[str, str] = {"ingestor": "ok"}
        checks = {
            "whisper": f"{resolved_settings.whisper_base_url.rstrip('/')}/v1/models",
            "anythingllm": f"{resolved_settings.anythingllm_base_url.rstrip('/')}/ping",
        }
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            for name, url in checks.items():
                try:
                    response = await client.get(url)
                    results[name] = "ok" if response.is_success else f"http_{response.status_code}"
                except httpx.HTTPError:
                    results[name] = "unreachable"
        return results

    @application.post(
        "/jobs",
        response_model=JobSubmission,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_api_key)],
    )
    async def create_job(request: JobRequest) -> JobSubmission:
        job, deduplicated = await manager.submit(request)
        return JobSubmission(
            job_id=job.id,
            status=job.status,
            stage=job.stage,
            deduplicated=deduplicated,
            status_path=f"/jobs/{job.id}",
        )

    @application.get(
        "/jobs",
        response_model=list[JobRecord],
        dependencies=[Depends(require_api_key)],
    )
    async def list_jobs(limit: int = Query(default=50, ge=1, le=500)) -> list[JobRecord]:
        return manager.list_jobs(limit)

    @application.get(
        "/jobs/{job_id}",
        response_model=JobRecord,
        dependencies=[Depends(require_api_key)],
    )
    async def get_job(job_id: str) -> JobRecord:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        return job

    @application.get(
        "/jobs/{job_id}/document",
        response_class=FileResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def get_document(job_id: str) -> FileResponse:
        job = manager.get(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")
        path = manager.document_file(job)
        if not path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档尚未生成")
        return FileResponse(path, media_type="text/markdown", filename=f"{job.id}.md")

    @application.post(
        "/jobs/{job_id}/sync",
        response_model=JobRecord,
        dependencies=[Depends(require_api_key)],
    )
    async def sync_job(job_id: str, workspace_slug: str | None = None) -> JobRecord:
        try:
            return await manager.sync(job_id, workspace_slug)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在") from exc
        except PipelineError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    # Keep this catch-all mount last so the existing UI/API routes win first.
    application.mount(
        "/",
        McpBearerAuthMiddleware(
            mcp_server.streamable_http_app(),
            resolved_settings.effective_mcp_api_key,
            mcp_access,
        ),
        name="mcp",
    )
    return application


app = create_app()
