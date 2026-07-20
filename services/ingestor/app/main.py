import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from .config import Settings
from .models import (
    BatchJobRequest,
    JobRecord,
    JobRequest,
    JobSubmission,
    WebDirectoryDeleteRequest,
    WebDirectoryRequest,
    WebJobDeleteRequest,
    WebJobMoveRequest,
    WebLoginRequest,
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
        yield

    application = FastAPI(
        title="AutoStuKnow Ingestor",
        version="1.0.0",
        description="YouTube → Faster Whisper → OpenAI-compatible summary → AnythingLLM",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.manager = manager
    application.state.credential_store = credential_store
    application.state.session_signer = session_signer

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
        return {"workspaces": workspaces}

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

    return application


app = create_app()
