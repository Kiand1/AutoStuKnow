import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from .config import Settings
from .models import BatchJobRequest, JobRecord, JobRequest, JobSubmission, WebLoginRequest
from .pipeline import JobManager, PipelineError
from .urls import canonicalize_youtube_url
from .web_auth import SESSION_COOKIE, LoginThrottle, WebSessionSigner


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    manager = JobManager(resolved_settings)
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
    application.state.session_signer = session_signer

    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        supplied = x_api_key or ""
        if not hmac.compare_digest(supplied, resolved_settings.ingestor_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的 X-API-Key",
            )

    async def require_web_session(request: Request) -> str:
        username = session_signer.verify(request.cookies.get(SESSION_COOKIE))
        if username != resolved_settings.web_ui_username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
        return username

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
        username = session_signer.verify(request.cookies.get(SESSION_COOKIE))
        authenticated = username == resolved_settings.web_ui_username
        return {
            "authenticated": authenticated,
            "username": username if authenticated else "",
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
            credentials.username,
            resolved_settings.web_ui_username,
        )
        password_matches = hmac.compare_digest(
            credentials.password,
            resolved_settings.web_ui_password,
        )
        if not (username_matches and password_matches):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
        login_throttle.clear(client_id)
        response = JSONResponse(
            {"authenticated": True, "username": resolved_settings.web_ui_username}
        )
        response.set_cookie(
            key=SESSION_COOKIE,
            value=session_signer.create(resolved_settings.web_ui_username),
            max_age=resolved_settings.web_ui_session_ttl_hours * 3600,
            httponly=True,
            secure=resolved_settings.web_ui_secure_cookie,
            samesite="strict",
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

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
                        **previous.model_dump(mode="json"),
                    }
                )
                continue
            job, deduplicated = await manager.submit(
                JobRequest(
                    url=canonical_url,
                    language=request.language,
                    workspace_slug=request.workspace_slug,
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
