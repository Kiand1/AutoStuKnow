import asyncio
import hmac
import json
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Receive, Scope, Send

from .catalog import normalize_directory_path, path_is_within
from .config import Settings
from .fusion import FusionManager
from .models import FusionRecord, JobRecord, LogicalKnowledgeBase
from .pipeline import (
    JobManager,
    PipelineError,
    get_anythingllm_workspace,
    list_anythingllm_workspaces,
)


class McpAccessController:
    """Persist the runtime MCP switch independently from container configuration."""

    def __init__(self, data_dir: Path, default_enabled: bool):
        self.path = data_dir / "mcp-settings.json"
        self._lock = threading.RLock()
        self._enabled = self._load(default_enabled)

    def _load(self, default_enabled: bool) -> bool:
        if not self.path.is_file():
            return default_enabled
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default_enabled
        enabled = payload.get("enabled") if isinstance(payload, dict) else None
        return enabled if isinstance(enabled, bool) else default_enabled

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({"version": 1, "enabled": enabled}, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
            self._enabled = enabled
            return self._enabled


class McpBearerAuthMiddleware:
    """Protect every MCP transport request with a static bearer token."""

    def __init__(self, app: ASGIApp, token: str, access: McpAccessController):
        self.app = app
        self.token = token.encode("utf-8")
        self.access = access

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self.access.enabled:
            body = b'{"error":"AutoStuKnow MCP is disabled"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": [
                        (b"content-type", b"application/json; charset=utf-8"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"")
        scheme, _, supplied = authorization.partition(b" ")
        authorized = (
            scheme.lower() == b"bearer"
            and bool(supplied)
            and hmac.compare_digest(supplied, self.token)
        )
        if authorized:
            await self.app(scope, receive, send)
            return

        body = b'{"error":"invalid or missing MCP bearer token"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    (b"www-authenticate", b'Bearer realm="AutoStuKnow MCP"'),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _job_summary(job: JobRecord, manager: JobManager) -> dict[str, object]:
    return {
        "job_id": job.id,
        "title": job.title or job.canonical_url,
        "category_path": job.category_path,
        "source_url": job.canonical_url,
        "status": job.status.value,
        "has_document": manager.document_file(job) is not None,
        "synced_to_anythingllm": bool(job.anythingllm_document_location),
    }


def _knowledge_tree(manager: JobManager, workspace_slug: str) -> dict[str, object]:
    root: dict[str, Any] = {"path": "", "directories": [], "knowledge": []}
    nodes: dict[str, dict[str, Any]] = {"": root}

    for path in manager.directory_paths(workspace_slug):
        current = ""
        parent = root
        for part in path.split("/"):
            current = f"{current}/{part}".strip("/")
            node = nodes.get(current)
            if node is None:
                node = {
                    "name": part,
                    "path": current,
                    "directories": [],
                    "knowledge": [],
                }
                nodes[current] = node
                parent["directories"].append(node)
            parent = node

    jobs = sorted(
        manager.jobs_in_workspace(workspace_slug),
        key=lambda item: ((item.category_path or "").casefold(), (item.title or "").casefold()),
    )
    for job in jobs:
        parent = nodes.get(job.category_path, root)
        parent["knowledge"].append(_job_summary(job, manager))

    return root


def _metadata_directory(metadata: dict[str, Any]) -> str:
    description = str(metadata.get("description") or "")
    marker = "; directory: "
    if marker not in description:
        return ""
    try:
        value = description.split(marker, 1)[1].split(";", 1)[0].strip()
        return normalize_directory_path(value)
    except ValueError:
        return ""


def _matches_category(
    result_category: str,
    selected_category: str,
    include_subdirectories: bool,
) -> bool:
    if not selected_category:
        return True
    if include_subdirectories:
        return path_is_within(result_category, selected_category)
    return result_category == selected_category


async def _vector_search(
    settings: Settings,
    workspace_slug: str,
    query: str,
    top_n: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    if not settings.anythingllm_api_key.strip():
        raise PipelineError("尚未配置 ANYTHINGLLM_API_KEY")
    encoded_slug = quote(workspace_slug, safe="")
    endpoint = (
        f"{settings.anythingllm_base_url.rstrip('/')}/v1/workspace/"
        f"{encoded_slug}/vector-search"
    )
    headers = {"Authorization": f"Bearer {settings.anythingllm_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json={
                    "query": query,
                    "topN": top_n,
                    "scoreThreshold": score_threshold,
                },
            )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PipelineError(f"AnythingLLM 向量检索失败：{exc}") from exc
    results = payload.get("results") or []
    if not isinstance(results, list):
        raise PipelineError("AnythingLLM 返回了无效的向量检索结果")
    return [item for item in results if isinstance(item, dict)]


async def _search_workspace(
    settings: Settings,
    workspace_slug: str,
    query: str,
    top_k: int,
    score_threshold: float,
    category_path: str = "",
    include_subdirectories: bool = True,
    max_chars_per_result: int = 6000,
    knowledge_tier: str = "raw",
) -> list[dict[str, object]]:
    selected_category = normalize_directory_path(category_path)
    requested = min(50, top_k * 5) if selected_category else top_k
    raw_results = await _vector_search(
        settings,
        workspace_slug,
        query,
        requested,
        score_threshold,
    )
    results: list[dict[str, object]] = []
    for item in raw_results:
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        result_category = _metadata_directory(metadata)
        if not _matches_category(result_category, selected_category, include_subdirectories):
            continue
        text = str(item.get("text") or "")
        results.append(
            {
                "id": item.get("id"),
                "title": metadata.get("title") or metadata.get("chunkSource"),
                "source_url": metadata.get("docSource") or metadata.get("url"),
                "category_path": result_category,
                "knowledge_tier": knowledge_tier,
                "score": item.get("score"),
                "distance": item.get("distance"),
                "text": text[:max_chars_per_result],
                "truncated": len(text) > max_chars_per_result,
            }
        )
        if len(results) >= top_k:
            break
    return results


def _logical_summary(
    logical: LogicalKnowledgeBase,
    fusion_manager: FusionManager,
) -> dict[str, object]:
    latest = fusion_manager.list_latest(logical.source_workspace_slug)
    return {
        "id": logical.id,
        "name": logical.name,
        "source_workspace": logical.source_workspace_name,
        "fusion_workspace": logical.fusion_workspace_name,
        "fusion_ready": bool(logical.fusion_workspace_slug),
        "fusion_topic_count": len(latest),
        "published_topic_count": sum(item.status.value == "published" for item in latest),
    }


def _resolve_logical_base(
    fusion_manager: FusionManager,
    identifier: str,
) -> LogicalKnowledgeBase:
    choices = fusion_manager.list_logical_bases()
    normalized = identifier.strip().casefold()
    if normalized:
        matched = [
            item
            for item in choices
            if normalized in {item.id.casefold(), item.name.casefold()}
        ]
        if len(matched) == 1:
            return matched[0]
        raise ValueError("逻辑知识库不存在；请先调用 list_logical_knowledge_bases")
    if len(choices) == 1:
        return choices[0]
    if not choices:
        raise ValueError("尚未创建逻辑知识库，请先在 AutoStuKnow 页面生成融合知识")
    raise ValueError("存在多个逻辑知识库，请传入 list_logical_knowledge_bases 返回的 name")


def _fusion_summary(
    record: FusionRecord,
    fusion_manager: FusionManager,
) -> dict[str, object]:
    return {
        "record_id": record.id,
        "topic_id": record.topic_id,
        "title": record.title,
        "version": record.version,
        "status": record.status.value,
        "category_path": record.category_path,
        "source_count": len(record.source_job_ids),
        "has_document": fusion_manager.document_file(record) is not None,
    }


def create_mcp_server(
    settings: Settings,
    manager: JobManager,
    fusion_manager: FusionManager,
) -> FastMCP:
    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
    server = FastMCP(
        name="AutoStuKnow Knowledge Base",
        instructions=(
            "Read-only access to knowledge stored in AnythingLLM by AutoStuKnow. "
            "Prefer search_logical_knowledge: it automatically searches curated fusion knowledge "
            "first and supplements it with raw evidence, without requiring workspace slugs. "
            "Use global_search_knowledge when the target logical knowledge base is unknown. "
            "Use get_knowledge only when a complete AutoStuKnow Markdown note is needed."
        ),
        stateless_http=True,
        json_response=True,
        streamable_http_path="/mcp",
        # The endpoint is intentionally reachable by NAS IP/reverse-proxy hostnames.
        # Bearer authentication below remains mandatory for every transport request.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @server.tool(annotations=read_only)
    async def list_logical_knowledge_bases() -> dict[str, object]:
        """List user-facing logical knowledge bases without exposing workspace routing details."""
        items = [
            _logical_summary(logical, fusion_manager)
            for logical in fusion_manager.list_logical_bases()
        ]
        return {"logical_knowledge_bases": items, "count": len(items)}

    @server.tool(annotations=read_only)
    async def list_workspaces() -> dict[str, object]:
        """List AnythingLLM workspaces and AutoStuKnow-managed knowledge counts."""
        workspaces = await list_anythingllm_workspaces(settings)
        enriched: list[dict[str, object]] = []
        for workspace in workspaces:
            slug = str(workspace["slug"])
            jobs = manager.jobs_in_workspace(slug)
            enriched.append(
                {
                    **workspace,
                    "autostuknow_knowledge_count": len(jobs),
                    "autostuknow_directory_count": len(manager.directory_paths(slug)),
                }
            )
        return {"workspaces": enriched, "count": len(enriched)}

    @server.tool(annotations=read_only)
    async def list_knowledge_tree(workspace_slug: str) -> dict[str, object]:
        """Return the multi-level directory tree and AutoStuKnow notes in one workspace."""
        normalized_slug = workspace_slug.strip()
        if not normalized_slug:
            raise ValueError("workspace_slug 不能为空")
        workspace = await get_anythingllm_workspace(settings, normalized_slug)
        return {
            "workspace": workspace,
            "tree": _knowledge_tree(manager, normalized_slug),
            "note": "目录树只包含 AutoStuKnow 管理的知识；手工上传文档仍可被向量检索。",
        }

    @server.tool(annotations=read_only)
    async def search_knowledge(
        workspace_slug: str,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.2,
        category_path: str = "",
        include_subdirectories: bool = True,
        max_chars_per_result: int = 6000,
    ) -> dict[str, object]:
        """Vector-search an AnythingLLM workspace and return relevant chunks with sources.

        category_path optionally limits AutoStuKnow-managed results to one virtual directory.
        Leave it empty to include the entire workspace, including manually uploaded documents.
        """
        normalized_slug = workspace_slug.strip()
        normalized_query = query.strip()
        if not normalized_slug:
            raise ValueError("workspace_slug 不能为空")
        if not normalized_query or len(normalized_query) > 2000:
            raise ValueError("query 长度必须为 1 到 2000 个字符")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold 必须在 0 到 1 之间")
        if not 500 <= max_chars_per_result <= 12000:
            raise ValueError("max_chars_per_result 必须在 500 到 12000 之间")
        selected_category = normalize_directory_path(category_path)

        await get_anythingllm_workspace(settings, normalized_slug)
        results = await _search_workspace(
            settings,
            normalized_slug,
            normalized_query,
            top_k,
            score_threshold,
            selected_category,
            include_subdirectories,
            max_chars_per_result,
        )

        return {
            "workspace_slug": normalized_slug,
            "query": normalized_query,
            "category_path": selected_category,
            "results": results,
            "count": len(results),
        }

    @server.tool(annotations=read_only)
    async def search_logical_knowledge(
        query: str,
        logical_knowledge_base: str = "",
        top_k: int = 6,
        score_threshold: float = 0.2,
        include_raw_evidence: bool = True,
        max_chars_per_result: int = 6000,
    ) -> dict[str, object]:
        """Search one logical KB fusion-first, then supplement with original evidence.

        logical_knowledge_base accepts the user-facing name returned by
        list_logical_knowledge_bases. It may be omitted when there is only one logical KB.
        """
        normalized_query = query.strip()
        if not normalized_query or len(normalized_query) > 2000:
            raise ValueError("query 长度必须为 1 到 2000 个字符")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1 到 20 之间")
        if not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold 必须在 0 到 1 之间")
        if not 500 <= max_chars_per_result <= 12000:
            raise ValueError("max_chars_per_result 必须在 500 到 12000 之间")
        logical = _resolve_logical_base(fusion_manager, logical_knowledge_base)
        fusion_results: list[dict[str, object]] = []
        if logical.fusion_workspace_slug:
            fusion_results = await _search_workspace(
                settings,
                logical.fusion_workspace_slug,
                normalized_query,
                top_k,
                score_threshold,
                max_chars_per_result=max_chars_per_result,
                knowledge_tier="fusion",
            )
        raw_results: list[dict[str, object]] = []
        if include_raw_evidence:
            raw_limit = max(1, top_k - len(fusion_results))
            if fusion_results:
                raw_limit = min(max(2, top_k // 2), top_k)
            raw_results = await _search_workspace(
                settings,
                logical.source_workspace_slug,
                normalized_query,
                raw_limit,
                score_threshold,
                max_chars_per_result=max_chars_per_result,
                knowledge_tier="raw_evidence",
            )
        combined = [*fusion_results, *raw_results]
        return {
            "logical_knowledge_base": logical.name,
            "query": normalized_query,
            "search_strategy": "fusion_first_then_raw_evidence",
            "results": combined,
            "fusion_count": len(fusion_results),
            "raw_evidence_count": len(raw_results),
            "count": len(combined),
            "answering_guidance": (
                "优先依据 knowledge_tier=fusion 的高层知识作答；用 raw_evidence 核验、补充"
                "细节，并在冲突时明确说明原始证据。"
            ),
        }

    @server.tool(annotations=read_only)
    async def global_search_knowledge(
        query: str,
        top_k_per_base: int = 4,
        score_threshold: float = 0.2,
        include_raw_evidence: bool = True,
    ) -> dict[str, object]:
        """Search every logical knowledge base when the relevant domain is unknown."""
        normalized_query = query.strip()
        if not normalized_query or len(normalized_query) > 2000:
            raise ValueError("query 长度必须为 1 到 2000 个字符")
        if not 1 <= top_k_per_base <= 10:
            raise ValueError("top_k_per_base 必须在 1 到 10 之间")
        logical_bases = fusion_manager.list_logical_bases()

        async def search_one(logical: LogicalKnowledgeBase) -> dict[str, object]:
            fusion_results: list[dict[str, object]] = []
            if logical.fusion_workspace_slug:
                fusion_results = await _search_workspace(
                    settings,
                    logical.fusion_workspace_slug,
                    normalized_query,
                    top_k_per_base,
                    score_threshold,
                    knowledge_tier="fusion",
                )
            raw_results: list[dict[str, object]] = []
            if include_raw_evidence:
                raw_results = await _search_workspace(
                    settings,
                    logical.source_workspace_slug,
                    normalized_query,
                    max(1, top_k_per_base - len(fusion_results)),
                    score_threshold,
                    knowledge_tier="raw_evidence",
                )
            return {
                "logical_knowledge_base": logical.name,
                "results": [*fusion_results, *raw_results],
            }

        matches = await asyncio.gather(*(search_one(item) for item in logical_bases))
        matches = [item for item in matches if item["results"]]
        return {
            "query": normalized_query,
            "search_strategy": "all_logical_bases_fusion_first",
            "matches": matches,
            "logical_base_count": len(logical_bases),
        }

    @server.tool(annotations=read_only)
    async def get_fusion_knowledge(
        record_id: str,
        max_chars: int = 50000,
    ) -> dict[str, object]:
        """Read a complete fusion Markdown draft or published version by record_id."""
        if not 1000 <= max_chars <= 100000:
            raise ValueError("max_chars 必须在 1000 到 100000 之间")
        record = fusion_manager.get(record_id.strip())
        if record is None:
            raise ValueError("融合知识不存在")
        document = fusion_manager.document_file(record)
        if document is None:
            raise ValueError("融合知识文档尚未生成或已不存在")
        content = document.read_text(encoding="utf-8", errors="replace")
        return {
            **_fusion_summary(record, fusion_manager),
            "markdown": content[:max_chars],
            "truncated": len(content) > max_chars,
            "total_chars": len(content),
        }

    @server.tool(annotations=read_only)
    async def get_knowledge(job_id: str, max_chars: int = 50000) -> dict[str, object]:
        """Read one complete AutoStuKnow Markdown note by the job_id from the tree."""
        normalized_job_id = job_id.strip()
        if not normalized_job_id:
            raise ValueError("job_id 不能为空")
        if not 1000 <= max_chars <= 100000:
            raise ValueError("max_chars 必须在 1000 到 100000 之间")
        job = manager.get(normalized_job_id)
        if job is None:
            raise ValueError("知识不存在")
        document_path = manager.document_file(job)
        if document_path is None:
            raise ValueError("知识文档尚未生成或已不存在")
        try:
            content = document_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(f"读取知识文档失败：{exc}") from exc
        return {
            **_job_summary(job, manager),
            "workspace_slug": manager.effective_workspace_slug(job),
            "markdown": content[:max_chars],
            "truncated": len(content) > max_chars,
            "total_chars": len(content),
        }

    return server
