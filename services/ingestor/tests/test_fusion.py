import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.config import Settings
from app.fusion import FusionManager
from app.models import (
    FusionContent,
    FusionScope,
    FusionSourceExtract,
    FusionStatus,
    JobRecord,
    JobStatus,
    WebFusionGenerateRequest,
)
from app.pipeline import JobManager


def build_managers(tmp_path: Path) -> tuple[JobManager, FusionManager, list[JobRecord]]:
    settings = Settings(
        data_dir=tmp_path,
        ingestor_api_key="test-key-that-is-at-least-24-characters",
        web_ui_username="admin",
        web_ui_password="test-web-password-123456789",
        web_ui_session_secret="test-web-session-secret-that-is-at-least-32-characters",
        llm_base_url="http://deepseek.test/v1",
        llm_api_key="deepseek-test-key",
        llm_model="deepseek-v4-pro",
        anythingllm_api_key="anythingllm-test-key",
        anythingllm_auto_sync=False,
    )
    jobs = JobManager(settings)
    records: list[JobRecord] = []
    for index in range(2):
        job_id = f"source-{index}"
        record = JobRecord(
            id=job_id,
            url=f"https://youtu.be/source0000{index}",
            canonical_url=f"https://www.youtube.com/watch?v=source0000{index}",
            workspace_slug="investment",
            category_path="虚拟币/双均线",
            status=JobStatus.completed,
            stage="completed",
            title=f"原始视频 {index}",
            document_path=f"jobs/{job_id}/document.md",
        )
        jobs.jobs[job_id] = record
        jobs.storage.save(record)
        (jobs.storage.job_dir(job_id) / "document.md").write_text(
            f"# 原始视频 {index}\n\n## 完整转录\n\n原始内容 {index}\n",
            encoding="utf-8",
        )
        records.append(record)
    return jobs, FusionManager(settings, jobs), records


async def wait_for_fusion(manager: FusionManager, record_id: str) -> None:
    for _ in range(100):
        record = manager.get(record_id)
        if record and record.status in {FusionStatus.draft, FusionStatus.failed}:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("fusion generation did not finish")


@pytest.mark.asyncio
async def test_fusion_draft_publish_and_versioning_preserve_raw_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, manager, source_jobs = build_managers(tmp_path)
    original_documents = {
        job.id: jobs.document_file(job).read_text(encoding="utf-8")  # type: ignore[union-attr]
        for job in source_jobs
    }
    monkeypatch.setattr(
        "app.fusion.get_anythingllm_workspace",
        AsyncMock(return_value={"id": 1, "name": "投资", "slug": "investment"}),
    )
    monkeypatch.setattr(
        "app.fusion.extract_source",
        AsyncMock(
            side_effect=lambda _settings, job, _document: FusionSourceExtract(
                job_id=job.id,
                title=job.title,
                source_url=job.canonical_url,
                summary=f"{job.title} 摘要",
                principles=["顺势交易"],
                risk_controls=["设置止损"],
            )
        ),
    )
    monkeypatch.setattr(
        "app.fusion.synthesize_extracts",
        AsyncMock(
            return_value=FusionContent(
                executive_summary="双均线策略的融合结论。",
                core_principles=["趋势明确后参与"],
                risk_controls=["单笔风险受限"],
                consensus=["必须设置止损"],
            )
        ),
    )
    request = WebFusionGenerateRequest(
        source_workspace_slug="investment",
        title="双均线交易系统",
        category_path="交易系统/趋势",
        scope=FusionScope.selected,
        selected_job_ids=[job.id for job in source_jobs],
    )

    first = await manager.generate(request)
    await wait_for_fusion(manager, first.id)
    first = manager.get(first.id)  # type: ignore[assignment]
    assert first is not None
    assert first.status == FusionStatus.draft
    draft = manager.document_file(first)
    assert draft is not None
    content = draft.read_text(encoding="utf-8")
    assert "文档类型：融合知识" in content
    assert "双均线策略的融合结论" in content
    assert all(job.id in content for job in source_jobs)

    ensure_workspace = AsyncMock(return_value="investment-fusion")
    upload = AsyncMock(
        side_effect=["custom-documents/fusion-v1.json", "custom-documents/fusion-v2.json"]
    )
    delete = AsyncMock()
    monkeypatch.setattr(manager, "ensure_fusion_workspace", ensure_workspace)
    monkeypatch.setattr("app.fusion.upload_document_to_anythingllm", upload)
    monkeypatch.setattr("app.fusion.delete_anythingllm_documents", delete)

    first = await manager.publish(first.id, first.title)
    assert first.status == FusionStatus.published
    assert first.fusion_workspace_slug == "investment-fusion"

    update_request = request.model_copy(update={"topic_id": first.topic_id})
    second = await manager.generate(update_request)
    await wait_for_fusion(manager, second.id)
    second = manager.get(second.id)  # type: ignore[assignment]
    assert second is not None and second.version == 2
    second = await manager.publish(second.id, second.title)

    assert second.status == FusionStatus.published
    assert manager.get(first.id).status == FusionStatus.superseded  # type: ignore[union-attr]
    delete.assert_awaited_once_with(
        manager.settings,
        "investment-fusion",
        ["custom-documents/fusion-v1.json"],
    )
    for job in source_jobs:
        document = jobs.document_file(job)
        assert document is not None
        assert document.read_text(encoding="utf-8") == original_documents[job.id]


@pytest.mark.asyncio
async def test_fusion_requires_two_valid_source_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manager, source_jobs = build_managers(tmp_path)
    monkeypatch.setattr(
        "app.fusion.get_anythingllm_workspace",
        AsyncMock(return_value={"id": 1, "name": "投资", "slug": "investment"}),
    )
    request = WebFusionGenerateRequest(
        source_workspace_slug="investment",
        title="只有一个来源",
        scope=FusionScope.selected,
        selected_job_ids=[source_jobs[0].id],
    )
    with pytest.raises(Exception, match="至少需要 2 条"):
        await manager.generate(request)
