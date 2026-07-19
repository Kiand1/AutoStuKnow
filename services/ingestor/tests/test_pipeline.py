import asyncio
from pathlib import Path

import pytest

import app.pipeline as pipeline
from app.config import Settings
from app.models import (
    JobRecord,
    JobRequest,
    JobStatus,
    Summary,
    SyncStatus,
    Transcript,
    VideoMetadata,
)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": tmp_path,
        "ingestor_api_key": "test-key-that-is-at-least-24-characters",
        "web_ui_username": "admin",
        "web_ui_password": "test-web-password-123456789",
        "web_ui_session_secret": "test-web-session-secret-that-is-at-least-32-characters",
        "anythingllm_auto_sync": False,
        "llm_base_url": "",
        "llm_model": "",
    }
    values.update(overrides)
    return Settings(**values)


def test_render_markdown_contains_timestamps_and_source() -> None:
    video = VideoMetadata(
        id="dQw4w9WgXcQ",
        title="Test video",
        webpage_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        uploader="Tester",
        duration=65,
    )
    transcript = Transcript(
        text="hello world",
        language="en",
        segments=[{"start": 3.5, "text": "hello"}, {"start": 64, "text": "world"}],
    )
    summary = Summary(summary="A summary", key_points=["One"], tags=["demo"])

    result = pipeline.render_markdown(video, transcript, summary)

    assert "# Test video" in result
    assert "[00:00:03] hello" in result
    assert "[00:01:04] world" in result
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in result


@pytest.mark.asyncio
async def test_job_pipeline_completes_without_llm_or_anythingllm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_inspect(
        settings: Settings,
        url: str,
        job_dir: Path,
        cache_dir: Path,
        requested_language: str,
    ) -> tuple[VideoMetadata, Transcript | None, list[str]]:
        return (
            VideoMetadata(
                id="dQw4w9WgXcQ",
                title="Test video",
                webpage_url=url,
                uploader="Tester",
                duration=60,
            ),
            None,
            [],
        )

    def fake_download(
        settings: Settings, url: str, job_dir: Path, cache_dir: Path
    ) -> tuple[VideoMetadata, Path]:
        audio = job_dir / "source.m4a"
        audio.write_bytes(b"fake audio")
        return (
            VideoMetadata(
                id="dQw4w9WgXcQ",
                title="Test video",
                webpage_url=url,
                uploader="Tester",
                duration=60,
            ),
            audio,
        )

    async def fake_transcribe(
        settings: Settings, audio_path: Path, requested_language: str
    ) -> Transcript:
        return Transcript(text="This is the transcript.", language="en")

    monkeypatch.setattr(pipeline, "inspect_video_for_subtitles", fake_inspect)
    monkeypatch.setattr(pipeline, "download_audio", fake_download)
    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)

    manager = pipeline.JobManager(make_settings(tmp_path))
    job, deduplicated = await manager.submit(JobRequest(url="https://youtu.be/dQw4w9WgXcQ"))
    assert not deduplicated

    for _ in range(100):
        current = manager.get(job.id)
        if current and current.status in {JobStatus.completed, JobStatus.failed}:
            break
        await asyncio.sleep(0.01)

    completed = manager.get(job.id)
    assert completed is not None
    assert completed.status == JobStatus.completed
    assert completed.document_path is not None
    assert (tmp_path / completed.document_path).is_file()
    assert any("未配置总结模型" in warning for warning in completed.warnings)

    same_job, second_was_deduplicated = await manager.submit(
        JobRequest(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    )
    assert second_was_deduplicated
    assert same_job.id == completed.id


@pytest.mark.asyncio
async def test_job_pipeline_uses_youtube_subtitle_without_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_inspect(
        settings: Settings,
        url: str,
        job_dir: Path,
        cache_dir: Path,
        requested_language: str,
    ) -> tuple[VideoMetadata, Transcript | None, list[str]]:
        return (
            VideoMetadata(
                id="dQw4w9WgXcQ",
                title="Subtitle video",
                webpage_url=url,
                uploader="Tester",
                duration=20,
            ),
            Transcript(
                text="Subtitle text",
                language="en",
                segments=[{"start": 0, "end": 2, "text": "Subtitle text"}],
                source="youtube_manual",
            ),
            [],
        )

    def should_not_download(*_: object, **__: object) -> None:
        raise AssertionError("audio must not be downloaded when subtitles are available")

    async def should_not_transcribe(*_: object, **__: object) -> None:
        raise AssertionError("Whisper must not run when subtitles are available")

    monkeypatch.setattr(pipeline, "inspect_video_for_subtitles", fake_inspect)
    monkeypatch.setattr(pipeline, "download_audio", should_not_download)
    monkeypatch.setattr(pipeline, "transcribe_audio", should_not_transcribe)

    manager = pipeline.JobManager(make_settings(tmp_path))
    job, _ = await manager.submit(JobRequest(url="https://youtu.be/dQw4w9WgXcQ"))
    for _ in range(100):
        current = manager.get(job.id)
        if current and current.status in {JobStatus.completed, JobStatus.failed}:
            break
        await asyncio.sleep(0.01)

    completed = manager.get(job.id)
    assert completed is not None
    assert completed.status == JobStatus.completed
    assert completed.transcript_source == "youtube_manual"
    document = (tmp_path / str(completed.document_path)).read_text(encoding="utf-8")
    assert "YouTube 人工字幕" in document


def test_subtitle_selection_prefers_manual_then_requested_automatic() -> None:
    info = {
        "language": "en",
        "subtitles": {
            "en": [{"ext": "json3", "url": "https://caption/manual-en"}],
            "de": [{"ext": "json3", "url": "https://caption/manual-de"}],
        },
        "automatic_captions": {
            "en": [{"ext": "json3", "url": "https://caption/auto-en"}],
            "zh-Hans": [{"ext": "json3", "url": "https://caption/auto-zh?tlang=zh-Hans"}],
        },
    }

    automatic = pipeline.select_youtube_subtitle(info, "zh", allow_automatic=True)
    manual = pipeline.select_youtube_subtitle(info, "auto", allow_automatic=True)
    unavailable = pipeline.select_youtube_subtitle(info, "zh", allow_automatic=False)

    assert automatic is not None
    assert automatic.source == "youtube_auto"
    assert automatic.language == "zh-Hans"
    assert manual is not None
    assert manual.source == "youtube_manual"
    assert manual.language == "en"
    assert unavailable is None


def test_json3_and_vtt_subtitles_are_parsed_with_timestamps() -> None:
    json3 = {
        "events": [
            {"tStartMs": 1000, "dDurationMs": 1500, "segs": [{"utf8": "Hello"}]},
            {"tStartMs": 3000, "dDurationMs": 1000, "segs": [{"utf8": "world"}]},
        ]
    }
    vtt = """WEBVTT

00:00:01.000 --> 00:00:02.500
你好

00:00:03.000 --> 00:00:04.000
世界
"""

    json_segments = pipeline.parse_json3_segments(json3)
    vtt_segments = pipeline.parse_vtt_segments(vtt)

    assert json_segments == [
        {"start": 1.0, "end": 2.5, "text": "Hello"},
        {"start": 3.0, "end": 4.0, "text": "world"},
    ]
    assert vtt_segments == [
        {"start": 1.0, "end": 2.5, "text": "你好"},
        {"start": 3.0, "end": 4.0, "text": "世界"},
    ]


def test_split_text_preserves_content_order() -> None:
    original = "第一段。" + "a" * 30 + "\n第二段。" + "b" * 30
    chunks = pipeline.split_text(original, 20)
    assert len(chunks) > 1
    assert "".join(chunks).replace("\n", "") == original.replace("\n", "")


def test_chat_completions_url() -> None:
    assert pipeline.chat_completions_url("http://ollama:11434/v1/") == (
        "http://ollama:11434/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_summary_model_adds_optional_deepseek_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"summary":"ok","key_points":[],"tags":[],"questions":[]}'
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, _: str, **kwargs: object) -> FakeResponse:
            captured.update(kwargs["json"])  # type: ignore[arg-type]
            return FakeResponse()

    monkeypatch.setattr(pipeline.httpx, "AsyncClient", FakeClient)
    settings = make_settings(
        tmp_path,
        llm_base_url="https://api.deepseek.com",
        llm_api_key="test-key",
        llm_model="deepseek-v4-flash",
        llm_thinking_mode="disabled",
        llm_json_mode=True,
    )

    summary = await pipeline.call_summary_model(settings, "content", "summarize")

    assert summary.summary == "ok"
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["response_format"] == {"type": "json_object"}


def test_workspace_contains_document() -> None:
    payload = {
        "workspace": [
            {
                "slug": "autostuknow",
                "documents": [{"docpath": "custom-documents/video.json"}],
            }
        ]
    }

    assert pipeline.workspace_contains_document(
        payload, "autostuknow", "custom-documents/video.json"
    )
    assert not pipeline.workspace_contains_document(
        payload, "autostuknow", "custom-documents/missing.json"
    )


@pytest.mark.asyncio
async def test_successful_retry_clears_stale_anythingllm_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(
        tmp_path,
        anythingllm_api_key="anythingllm-test-key",
        anythingllm_workspace_slug="autostuknow",
    )
    manager = pipeline.JobManager(settings)
    job = JobRecord(
        id="retry-job",
        url="https://youtu.be/dQw4w9WgXcQ",
        canonical_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        status=JobStatus.completed,
        stage="completed",
        document_path="jobs/retry-job/document.md",
        sync_status=SyncStatus.failed,
        warnings=[
            "AnythingLLM 同步失败：timeout",
            "AnythingLLM 自动同步未配置；配置后重试。",
            "AI 总结失败：保留转录",
        ],
    )
    manager.jobs[job.id] = job
    document = manager.storage.job_dir(job.id) / "document.md"
    document.write_text("# test", encoding="utf-8")

    async def fake_upload(**_: object) -> str:
        return "custom-documents/document.json"

    monkeypatch.setattr(pipeline, "upload_to_anythingllm", fake_upload)

    synced = await manager.sync(job.id)

    assert synced.sync_status == SyncStatus.synced
    assert synced.warnings == ["AI 总结失败：保留转录"]
