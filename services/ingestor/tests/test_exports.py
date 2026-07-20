import json
import zipfile
from pathlib import Path

from app.catalog import directory_ancestors
from app.exports import MAX_ARCHIVE_PATH_UTF8_BYTES, build_knowledge_archive
from app.models import JobRecord, JobStatus


def test_archive_paths_are_short_enough_for_windows_explorer(tmp_path: Path) -> None:
    category_path = "/".join(
        f"第{index}层-这是一个很长的中文目录名称" for index in range(1, 13)
    )
    job = JobRecord(
        id="windows-compatible-export",
        url="https://youtu.be/rrrrrrrrrrr",
        canonical_url="https://www.youtube.com/watch?v=rrrrrrrrrrr",
        workspace_slug="research",
        category_path=category_path,
        status=JobStatus.completed,
        stage="completed",
        title="【超长中文标题】" + "双均线交易系统风险管理" * 30,
        document_path="jobs/windows-compatible-export/document.md",
    )
    document = tmp_path / "document.md"
    document.write_text("# 内容\n", encoding="utf-8", newline="\n")

    archive_path = build_knowledge_archive(
        tmp_path,
        "very-long-workspace-name-" * 5,
        [(job, document)],
        directory_ancestors(category_path),
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.testzip() is None
            assert all(
                len(name.encode("utf-8")) <= MAX_ARCHIVE_PATH_UTF8_BYTES
                for name in archive.namelist()
            )
            markdown = [name for name in archive.namelist() if name.endswith(".md")]
            assert len(markdown) == 1
            assert "__windows-comp.md" in markdown[0]
            manifest_name = next(
                name for name in archive.namelist() if name.endswith(".json")
            )
            manifest = json.loads(archive.read(manifest_name))
            assert manifest["documents"][0]["archive_path"] == markdown[0]
    finally:
        archive_path.unlink(missing_ok=True)
