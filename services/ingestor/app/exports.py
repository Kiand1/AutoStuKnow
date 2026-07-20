import hashlib
import json
import os
import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from .catalog import normalize_directory_path, path_is_within
from .models import JobRecord

_INVALID_EXPORT_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
MAX_ARCHIVE_PATH_UTF8_BYTES = 220
MAX_EXPORT_SEGMENT_UTF8_BYTES = 80
MAX_ARCHIVE_DIRECTORY_SEGMENT_UTF8_BYTES = 60
MAX_KNOWLEDGE_TITLE_UTF8_BYTES = 120
MIN_ARCHIVE_FILENAME_UTF8_BYTES = 40


def truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    digest = hashlib.sha1(encoded).hexdigest()[:6]
    suffix = f"~{digest}"
    head_bytes = max_bytes - len(suffix)
    if head_bytes <= 0:
        return digest[:max_bytes]
    head = encoded[:head_bytes].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{head}{suffix}" if head else digest[:max_bytes]


def safe_export_segment(
    value: str | None,
    fallback: str,
    max_bytes: int = MAX_EXPORT_SEGMENT_UTF8_BYTES,
) -> str:
    cleaned = _INVALID_EXPORT_CHARACTERS.sub("_", (value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback
    if cleaned.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    cleaned = cleaned[:100].rstrip(" .") or fallback
    return truncate_utf8(cleaned, max_bytes)


def knowledge_filename(
    job: JobRecord,
    max_title_bytes: int = MAX_KNOWLEDGE_TITLE_UTF8_BYTES,
) -> str:
    title = safe_export_segment(job.title, "未命名知识", max_bytes=max_title_bytes)
    identifier = safe_export_segment(job.id[:12], "knowledge")
    return f"{title}__{identifier}.md"


def safe_archive_directory(root_name: str, directory_path: str) -> str:
    normalized = normalize_directory_path(directory_path)
    if not normalized:
        return ""
    raw_segments = normalized.split("/")
    separators = len(raw_segments) - 1
    available = (
        MAX_ARCHIVE_PATH_UTF8_BYTES
        - len(root_name.encode("utf-8"))
        - 2
        - MIN_ARCHIVE_FILENAME_UTF8_BYTES
        - separators
    )
    segment_budget = max(
        8,
        min(
            MAX_ARCHIVE_DIRECTORY_SEGMENT_UTF8_BYTES,
            available // len(raw_segments),
        ),
    )
    return "/".join(
        safe_export_segment(segment, "directory", max_bytes=segment_budget)
        for segment in raw_segments
    )


def archive_knowledge_filename(job: JobRecord, archive_prefix: str) -> str:
    identifier = safe_export_segment(job.id[:12], "knowledge")
    suffix = f"__{identifier}.md"
    title_budget = (
        MAX_ARCHIVE_PATH_UTF8_BYTES
        - len(archive_prefix.encode("utf-8"))
        - len(suffix.encode("utf-8"))
    )
    return knowledge_filename(job, max_title_bytes=max(8, title_budget))


def archive_download_name(workspace_slug: str, selected_path: str = "") -> str:
    workspace = safe_export_segment(workspace_slug, "knowledge-base")
    if not selected_path:
        return f"{workspace}-知识库.zip"
    directory = safe_export_segment(selected_path.split("/")[-1], "directory")
    return f"{workspace}-{directory}.zip"


def build_knowledge_archive(
    cache_dir: Path,
    workspace_slug: str,
    jobs: list[tuple[JobRecord, Path]],
    directories: list[str],
    selected_path: str = "",
) -> Path:
    normalized_scope = normalize_directory_path(selected_path)
    root_name = safe_export_segment(workspace_slug, "knowledge-base", max_bytes=60)
    scoped_jobs = [
        (job, document)
        for job, document in jobs
        if not normalized_scope or path_is_within(job.category_path, normalized_scope)
    ]
    scoped_directories = [
        path
        for path in directories
        if not normalized_scope or path_is_within(path, normalized_scope)
    ]

    cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".knowledge-export-",
        suffix=".zip",
        dir=cache_dir,
    )
    os.close(descriptor)
    archive_path = Path(temporary_name)
    manifest_documents: list[dict[str, object]] = []
    used_names: set[str] = set()
    used_directories: set[str] = set()

    try:
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.writestr(f"{root_name}/", "")
            used_directories.add(f"{root_name}/".casefold())
            for directory in scoped_directories:
                safe_directory = safe_archive_directory(root_name, directory)
                archive_directory = f"{root_name}/{safe_directory}/"
                if archive_directory.casefold() not in used_directories:
                    archive.writestr(archive_directory, "")
                    used_directories.add(archive_directory.casefold())

            for job, document_path in scoped_jobs:
                safe_directory = safe_archive_directory(root_name, job.category_path)
                archive_prefix = (
                    f"{root_name}/{safe_directory}/" if safe_directory else f"{root_name}/"
                )
                base_name = archive_knowledge_filename(job, archive_prefix)
                relative_name = f"{safe_directory}/{base_name}" if safe_directory else base_name
                archive_name = f"{root_name}/{relative_name}"
                duplicate_index = 2
                while archive_name.casefold() in used_names:
                    stem = Path(base_name).stem
                    relative_name = (
                        f"{safe_directory}/{stem}-{duplicate_index}.md"
                        if safe_directory
                        else f"{stem}-{duplicate_index}.md"
                    )
                    archive_name = f"{root_name}/{relative_name}"
                    duplicate_index += 1
                used_names.add(archive_name.casefold())
                archive.write(document_path, archive_name)
                manifest_documents.append(
                    {
                        "id": job.id,
                        "title": job.title or job.canonical_url,
                        "category_path": job.category_path,
                        "source_url": job.canonical_url,
                        "archive_path": archive_name,
                        "size_bytes": document_path.stat().st_size,
                        "updated_at": job.updated_at.isoformat(),
                    }
                )

            manifest = {
                "version": 1,
                "workspace_slug": workspace_slug,
                "scope": normalized_scope or "root",
                "exported_at": datetime.now(UTC).isoformat(),
                "directories": scoped_directories,
                "documents": manifest_documents,
            }
            archive.writestr(
                f"{root_name}/知识库信息.json",
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path
