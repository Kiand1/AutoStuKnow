import json
import os
import threading
from pathlib import Path

MAX_DIRECTORY_DEPTH = 12
MAX_DIRECTORY_SEGMENT_LENGTH = 80
MAX_DIRECTORY_PATH_LENGTH = 512


def normalize_directory_path(value: str | None) -> str:
    """Return a stable, slash-separated directory path or the workspace root."""
    raw = (value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    segments = [segment.strip() for segment in raw.split("/") if segment.strip()]
    if len(segments) > MAX_DIRECTORY_DEPTH:
        raise ValueError(f"目录最多支持 {MAX_DIRECTORY_DEPTH} 层")
    for segment in segments:
        if segment in {".", ".."}:
            raise ValueError("目录名称不能是 . 或 ..")
        if len(segment) > MAX_DIRECTORY_SEGMENT_LENGTH:
            raise ValueError(f"单级目录名称不能超过 {MAX_DIRECTORY_SEGMENT_LENGTH} 个字符")
        if any(ord(character) < 32 or ord(character) == 127 for character in segment):
            raise ValueError("目录名称不能包含控制字符")
    normalized = "/".join(segments)
    if len(normalized) > MAX_DIRECTORY_PATH_LENGTH:
        raise ValueError(f"完整目录路径不能超过 {MAX_DIRECTORY_PATH_LENGTH} 个字符")
    return normalized


def directory_ancestors(path: str) -> list[str]:
    segments = normalize_directory_path(path).split("/")
    if segments == [""]:
        return []
    return ["/".join(segments[:index]) for index in range(1, len(segments) + 1)]


def path_is_within(candidate: str, directory: str) -> bool:
    normalized_candidate = normalize_directory_path(candidate)
    normalized_directory = normalize_directory_path(directory)
    return normalized_candidate == normalized_directory or normalized_candidate.startswith(
        f"{normalized_directory}/"
    )


def directory_sort_key(path: str) -> tuple[str, ...]:
    return tuple(segment.casefold() for segment in path.split("/"))


class DirectoryCatalog:
    """Persist user-created virtual directories independently from AnythingLLM."""

    def __init__(self, data_dir: Path):
        self.path = data_dir / "catalog.json"
        self._lock = threading.RLock()
        self._directories = self._load()

    def _load(self) -> dict[str, set[str]]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        workspaces = payload.get("workspaces", {}) if isinstance(payload, dict) else {}
        result: dict[str, set[str]] = {}
        if not isinstance(workspaces, dict):
            return result
        for workspace_slug, values in workspaces.items():
            if not isinstance(workspace_slug, str) or not isinstance(values, list):
                continue
            normalized: set[str] = set()
            for value in values:
                if not isinstance(value, str):
                    continue
                try:
                    path = normalize_directory_path(value)
                except ValueError:
                    continue
                if path:
                    normalized.update(directory_ancestors(path))
            if normalized:
                result[workspace_slug] = normalized
        return result

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        payload = {
            "version": 1,
            "workspaces": {
                slug: sorted(paths, key=directory_sort_key)
                for slug, paths in sorted(self._directories.items())
                if paths
            },
        }
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def list_paths(self, workspace_slug: str) -> list[str]:
        with self._lock:
            return sorted(self._directories.get(workspace_slug, set()), key=directory_sort_key)

    def create(self, workspace_slug: str, path: str) -> str:
        normalized = normalize_directory_path(path)
        if not normalized:
            raise ValueError("目录路径不能为空")
        with self._lock:
            self._directories.setdefault(workspace_slug, set()).update(
                directory_ancestors(normalized)
            )
            self._save()
        return normalized

    def register(self, workspace_slug: str | None, path: str | None) -> None:
        if not workspace_slug:
            return
        normalized = normalize_directory_path(path)
        if not normalized:
            return
        with self._lock:
            paths = self._directories.setdefault(workspace_slug, set())
            before = len(paths)
            paths.update(directory_ancestors(normalized))
            if len(paths) != before:
                self._save()

    def delete(self, workspace_slug: str, path: str) -> list[str]:
        normalized = normalize_directory_path(path)
        if not normalized:
            raise ValueError("不能删除知识库根目录")
        with self._lock:
            paths = self._directories.get(workspace_slug, set())
            removed = sorted(value for value in paths if path_is_within(value, normalized))
            paths.difference_update(removed)
            if paths:
                self._directories[workspace_slug] = paths
            else:
                self._directories.pop(workspace_slug, None)
            self._save()
        return removed

    def delete_workspace(self, workspace_slug: str) -> list[str]:
        with self._lock:
            removed = sorted(
                self._directories.pop(workspace_slug, set()),
                key=directory_sort_key,
            )
            self._save()
        return removed
