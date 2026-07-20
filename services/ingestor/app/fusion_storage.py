import json
import os
import shutil
import threading
from pathlib import Path

from .models import FusionRecord, LogicalKnowledgeBase


class FusionStorage:
    """Atomic local storage for logical knowledge bases and fusion versions."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.root = data_dir / "fusions"
        self.logical_path = self.root / "logical-knowledge-bases.json"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def load_logical_bases(self) -> dict[str, LogicalKnowledgeBase]:
        with self._lock:
            if not self.logical_path.is_file():
                return {}
            try:
                payload = json.loads(self.logical_path.read_text(encoding="utf-8"))
                items = payload.get("items", []) if isinstance(payload, dict) else []
                records = [LogicalKnowledgeBase.model_validate(item) for item in items]
            except (OSError, ValueError, TypeError):
                return {}
            return {record.id: record for record in records}

    def save_logical_bases(self, records: dict[str, LogicalKnowledgeBase]) -> None:
        with self._lock:
            self._write_json(
                self.logical_path,
                {
                    "version": 1,
                    "items": [
                        item.model_dump(mode="json")
                        for item in sorted(
                            records.values(),
                            key=lambda value: value.name.casefold(),
                        )
                    ],
                },
            )

    def version_dir(self, record: FusionRecord) -> Path:
        return self.root / record.topic_id / f"v{record.version}"

    def save_record(self, record: FusionRecord) -> None:
        with self._lock:
            self._write_json(
                self.version_dir(record) / "fusion.json",
                record.model_dump(mode="json"),
            )

    def load_records(self) -> dict[str, FusionRecord]:
        records: dict[str, FusionRecord] = {}
        with self._lock:
            for path in self.root.glob("*/v*/fusion.json"):
                try:
                    record = FusionRecord.model_validate_json(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                records[record.id] = record
        return records

    def delete_topic(self, topic_id: str) -> None:
        with self._lock:
            target = (self.root / topic_id).resolve()
            if self.root.resolve() not in target.parents:
                raise ValueError("无效的融合知识 ID")
            if target.is_dir():
                shutil.rmtree(target)
