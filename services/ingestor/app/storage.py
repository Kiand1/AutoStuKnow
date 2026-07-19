import json
import os
from pathlib import Path

from .models import JobRecord


class JobStorage:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.jobs_dir = data_dir / "jobs"
        self.cache_dir = data_dir / "cache"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(self, job: JobRecord) -> None:
        path = self.job_dir(job.id) / "job.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(job.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def load_all(self) -> dict[str, JobRecord]:
        jobs: dict[str, JobRecord] = {}
        if not self.jobs_dir.exists():
            return jobs
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            jobs[job.id] = job
        return jobs
