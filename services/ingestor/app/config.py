from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("/data")
    ingestor_api_key: str = Field(min_length=24)
    mcp_api_key: str = ""
    mcp_enabled: bool = False
    web_ui_username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    web_ui_password: str = Field(min_length=12, max_length=256)
    web_ui_session_secret: str = Field(min_length=32)
    web_ui_session_ttl_hours: int = Field(default=24, ge=1, le=720)
    web_ui_secure_cookie: bool = False

    whisper_base_url: str = "http://whisper:9000"
    whisper_api_key: str = ""
    whisper_language: str = "auto"

    max_video_duration_minutes: int = Field(default=180, ge=1, le=1_440)
    max_download_size_mb: int = Field(default=1_024, ge=10, le=102_400)
    max_concurrent_jobs: int = Field(default=1, ge=1, le=8)
    keep_audio: bool = False
    ytdlp_cookies_file: str = ""
    prefer_youtube_subtitles: bool = True
    allow_automatic_subtitles: bool = True

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_thinking_mode: str = Field(default="", pattern=r"^(enabled|disabled)?$")
    llm_json_mode: bool = False
    summary_language: str = "zh-CN"
    llm_chunk_chars: int = Field(default=12_000, ge=2_000, le=100_000)
    llm_max_tokens: int = Field(default=1_800, ge=256, le=16_384)
    fusion_llm_max_tokens: int = Field(default=4_000, ge=512, le=32_768)

    anythingllm_base_url: str = "http://anythingllm:3001/api"
    anythingllm_api_key: str = ""
    anythingllm_workspace_slug: str = ""
    anythingllm_auto_sync: bool = True
    anythingllm_sync_timeout_seconds: int = Field(default=1_800, ge=30, le=7_200)

    @field_validator("ingestor_api_key")
    @classmethod
    def reject_placeholder_key(cls, value: str) -> str:
        if value.startswith("replace-with-"):
            raise ValueError("INGESTOR_API_KEY is still a placeholder; generate a random key")
        return value

    @field_validator("mcp_api_key")
    @classmethod
    def validate_mcp_api_key(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return ""
        if normalized.startswith("replace-with-"):
            raise ValueError("MCP_API_KEY is still a placeholder; run scripts/init-nas.sh")
        if len(normalized) < 24:
            raise ValueError("MCP_API_KEY must contain at least 24 characters")
        return normalized

    @field_validator("web_ui_password", "web_ui_session_secret")
    @classmethod
    def reject_web_placeholders(cls, value: str) -> str:
        if value.startswith("replace-with-"):
            raise ValueError("Web UI credentials are still placeholders; run scripts/init-nas.sh")
        return value

    @property
    def max_video_duration_seconds(self) -> int:
        return self.max_video_duration_minutes * 60

    @property
    def max_download_size_bytes(self) -> int:
        return self.max_download_size_mb * 1024 * 1024

    @property
    def summarizer_enabled(self) -> bool:
        return bool(self.llm_base_url.strip() and self.llm_model.strip())

    @property
    def anythingllm_sync_enabled(self) -> bool:
        return bool(self.anythingllm_auto_sync and self.anythingllm_api_key.strip())

    @property
    def effective_mcp_api_key(self) -> str:
        """Keep upgraded deployments working until init-nas generates a dedicated key."""
        return self.mcp_api_key or self.ingestor_api_key
