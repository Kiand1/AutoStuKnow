import re
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException, status

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,20}$")


def canonicalize_youtube_url(raw_url: str) -> str:
    """Accept one YouTube video URL and return a stable watch URL."""
    try:
        parsed = urlparse(raw_url.strip())
    except ValueError as exc:
        raise _invalid_url("URL 无法解析") from exc

    if parsed.scheme not in {"http", "https"}:
        raise _invalid_url("只允许 http/https YouTube URL")
    if parsed.username or parsed.password:
        raise _invalid_url("URL 不能包含用户名或密码")

    host = (parsed.hostname or "").rstrip(".").lower()
    allowed = (
        host == "youtu.be"
        or host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtube-nocookie.com"
        or host.endswith(".youtube-nocookie.com")
    )
    if not allowed:
        raise _invalid_url("当前 V1 只允许单个 YouTube 视频")

    video_id: str | None = None
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be" and path_parts:
        video_id = path_parts[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [None])[0]
    elif path_parts and path_parts[0] in {"shorts", "embed", "live"} and len(path_parts) >= 2:
        video_id = path_parts[1]

    if not video_id or not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise _invalid_url("请提交 watch、shorts、live、embed 或 youtu.be 的单视频链接")

    return f"https://www.youtube.com/watch?v={video_id}"


def _invalid_url(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=message)
