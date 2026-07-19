import pytest
from fastapi import HTTPException

from app.urls import canonicalize_youtube_url


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=ignored",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        (
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        (
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        (
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        ),
    ],
)
def test_canonicalize_supported_urls(source: str, expected: str) -> None:
    assert canonicalize_youtube_url(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "file:///etc/passwd",
        "https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/playlist?list=PL123",
    ],
)
def test_rejects_non_video_or_untrusted_urls(source: str) -> None:
    with pytest.raises(HTTPException):
        canonicalize_youtube_url(source)
