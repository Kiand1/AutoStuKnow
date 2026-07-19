import os

os.environ.setdefault("INGESTOR_API_KEY", "test-key-that-is-at-least-24-characters")
os.environ.setdefault("WEB_UI_USERNAME", "admin")
os.environ.setdefault("WEB_UI_PASSWORD", "test-web-password-123456789")
os.environ.setdefault(
    "WEB_UI_SESSION_SECRET",
    "test-web-session-secret-that-is-at-least-32-characters",
)
