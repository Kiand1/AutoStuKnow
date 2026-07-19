import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque

SESSION_COOKIE = "autostuknow_session"


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class WebSessionSigner:
    def __init__(self, secret: str, ttl_seconds: int) -> None:
        self._secret = secret.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def create(self, username: str, now: int | None = None) -> str:
        issued_at = int(time.time() if now is None else now)
        payload = {
            "exp": issued_at + self.ttl_seconds,
            "nonce": secrets.token_urlsafe(12),
            "username": username,
            "version": 1,
        }
        encoded = _base64url_encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        return f"{encoded}.{_base64url_encode(signature)}"

    def verify(self, token: str | None, now: int | None = None) -> str | None:
        if not token or token.count(".") != 1:
            return None
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = _base64url_encode(
            hmac.new(self._secret, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        if not hmac.compare_digest(supplied_signature, expected_signature):
            return None
        try:
            payload = json.loads(_base64url_decode(encoded))
            expires_at = int(payload["exp"])
            username = str(payload["username"])
            version = int(payload["version"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        current_time = int(time.time() if now is None else now)
        if version != 1 or expires_at <= current_time or not username:
            return None
        return username


class LoginThrottle:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 600) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: defaultdict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        attempts = self._attempts[client_id]
        cutoff = current_time - self.window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        if len(attempts) >= self.max_attempts:
            return False
        attempts.append(current_time)
        return True

    def clear(self, client_id: str) -> None:
        self._attempts.pop(client_id, None)
