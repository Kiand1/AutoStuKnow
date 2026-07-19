import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

SESSION_COOKIE = "autostuknow_session"
PASSWORD_HASH_ITERATIONS = 600_000


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


@dataclass(frozen=True)
class WebSession:
    username: str
    credential_revision: int


class WebSessionSigner:
    def __init__(self, secret: str, ttl_seconds: int) -> None:
        self._secret = secret.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def create(
        self,
        username: str,
        credential_revision: int = 0,
        now: int | None = None,
    ) -> str:
        issued_at = int(time.time() if now is None else now)
        payload = {
            "credential_revision": credential_revision,
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

    def verify(self, token: str | None, now: int | None = None) -> WebSession | None:
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
            credential_revision = int(payload["credential_revision"])
            expires_at = int(payload["exp"])
            username = str(payload["username"])
            version = int(payload["version"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        current_time = int(time.time() if now is None else now)
        if (
            version != 1
            or credential_revision < 0
            or expires_at <= current_time
            or not username
        ):
            return None
        return WebSession(username=username, credential_revision=credential_revision)


class WebCredentialStore:
    def __init__(
        self,
        data_dir: Path,
        initial_password: str,
        iterations: int = PASSWORD_HASH_ITERATIONS,
    ) -> None:
        self.auth_dir = data_dir / "auth"
        self.path = self.auth_dir / "web-credentials.json"
        self._initial_password = initial_password
        self._iterations = iterations
        self._lock = Lock()
        self._record = self._load()

    @property
    def must_change_password(self) -> bool:
        return self._record is None

    @property
    def revision(self) -> int:
        return 0 if self._record is None else int(self._record["revision"])

    def verify(self, password: str) -> bool:
        with self._lock:
            record = self._record
        if record is None:
            return hmac.compare_digest(
                password.encode("utf-8"),
                self._initial_password.encode("utf-8"),
            )
        salt = _base64url_decode(str(record["salt"]))
        expected = _base64url_decode(str(record["password_hash"]))
        supplied = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            int(record["iterations"]),
        )
        return hmac.compare_digest(supplied, expected)

    def set_password(self, password: str) -> int:
        if len(password) < 8 or len(password) > 256:
            raise ValueError("新密码长度必须为 8 至 256 位")
        salt = secrets.token_bytes(16)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self._iterations,
        )
        with self._lock:
            revision = self.revision + 1
            record: dict[str, str | int] = {
                "algorithm": "pbkdf2_sha256",
                "iterations": self._iterations,
                "password_hash": _base64url_encode(password_hash),
                "revision": revision,
                "salt": _base64url_encode(salt),
                "version": 1,
            }
            self._write(record)
            self._record = record
        return revision

    def _load(self) -> dict[str, str | int] | None:
        if not self.path.exists():
            return None
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                int(record["version"]) != 1
                or str(record["algorithm"]) != "pbkdf2_sha256"
                or int(record["iterations"]) < 100_000
                or int(record["revision"]) < 1
            ):
                raise ValueError("unsupported credential format")
            salt = _base64url_decode(str(record["salt"]))
            password_hash = _base64url_decode(str(record["password_hash"]))
            if len(salt) < 16 or len(password_hash) != hashlib.sha256().digest_size:
                raise ValueError("invalid credential hash")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Web credential file is invalid: {self.path}") from exc
        return record

    def _write(self, record: dict[str, str | int]) -> None:
        self.auth_dir.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            os.chmod(self.auth_dir, 0o700)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with suppress(OSError):
            os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)


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
