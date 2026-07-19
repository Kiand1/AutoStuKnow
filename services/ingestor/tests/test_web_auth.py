from app.web_auth import LoginThrottle, WebSessionSigner


def test_signed_web_session_accepts_valid_and_rejects_tampered_or_expired() -> None:
    signer = WebSessionSigner("s" * 48, ttl_seconds=60)
    token = signer.create("admin", now=1_000)

    assert signer.verify(token, now=1_059) == "admin"
    assert signer.verify(token, now=1_060) is None
    assert signer.verify(token + "tampered", now=1_001) is None


def test_login_throttle_limits_attempts_and_can_be_cleared() -> None:
    throttle = LoginThrottle(max_attempts=2, window_seconds=10)

    assert throttle.allow("client", now=100)
    assert throttle.allow("client", now=101)
    assert not throttle.allow("client", now=102)
    assert throttle.allow("client", now=111)
    throttle.clear("client")
    assert throttle.allow("client", now=112)
