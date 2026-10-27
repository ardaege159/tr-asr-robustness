from trasr.models import _rate_limit_text, _retry_after


def test_retry_after_formats():
    assert _retry_after("Please try again in 3m12.24s. Need more tokens?") == 192.24
    assert _retry_after("Please retry in 22h39m9.5s.") == 22 * 3600 + 39 * 60 + 9.5
    assert _retry_after("try again in 577ms") == 0.577
    assert _retry_after("quota exceeded") is None


def test_rate_limit_text_includes_body():
    class Response:
        text = '[{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}]'

    class Err(Exception):
        response = Response()

    assert "PerDay" in _rate_limit_text(Err("Error code: 429"))
