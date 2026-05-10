"""Basic unit tests for tiny-boss."""

import pytest
from tiny_boss.protocol import _extract_json
from tiny_boss.clients import _retry


class TestExtractJson:
    def test_valid_json(self):
        assert _extract_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_markdown(self):
        result = _extract_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_in_text(self):
        result = _extract_json('some text {"b": 2} more text')
        assert result == {"b": 2}

    def test_plain_text_raises(self):
        with pytest.raises(ValueError):
            _extract_json("no json here at all")


class TestRetry:
    def test_succeeds_first_try(self):
        calls = []
        result = _retry(lambda: calls.append(1) or "ok")
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_on_failure(self):
        counter = {"n": 0}

        def flaky():
            counter["n"] += 1
            if counter["n"] < 3:
                # Simulate a rate limit error
                err = Exception("rate limited")
                err.status_code = 429
                raise err
            return "recovered"

        result = _retry(flaky, max_attempts=3)
        assert result == "recovered"
        assert counter["n"] == 3

    def test_raises_after_exhausted(self):
        def always_fail():
            err = Exception("overloaded")
            err.status_code = 503
            raise err

        with pytest.raises(Exception, match="overloaded"):
            _retry(always_fail, max_attempts=2)

    def test_fails_fast_on_config_error(self):
        def bad_auth():
            # AuthenticationError is not transient
            err = ValueError("invalid api key")
            raise err

        with pytest.raises(ValueError, match="invalid api key"):
            _retry(bad_auth, max_attempts=3)
