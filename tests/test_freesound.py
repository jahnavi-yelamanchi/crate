"""Freesound client — retry/backoff on 429 without real network."""

import time

import crate.data.freesound as fs


class _Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fs.requests.HTTPError(f"{self.status_code}")


def test_get_backs_off_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp(429, headers={"Retry-After": "0"})
        return _Resp(200, {"results": [{"id": 1}]})

    monkeypatch.setattr(fs.requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    data = fs._get("http://x", {})
    assert data["results"][0]["id"] == 1
    assert calls["n"] == 2  # one retry after the 429


def test_get_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(fs.requests, "get", lambda *a, **k: _Resp(429, headers={"Retry-After": "0"}))
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    try:
        fs._get("http://x", {}, retries=3)
        assert False, "should have raised"
    except fs.requests.HTTPError:
        pass
