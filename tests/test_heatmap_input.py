"""Regression tests for heatmap URL SSRF validation and proxy Fake-IP mode."""

from __future__ import annotations

import socket

import pytest

from tradingagents import heatmap_input


def _dns_result(address: str):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]


@pytest.mark.unit
def test_public_https_accepts_proxy_fake_ip_after_public_doh_verification(monkeypatch):
    monkeypatch.setattr(
        heatmap_input.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _dns_result("198.18.26.185"),
    )
    seen = []

    def fake_doh(hostname):
        seen.append(hostname)
        return frozenset({"104.21.61.238", "172.67.216.251"})

    monkeypatch.setattr(heatmap_input, "_resolve_public_addresses_doh", fake_doh)

    heatmap_input._validate_public_https(
        "https://img.seekno.de/i/u/2026/08/22/nb11gq.webp"
    )

    assert seen == ["img.seekno.de"]


@pytest.mark.unit
def test_private_address_is_not_allowed_to_use_fake_ip_exception(monkeypatch):
    monkeypatch.setattr(
        heatmap_input.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _dns_result("192.168.2.10"),
    )
    monkeypatch.setattr(
        heatmap_input,
        "_resolve_public_addresses_doh",
        lambda _hostname: pytest.fail("ordinary private DNS must not use DoH exception"),
    )

    with pytest.raises(heatmap_input.HeatmapInputError, match="non-public"):
        heatmap_input._validate_public_https("https://internal.example/heatmap.webp")


@pytest.mark.unit
def test_doh_verifier_requires_only_public_a_and_aaaa_records(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(_url, *, params, **_kwargs):
        answer = (
            [{"name": "img.seekno.de", "type": 1, "data": "104.21.61.238"}]
            if params["type"] == "A"
            else [{"name": "img.seekno.de", "type": 28, "data": "2606:4700::6815:3dee"}]
        )
        return FakeResponse(
            {
                "Status": 0,
                "Question": [{"name": "img.seekno.de", "type": 1}],
                "Answer": answer,
            }
        )

    monkeypatch.setattr(heatmap_input.requests, "get", fake_get)
    heatmap_input._resolve_public_addresses_doh.cache_clear()

    assert heatmap_input._resolve_public_addresses_doh("img.seekno.de") == frozenset(
        {"104.21.61.238", "2606:4700::6815:3dee"}
    )


@pytest.mark.unit
def test_doh_verifier_rejects_private_real_record(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Status": 0,
                "Question": [{"name": "bad.example", "type": 1}],
                "Answer": [{"name": "bad.example", "type": 1, "data": "10.0.0.5"}],
            }

    monkeypatch.setattr(heatmap_input.requests, "get", lambda *_args, **_kwargs: FakeResponse())
    heatmap_input._resolve_public_addresses_doh.cache_clear()

    with pytest.raises(heatmap_input.HeatmapInputError, match="non-public"):
        heatmap_input._resolve_public_addresses_doh("bad.example")
