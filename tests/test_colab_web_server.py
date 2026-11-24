from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.web_api.app as app_module
import src.web_api.colab_server as colab_server_module
from src.model_profiles import ALL_PROFILE_IDS, ModelProfileSnapshot, ModelProfileStatus
from src.web_api.colab_server import create_colab_app, main, resolve_request_origin
from src.web_contract import API_VERSION


@pytest.fixture(autouse=True)
def _mock_profiles(monkeypatch):
    snapshot = ModelProfileSnapshot(
        selection_mode="test",
        explicit_selection_required=False,
        enabled_profiles=ALL_PROFILE_IDS,
        profiles=tuple(
            ModelProfileStatus(id=profile_id, enabled=True, available=True)
            for profile_id in ALL_PROFILE_IDS
        ),
        ready=True,
    )
    monkeypatch.setattr(app_module, "inspect_model_profiles", lambda: snapshot)


def test_resolve_request_origin_prefers_forwarded_headers():
    origin = resolve_request_origin(
        [
            (b"host", b"127.0.0.1:7860"),
            (b"x-forwarded-host", b"demo-tunnel.trycloudflare.com"),
            (b"x-forwarded-proto", b"https, http"),
        ]
    )
    assert origin == "https://demo-tunnel.trycloudflare.com"


def test_resolve_request_origin_infers_https_for_trycloudflare():
    origin = resolve_request_origin([(b"host", b"abc-xyz.trycloudflare.com")])
    assert origin == "https://abc-xyz.trycloudflare.com"


def test_resolve_request_origin_rejects_injected_host_and_supports_ipv6():
    injected = resolve_request_origin(
        [
            (b"x-forwarded-host", b"user:pass@evil.example/a.trycloudflare.com"),
            (b"host", b"evil.example/path"),
        ],
        fallback_host="0.0.0.0",
        fallback_port=7860,
    )
    assert injected == "http://127.0.0.1:7860"

    ipv6 = resolve_request_origin([(b"host", b"[::1]:7860")])
    assert ipv6 == "http://[::1]:7860"


def test_colab_app_serves_dynamic_runtime_config_head_and_security_headers(tmp_path):
    app = create_colab_app(data_root=tmp_path / "data")
    with TestClient(app) as client:
        config_res = client.get(
            "/runtime-config.json",
            headers={
                "Host": "colab-ui.trycloudflare.com",
                "X-Forwarded-Proto": "https",
            },
        )
        assert config_res.status_code == 200
        assert config_res.headers["cache-control"] == "no-store"
        assert config_res.headers["x-content-type-options"] == "nosniff"
        assert config_res.headers["referrer-policy"] == "no-referrer"
        assert config_res.json() == {
            "managed": True,
            "frontend_url": "https://colab-ui.trycloudflare.com",
            "backend_url": "https://colab-ui.trycloudflare.com",
            "expected_api_version": API_VERSION,
        }

        head_res = client.head(
            "/runtime-config.json",
            headers={"Host": "colab-ui.trycloudflare.com"},
        )
        assert head_res.status_code == 200
        assert head_res.content == b""
        assert int(head_res.headers["content-length"]) > 0

        index_res = client.get("/")
        assert index_res.status_code == 200
        assert index_res.headers["x-content-type-options"] == "nosniff"
        assert index_res.headers["referrer-policy"] == "no-referrer"
        assert "Music to MIDI" in index_res.text

        health_res = client.get(
            "/api/v1/health",
            headers={"Origin": "https://untrusted.example"},
        )
        assert health_res.status_code == 200
        assert "access-control-allow-origin" not in health_res.headers

        sf_js_res = client.get("/api/v1/soundfont-ui.js")
        assert sf_js_res.status_code == 200


def test_colab_app_rejects_missing_frontend_root(tmp_path):
    with pytest.raises(RuntimeError, match="frontend index is missing"):
        create_colab_app(
            data_root=tmp_path / "data",
            frontend_root=tmp_path / "missing-web",
        )


def test_colab_server_main_restricts_forwarded_ips_and_runs_uvicorn(monkeypatch, tmp_path):
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(colab_server_module.uvicorn, "run", fake_run)
    exit_code = main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "7861",
            "--data-dir",
            str(tmp_path / "data"),
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 7861
    assert captured["kwargs"]["proxy_headers"] is True
    assert captured["kwargs"]["forwarded_allow_ips"] == "127.0.0.1,::1"


def test_colab_web_ui_notebook_contract():
    notebook = json.loads(Path("colab_web_ui.ipynb").read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert len(code_cells) == 4
    for cell_source in code_cells:
        ast.parse(cell_source)

    joined = "\n".join(code_cells)
    assert 'MPLBACKEND"] = "Agg"' in joined
    assert "src.web_api.colab_server" in joined
    assert "src.model_profiles" in joined
    assert "if ENABLE_MUSCRIPTOR:" in joined
    assert "https://github.com/anhlocvu/music-to-midi.git" in joined


def test_colab_app_exposes_vietnamese_capabilities_and_project_labels(tmp_path):
    app = create_colab_app(data_root=tmp_path / "data")
    with TestClient(app) as client:
        caps = client.get("/api/v1/capabilities")
        assert caps.status_code == 200
        payload = caps.json()
        assert all("label_vi" in item and item["label_vi"] for item in payload["modes"])
        assert all("label_vi" in item and item["label_vi"] for item in payload["muscriptor_models"])
        assert all("label_vi" in item and item["label_vi"] for item in payload["muscriptor_instruments"])
        assert all("label_vi" in item and item["label_vi"] for item in payload["tempo_modes"])

        proj_caps = client.get("/api/v1/projects/capabilities")
        assert proj_caps.status_code == 200
        assert "vi_VN" in proj_caps.json()["labels"]
