"""Single-origin FastAPI + Web frontend launcher for Google Colab."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

if "matplotlib_inline" in os.environ.get("MPLBACKEND", ""):
    os.environ["MPLBACKEND"] = "Agg"

import uvicorn
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.gui.web.server_runtime import configure_uvicorn_websocket_protocol
from src.utils.runtime_paths import get_resource_path
from src.web_api.app import create_app
from src.web_contract import API_VERSION

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860
DEFAULT_RETENTION_DAYS = 30
DEFAULT_RETENTION_MAX_JOBS = 200
_LOOPBACK_HOSTS = {"0.0.0.0", "::", ""}
_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (
        b"permissions-policy",
        b"camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    ),
)


def _header_map(raw_headers: Iterable[tuple[bytes, bytes]]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in raw_headers:
        key = name.decode("latin-1").strip().lower()
        val = value.decode("latin-1").strip()
        if key and val and key not in headers:
            headers[key] = val
    return headers


def _first_forwarded_token(value: str) -> str:
    return value.split(",", 1)[0].strip()


def _validated_host_header(candidate: str) -> tuple[str, str] | None:
    """Validate a Host/X-Forwarded-Host header and return (netloc, hostname) or None."""

    value = str(candidate).strip()
    if not value or any(ch.isspace() for ch in value):
        return None
    try:
        parsed = urlsplit(f"http://{value}")
        port = parsed.port
    except ValueError:
        return None
    if (
        not parsed.hostname
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    if port is not None and not 1 <= port <= 65535:
        return None
    return parsed.netloc, parsed.hostname


def resolve_request_origin(
    raw_headers: Iterable[tuple[bytes, bytes]],
    *,
    fallback_scheme: str = "http",
    fallback_host: str = "127.0.0.1",
    fallback_port: int = DEFAULT_PORT,
) -> str:
    """Resolve and validate the browser-visible origin from request/proxy headers."""

    headers = _header_map(raw_headers)
    forwarded_host = _first_forwarded_token(headers.get("x-forwarded-host", ""))
    parsed_host = _validated_host_header(forwarded_host) or _validated_host_header(
        headers.get("host", "")
    )
    if parsed_host is None:
        safe_host = "127.0.0.1" if fallback_host in _LOOPBACK_HOSTS else fallback_host
        fallback_candidate = (
            f"[{safe_host}]:{fallback_port}"
            if ":" in safe_host and not safe_host.startswith("[")
            else f"{safe_host}:{fallback_port}"
        )
        parsed_host = _validated_host_header(fallback_candidate) or (
            f"127.0.0.1:{DEFAULT_PORT}",
            "127.0.0.1",
        )

    netloc, hostname = parsed_host
    forwarded_proto = _first_forwarded_token(headers.get("x-forwarded-proto", "")).lower()
    if forwarded_proto in {"http", "https"}:
        scheme = forwarded_proto
    elif hostname.lower().endswith(".trycloudflare.com"):
        scheme = "https"
    elif fallback_scheme in {"http", "https"}:
        scheme = fallback_scheme
    else:
        scheme = "http"
    return f"{scheme}://{netloc}".rstrip("/")


def build_runtime_config(origin: str) -> dict[str, object]:
    """Return same-origin runtime-config.json payload for the standalone Web UI."""

    normalized = origin.rstrip("/")
    return {
        "managed": True,
        "frontend_url": normalized,
        "backend_url": normalized,
        "expected_api_version": API_VERSION,
    }


class ColabRuntimeConfigMiddleware:
    """Serve dynamic /runtime-config.json and baseline security headers."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        fallback_host: str = "127.0.0.1",
        fallback_port: int = DEFAULT_PORT,
    ) -> None:
        self.app = app
        self._fallback_host = fallback_host
        self._fallback_port = fallback_port

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method") or "GET").upper()
        if scope.get("path") == "/runtime-config.json" and method in {"GET", "HEAD"}:
            origin = resolve_request_origin(
                scope.get("headers") or (),
                fallback_scheme=str(scope.get("scheme") or "http"),
                fallback_host=self._fallback_host,
                fallback_port=self._fallback_port,
            )
            body = json.dumps(build_runtime_config(origin), ensure_ascii=False).encode("utf-8")
            headers = [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"cache-control", b"no-store"),
                (b"content-length", str(len(body)).encode("ascii")),
                *_SECURITY_HEADERS,
            ]
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send(
                {
                    "type": "http.response.body",
                    "body": b"" if method == "HEAD" else body,
                }
            )
            return

        async def send_with_security_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                raw = list(message.get("headers") or ())
                existing = {name.lower() for name, _ in raw}
                for header_name, header_value in _SECURITY_HEADERS:
                    if header_name not in existing:
                        raw.append((header_name, header_value))
                message = {**message, "headers": raw}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


def create_colab_app(
    *,
    data_root: str | Path | None = None,
    frontend_root: str | Path | None = None,
    cors_origins: list[str] | None = None,
    max_upload_bytes: int | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    retention_max_jobs: int = DEFAULT_RETENTION_MAX_JOBS,
    fallback_host: str = "127.0.0.1",
    fallback_port: int = DEFAULT_PORT,
) -> ASGIApp:
    """Create the single-origin FastAPI + Web UI ASGI application for Colab."""

    resolved_frontend = Path(frontend_root or get_resource_path("web")).resolve()
    app = create_app(
        data_root=data_root,
        cors_origins=cors_origins,
        frontend_root=resolved_frontend,
        max_upload_bytes=max_upload_bytes,
        retention_days=retention_days,
        retention_max_jobs=retention_max_jobs,
    )
    return ColabRuntimeConfigMiddleware(
        app,
        fallback_host=fallback_host,
        fallback_port=fallback_port,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Music to MIDI Web UI + Inference API for Colab"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--frontend-dir", default=None)
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="Optional explicit CORS origin; omit for single-origin mode",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
    parser.add_argument("--retention-max-jobs", type=int, default=DEFAULT_RETENTION_MAX_JOBS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = create_colab_app(
        data_root=args.data_dir,
        frontend_root=args.frontend_dir,
        cors_origins=args.cors_origin,
        retention_days=args.retention_days,
        retention_max_jobs=args.retention_max_jobs,
        fallback_host=args.host,
        fallback_port=args.port,
    )
    websocket_protocol = configure_uvicorn_websocket_protocol()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=str(args.log_level).lower(),
        ws=websocket_protocol,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1,::1",
    )
    return 0


__all__ = [
    "ColabRuntimeConfigMiddleware",
    "build_runtime_config",
    "create_colab_app",
    "main",
    "resolve_request_origin",
]

if __name__ == "__main__":
    raise SystemExit(main())
