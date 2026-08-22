"""Validation and local staging for optional liquidation-heatmap images."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import shutil
import socket
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

MAX_HEATMAP_BYTES = 12 * 1024 * 1024
MAX_REDIRECTS = 3
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"RIFF", "image/webp", ".webp"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
)


class HeatmapInputError(ValueError):
    """The optional heatmap input was unsafe, inaccessible, or not an image."""


def _image_type(data: bytes) -> tuple[str, str]:
    for magic, mime, suffix in _MAGIC:
        if data.startswith(magic):
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime, suffix
    raise HeatmapInputError("heatmap input is not a supported PNG, JPEG, WebP, or GIF image")


def _validate_public_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise HeatmapInputError("remote heatmap input must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise HeatmapInputError("remote heatmap URL must not contain credentials")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise HeatmapInputError(f"could not resolve heatmap host: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise HeatmapInputError("remote heatmap URL resolves to a non-public address")


def _read_local(path: Path) -> tuple[bytes, str | None]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise HeatmapInputError("local heatmap input is not a regular file")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_HEATMAP_BYTES:
        raise HeatmapInputError(f"heatmap image must be between 1 byte and {MAX_HEATMAP_BYTES} bytes")
    return resolved.read_bytes(), datetime.fromtimestamp(
        resolved.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")


def _download(url: str) -> tuple[bytes, str | None, str]:
    current = url
    session = requests.Session()
    headers = {"User-Agent": "TradingAgents/crypto-heatmap"}
    for _ in range(MAX_REDIRECTS + 1):
        _validate_public_https(current)
        response = session.get(
            current,
            headers=headers,
            timeout=20,
            stream=True,
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                raise HeatmapInputError("heatmap redirect omitted its destination")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        declared = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if declared and not (
            declared.startswith("image/")
            or declared in {"application/octet-stream", "binary/octet-stream"}
        ):
            raise HeatmapInputError(f"heatmap URL returned unsupported content type {declared!r}")
        body = bytearray()
        for chunk in response.iter_content(64 * 1024):
            body.extend(chunk)
            if len(body) > MAX_HEATMAP_BYTES:
                raise HeatmapInputError("heatmap download exceeds the size limit")
        capture = response.headers.get("last-modified")
        return bytes(body), capture, current
    raise HeatmapInputError("heatmap URL exceeded the redirect limit")


def stage_heatmap_input(value: str, cache_dir: str | Path, analysis_date: str) -> dict:
    """Validate an image, copy it to the cache, and return serializable metadata."""
    original = str(value or "").strip()
    if not original:
        return {}

    provided_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        if original.lower().startswith(("http://", "https://")):
            data, capture_time, resolved_source = _download(original)
            source_kind = "https_url"
            capture_time_source = "http_last_modified" if capture_time else "unavailable"
        else:
            data, capture_time = _read_local(Path(original))
            resolved_source = str(Path(original).expanduser().resolve())
            source_kind = "local_file"
            capture_time_source = "file_mtime"
    except HeatmapInputError:
        raise
    except (OSError, requests.RequestException) as exc:
        raise HeatmapInputError(f"could not stage heatmap input: {exc}") from exc

    mime_type, suffix = _image_type(data)
    digest = hashlib.sha256(data).hexdigest()
    destination_dir = Path(cache_dir) / "heatmaps"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{digest}{suffix}"
    if not destination.exists():
        destination.write_bytes(data)

    normalized_capture = None
    if capture_time:
        try:
            normalized_capture = datetime.fromisoformat(
                str(capture_time).replace("Z", "+00:00")
            )
        except ValueError:
            try:
                normalized_capture = parsedate_to_datetime(str(capture_time))
            except (TypeError, ValueError):
                normalized_capture = None
    capture_date = normalized_capture.date() if normalized_capture else None
    requested_date = datetime.strptime(analysis_date, "%Y-%m-%d").date()
    relation = "capture_time_unknown"
    if capture_date:
        delta_days = (capture_date - requested_date).days
        if delta_days > 1:
            relation = "post_cutoff_reference"
        elif delta_days >= 0:
            relation = "live_post_close_context"
        else:
            relation = "at_or_before_cutoff"

    metadata = {
        "original_input": original,
        "resolved_source": resolved_source,
        "source_kind": source_kind,
        "local_path": str(destination),
        "mime_type": mime_type,
        "sha256": digest,
        "size_bytes": len(data),
        "provided_at": provided_at,
        "capture_time": normalized_capture.isoformat(timespec="seconds") if normalized_capture else capture_time,
        "capture_time_source": capture_time_source,
        "time_relation": relation,
        "nature": "estimated_visual_extraction",
    }
    (destination.with_suffix(destination.suffix + ".json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def copy_heatmap_artifact(metadata: dict, destination_dir: str | Path) -> dict:
    """Copy a staged heatmap and provenance JSON into a saved report tree."""
    if not metadata or not metadata.get("local_path"):
        return metadata
    source = Path(metadata["local_path"])
    if not source.exists():
        return metadata
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = destination_dir / f"liquidation_heatmap{source.suffix}"
    shutil.copy2(source, copied)
    exported = dict(metadata)
    exported["report_artifact"] = copied.name
    (destination_dir / "liquidation_heatmap.metadata.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return exported
