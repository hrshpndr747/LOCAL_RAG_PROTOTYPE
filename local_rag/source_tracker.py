from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .media import VIDEO_SUFFIXES, media_sidecar


def canonical_source_id(path: Path) -> str:
    """Return a normalized, canonical source identifier for a filesystem path."""
    return os.path.normcase(os.path.normpath(str(path.resolve())))


def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hex digest of a file in streaming chunks."""
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return ""


def text_sha256(text: str) -> str:
    """Compute SHA-256 for a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_associated_sidecars(path: Path) -> dict[str, Path]:
    """Return all known associated sidecar / companion file paths for a source file."""
    sidecars: dict[str, Path] = {
        "media": media_sidecar(path),
        "confluence": path.with_name(f".{path.name}.confluence.json"),
    }
    if path.suffix.lower() in VIDEO_SUFFIXES:
        sidecars["companion_vtt"] = path.with_suffix(".vtt")
    return sidecars


def get_sidecar_fingerprints(path: Path) -> dict[str, dict[str, Any]]:
    """Inspect and return size, mtime, and content hashes for associated sidecars."""
    sidecars = get_associated_sidecars(path)
    result: dict[str, dict[str, Any]] = {}
    for key, sidecar_path in sidecars.items():
        if sidecar_path.is_file():
            try:
                stat = sidecar_path.stat()
                result[key] = {
                    "path": str(sidecar_path),
                    "exists": True,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "hash": file_sha256(sidecar_path),
                }
            except OSError:
                result[key] = {"path": str(sidecar_path), "exists": False}
        else:
            result[key] = {"path": str(sidecar_path), "exists": False}
    return result


@dataclass
class SourceRecord:
    source_id: str
    source_path: str
    file_path: str
    file_size: int
    mtime_ns: int
    content_hash: str
    sidecars: dict[str, dict[str, Any]] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    chunk_input_hashes: list[str] = field(default_factory=list)
    status: str = "success"  # "success", "failed"
    last_indexed_at: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        return cls(**data)


@dataclass
class DiscoveredSource:
    path: Path
    root: Path
    source_id: str
    source_path: str
    stat: os.stat_result


@dataclass
class ChangeDetectionResult:
    new_sources: dict[str, DiscoveredSource] = field(default_factory=dict)
    modified_sources: dict[str, DiscoveredSource] = field(default_factory=dict)
    unchanged_sources: dict[str, DiscoveredSource] = field(default_factory=dict)
    deleted_source_ids: list[str] = field(default_factory=list)
    failed_sources: dict[str, str] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_sources or self.modified_sources or self.deleted_source_ids or self.failed_sources)


def _sidecars_match_quick(current_sidecars: dict[str, dict[str, Any]], tracked_sidecars: dict[str, dict[str, Any]]) -> bool:
    """Quick comparison of sidecars based on existence, size, and mtime."""
    if set(current_sidecars.keys()) != set(tracked_sidecars.keys()):
        return False
    for key, curr in current_sidecars.items():
        tr = tracked_sidecars.get(key)
        if not tr:
            return False
        if curr.get("exists") != tr.get("exists"):
            return False
        if curr.get("exists"):
            if curr.get("size") != tr.get("size") or curr.get("mtime_ns") != tr.get("mtime_ns"):
                return False
    return True


def _sidecars_match_hash(current_sidecars: dict[str, dict[str, Any]], tracked_sidecars: dict[str, dict[str, Any]]) -> bool:
    """Full hash comparison of sidecars."""
    for key, curr in current_sidecars.items():
        tr = tracked_sidecars.get(key, {})
        if curr.get("exists") != tr.get("exists"):
            return False
        if curr.get("exists"):
            curr_hash = curr.get("hash") or file_sha256(Path(curr["path"]))
            tr_hash = tr.get("hash")
            if curr_hash != tr_hash:
                return False
    return True


def detect_changes(
    discovered: Iterable[DiscoveredSource],
    tracked: dict[str, SourceRecord],
) -> ChangeDetectionResult:
    """Classify discovered sources against the tracked state into new, modified, unchanged, deleted."""
    result = ChangeDetectionResult()
    discovered_by_id: dict[str, DiscoveredSource] = {}

    for disc in discovered:
        discovered_by_id[disc.source_id] = disc
        rec = tracked.get(disc.source_id)
        if rec is None:
            result.new_sources[disc.source_id] = disc
            continue

        if rec.status == "failed":
            result.modified_sources[disc.source_id] = disc
            continue

        curr_sidecars = get_sidecar_fingerprints(disc.path)
        size_matches = (disc.stat.st_size == rec.file_size)
        mtime_matches = (disc.stat.st_mtime_ns == rec.mtime_ns)
        sidecars_match = _sidecars_match_quick(curr_sidecars, rec.sidecars)

        if size_matches and mtime_matches and sidecars_match:
            result.unchanged_sources[disc.source_id] = disc
            continue

        # Timestamps or sizes differed, check content hashes
        curr_hash = file_sha256(disc.path)
        hashes_match = (curr_hash == rec.content_hash)
        sidecar_hashes_match = _sidecars_match_hash(curr_sidecars, rec.sidecars)

        if hashes_match and sidecar_hashes_match:
            # Content is identical despite timestamp change: update metadata without re-indexing
            rec.file_size = disc.stat.st_size
            rec.mtime_ns = disc.stat.st_mtime_ns
            rec.sidecars = curr_sidecars
            result.unchanged_sources[disc.source_id] = disc
        else:
            result.modified_sources[disc.source_id] = disc

    # Check for deleted sources
    for source_id, rec in tracked.items():
        if source_id not in discovered_by_id:
            result.deleted_source_ids.append(source_id)

    return result
