from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import Chunk
from .source_tracker import SourceRecord


@dataclass
class IndexManifest:
    created_at: str
    embedding_model: str
    chunk_count: int
    source_count: int
    errors: list[str]
    config_signature: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"


def _safe_replace(src: Path, dst: Path, max_attempts: int = 5, delay: float = 0.05) -> None:
    """Atomically replace dst with src on Windows with retry for intermittent locks."""
    for attempt in range(max_attempts):
        try:
            os.replace(str(src), str(dst))
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                try:
                    if dst.exists():
                        dst.unlink()
                    os.replace(str(src), str(dst))
                    return
                except Exception:
                    shutil.copy2(str(src), str(dst))
                    try:
                        src.unlink()
                    except OSError:
                        pass
                    return


class IndexStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.chunks_path = directory / "chunks.json"
        self.embeddings_path = directory / "embeddings.npy"
        self.manifest_path = directory / "manifest.json"
        self.sources_path = directory / "sources.json"
        self.backup_dir = directory / "backup"

    def exists(self) -> bool:
        return self.chunks_path.exists() and self.embeddings_path.exists() and self.manifest_path.exists()

    def has_sources(self) -> bool:
        return self.sources_path.exists()

    def load_sources(self) -> dict[str, SourceRecord] | None:
        if not self.sources_path.exists():
            return None
        try:
            raw = json.loads(self.sources_path.read_text(encoding="utf-8"))
            return {k: SourceRecord.from_dict(v) for k, v in raw.items()}
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def save(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: str,
        errors: list[str],
        sources: dict[str, SourceRecord] | None = None,
        config_signature: dict[str, Any] | None = None,
    ) -> IndexManifest:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) and embedding count ({len(embeddings)}) do not match.")
        if embeddings.ndim != 2 and len(chunks) > 0:
            raise ValueError(f"Embeddings must be a 2D array, got shape {embeddings.shape}.")

        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = IndexManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            embedding_model=embedding_model,
            chunk_count=len(chunks),
            source_count=len({chunk.source_path for chunk in chunks}),
            errors=errors,
            config_signature=config_signature or {},
            schema_version="2.0",
        )

        staging_dir = self.directory / f".staging_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            stg_chunks = staging_dir / "chunks.json"
            stg_embeddings = staging_dir / "embeddings.npy"
            stg_manifest = staging_dir / "manifest.json"
            stg_sources = staging_dir / "sources.json"

            stg_chunks.write_text(
                json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            np.save(stg_embeddings, embeddings.astype(np.float32))
            stg_manifest.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

            if sources is not None:
                sources_payload = {k: rec.to_dict() for k, rec in sources.items()}
                stg_sources.write_text(json.dumps(sources_payload, indent=2), encoding="utf-8")

            # Verify staged files before activating
            verify_chunks = json.loads(stg_chunks.read_text(encoding="utf-8"))
            verify_embs = np.load(stg_embeddings, allow_pickle=False)
            if len(verify_chunks) != len(verify_embs):
                raise ValueError("Verification failed: Staged chunk and embedding counts do not match.")

            # Create backup of existing active files
            if self.exists():
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                for fpath in (self.chunks_path, self.embeddings_path, self.manifest_path, self.sources_path):
                    if fpath.exists():
                        shutil.copy2(str(fpath), str(self.backup_dir / fpath.name))

            # Atomically replace active files
            _safe_replace(stg_chunks, self.chunks_path)
            _safe_replace(stg_embeddings, self.embeddings_path)
            _safe_replace(stg_manifest, self.manifest_path)
            if sources is not None and stg_sources.exists():
                _safe_replace(stg_sources, self.sources_path)

        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        return manifest

    def load(self) -> tuple[list[Chunk], np.ndarray, IndexManifest]:
        if not self.exists():
            raise FileNotFoundError("The RAG index has not been built yet.")
        chunks = [Chunk.from_dict(item) for item in json.loads(self.chunks_path.read_text(encoding="utf-8"))]
        embeddings = np.load(self.embeddings_path, allow_pickle=False)
        manifest_raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        known_fields = {f.name for f in fields(IndexManifest)}
        manifest = IndexManifest(**{k: v for k, v in manifest_raw.items() if k in known_fields})
        if len(chunks) != len(embeddings):
            raise ValueError("The saved index is inconsistent. Rebuild it.")
        return chunks, embeddings, manifest

    def restore_from_backup(self) -> bool:
        """Attempt to restore the previous index snapshot from the backup directory."""
        if not self.backup_dir.exists():
            return False
        bk_chunks = self.backup_dir / "chunks.json"
        bk_embeddings = self.backup_dir / "embeddings.npy"
        bk_manifest = self.backup_dir / "manifest.json"
        if not (bk_chunks.exists() and bk_embeddings.exists() and bk_manifest.exists()):
            return False
        shutil.copy2(str(bk_chunks), str(self.chunks_path))
        shutil.copy2(str(bk_embeddings), str(self.embeddings_path))
        shutil.copy2(str(bk_manifest), str(self.manifest_path))
        bk_sources = self.backup_dir / "sources.json"
        if bk_sources.exists():
            shutil.copy2(str(bk_sources), str(self.sources_path))
        return True
