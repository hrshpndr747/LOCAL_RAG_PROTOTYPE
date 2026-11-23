#!/usr/bin/env python
"""Diagnostic and verification CLI for incremental indexing in Local RAG."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_rag.config import Settings
from local_rag.loaders import discover_sources
from local_rag.service import RAGService
from local_rag.source_tracker import detect_changes
from local_rag.table_parsing import TABLE_PARSER_VERSION, get_parser_version


def check_status(service: RAGService) -> None:
    print("=" * 60)
    print("Local RAG Index Diagnostic Tool")
    print("=" * 60)
    print(f"Data directory:  {service.settings.data_dir}")
    print(f"Index directory: {service.settings.index_dir}")
    print(f"Embedding model: {service.settings.embed_model}")
    print(f"Chat model:      {service.settings.chat_model}")
    print(f"Table parser:    {TABLE_PARSER_VERSION}")
    print("-" * 60)

    if not service.store.exists():
        print("Index Status: [NOT BUILT]")
        print("Run with --update or --rebuild to build the initial index.")
        return

    manifest = service.manifest
    print("Index Status: [ACTIVE]")
    print(f"Created at:      {manifest.created_at}")
    print(f"Indexed model:   {manifest.embedding_model}")
    print(f"Total chunks:    {manifest.chunk_count}")
    print(f"Total sources:   {manifest.source_count}")
    print(f"Has sources.json: {'Yes (v2 incremental)' if service.store.has_sources() else 'No (legacy v1)'}")
    print(f"Has backup dir:   {'Yes' if service.store.backup_dir.exists() else 'No'}")

    if manifest.errors:
        print(f"Reported errors ({len(manifest.errors)}):")
        for err in manifest.errors[:5]:
            print(f"  - {err}")
        if len(manifest.errors) > 5:
            print(f"  ... and {len(manifest.errors) - 5} more")

    print("-" * 60)
    print("Checking filesystem for pending source changes...")
    roots = service.source_roots()
    discovered = discover_sources(roots, service.settings.index_dir)
    chunks, _, _ = service.store.load()
    tracked = service._get_or_migrate_sources(chunks, discovered)
    changes = detect_changes(discovered, tracked)

    print(f"Discovered on disk: {len(discovered)} supported files")
    print(f"Pending changes:    {'YES' if changes.has_changes else 'NONE (Index is up to date)'}")
    print(f"  - New files:       {len(changes.new_sources)}")
    for s in list(changes.new_sources.values())[:5]:
        print(f"      + {s.source_path}")
    print(f"  - Modified files:  {len(changes.modified_sources)}")
    for s in list(changes.modified_sources.values())[:5]:
        print(f"      * {s.source_path}")
    print(f"  - Deleted files:   {len(changes.deleted_source_ids)}")
    for sid in changes.deleted_source_ids[:5]:
        print(f"      - {tracked[sid].source_path if sid in tracked else sid}")
    print(f"  - Unchanged files: {len(changes.unchanged_sources)}")
    print("=" * 60)


def run_incremental_update(service: RAGService) -> None:
    print("Starting incremental index update...")
    start = time.perf_counter()
    result = service.update_index()
    elapsed = time.perf_counter() - start

    print("\nResult:")
    print(f"Status:             {result.status}")
    print(f"Message:            {result.message}")
    print(f"New files:          {len(result.new_files)}")
    print(f"Modified files:     {len(result.modified_files)}")
    print(f"Deleted files:      {len(result.deleted_files)}")
    print(f"Unchanged files:    {len(result.unchanged_files)}")
    print(f"New embeddings:     {result.new_embeddings}")
    print(f"Reused embeddings:  {result.reused_embeddings}")
    print(f"Total files:        {result.total_files}")
    print(f"Total chunks:       {result.total_chunks}")
    print(f"Processing time:    {result.elapsed_seconds:.2f}s (wall time: {elapsed:.2f}s)")

    if result.failed_files:
        print("\nFailed files:")
        for path, err in result.failed_files.items():
            print(f"  - {path}: {err}")


def run_full_rebuild(service: RAGService) -> None:
    print("Starting full rebuild from scratch...")
    start = time.perf_counter()
    manifest = service.build_index()
    elapsed = time.perf_counter() - start

    print("\nResult:")
    print(f"Full index rebuilt: {manifest.chunk_count} chunks from {manifest.source_count} files in {elapsed:.2f}s.")
    if manifest.errors:
        print("Skipped files / errors:")
        for err in manifest.errors[:10]:
            print(f"  - {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and manage incremental indexing.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Inspect index status and pending changes (default)")
    group.add_argument("--update", action="store_true", help="Execute incremental update")
    group.add_argument("--rebuild", action="store_true", help="Execute full index rebuild")
    args = parser.parse_args()

    settings = Settings.from_env()
    service = RAGService(settings)

    if args.update:
        run_incremental_update(service)
    elif args.rebuild:
        run_full_rebuild(service)
    else:
        check_status(service)


if __name__ == "__main__":
    main()
