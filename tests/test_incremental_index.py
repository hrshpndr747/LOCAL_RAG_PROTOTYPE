from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from local_rag.config import Settings
from local_rag.index_store import IndexManifest, IndexStore
from local_rag.models import Chunk
from local_rag.service import RAGService


class TrackingFakeClient:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.embed_calls: list[list[str]] = []
        self.fail_next_embed = False

    def embed(self, texts, model: str, batch_size: int = 24) -> np.ndarray:
        if self.fail_next_embed:
            raise RuntimeError("Simulated Ollama connection error during embedding")
        values = list(texts)
        self.embed_calls.append(values)
        if not values:
            return np.empty((0, self.dim), dtype=np.float32)

        matrix = []
        for text in values:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [float(b) / 255.0 for b in h[: self.dim]]
            norm = np.linalg.norm(vec) or 1.0
            matrix.append([v / norm for v in vec])
        return np.array(matrix, dtype=np.float32)

    def chat(self, model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        return "Simulated answer [S1]"


def make_service(tmp_path: Path, client: TrackingFakeClient | None = None, embed_model: str = "fake-embed") -> tuple[RAGService, TrackingFakeClient]:
    client = client or TrackingFakeClient()
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="fake-chat",
        embed_model=embed_model,
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "index",
        top_k=4,
        min_relevance=0.01,
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    service = RAGService(settings, client=client)
    return service, client


# Test 1 & 2: Initial indexing and no changes on subsequent runs
def test_initial_indexing_and_no_changes_generates_zero_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    doc1 = service.settings.data_dir / "doc1.txt"
    doc2 = service.settings.data_dir / "doc2.txt"
    doc1.write_text("Company travel policy allows business class for flights over 6 hours.", encoding="utf-8")
    doc2.write_text("Expense claims must be submitted within 30 days of return.", encoding="utf-8")

    # Initial run
    result1 = service.update_index()
    assert result1.status == "initial"
    assert result1.total_files == 2
    assert result1.total_chunks == 2
    assert result1.new_embeddings == 2
    assert len(client.embed_calls) == 1

    # Second run without changes
    client.embed_calls.clear()
    result2 = service.update_index()
    assert result2.status == "up_to_date"
    assert result2.new_embeddings == 0
    assert result2.reused_embeddings == 0
    assert len(client.embed_calls) == 0  # Zero calls to embedder!
    assert "already up to date" in result2.message.lower()


# Test 3: Adding one file embeds only new content
def test_adding_one_file_embeds_only_new_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    doc1 = service.settings.data_dir / "doc1.txt"
    doc1.write_text("First document content about billing procedures.", encoding="utf-8")

    res1 = service.update_index()
    assert res1.total_files == 1

    # Add second file
    doc2 = service.settings.data_dir / "doc2.txt"
    doc2.write_text("Second document content about customer support procedures.", encoding="utf-8")

    client.embed_calls.clear()
    res2 = service.update_index()
    assert res2.status == "updated"
    assert len(res2.new_files) == 1
    assert "doc2.txt" in res2.new_files[0]
    assert res2.new_embeddings == 1
    assert res2.reused_embeddings == 1  # doc1 reused
    assert len(client.embed_calls) == 1
    assert any("customer support" in call_text for call_text in client.embed_calls[0])
    assert not any("billing procedures" in call_text for call_text in client.embed_calls[0])


# Test 4: Modifying one file does not re-embed unrelated files
def test_modifying_one_file_does_not_reembed_unrelated_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    file_a = service.settings.data_dir / "file_a.txt"
    file_b = service.settings.data_dir / "file_b.txt"
    file_a.write_text("Unrelated document content alpha.", encoding="utf-8")
    file_b.write_text("Original content for document beta.", encoding="utf-8")

    service.update_index()

    # Modify file_b only
    file_b.write_text("Modified content for document beta with updated details.", encoding="utf-8")
    client.embed_calls.clear()

    res = service.update_index()
    assert res.status == "updated"
    assert len(res.modified_files) == 1
    assert "file_b.txt" in res.modified_files[0]
    assert len(res.unchanged_files) == 1
    assert "file_a.txt" in res.unchanged_files[0]
    # Verify file_a was NOT sent to embedder
    all_embedded = [item for batch in client.embed_calls for item in batch]
    assert any("beta" in item for item in all_embedded)
    assert not any("alpha" in item for item in all_embedded)


# Test 5: Unchanged chunks in a modified file reuse embeddings safely
def test_unchanged_chunks_in_modified_file_reuse_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)

    # Create a document long enough to produce multiple chunks (each chunk <= 260 words)
    part1 = " ".join(["apple"] * 270)
    part2 = " ".join(["banana"] * 270)
    file_c = service.settings.data_dir / "file_c.txt"
    file_c.write_text(f"{part1}\n\n{part2}", encoding="utf-8")

    res1 = service.update_index()
    initial_chunks = res1.total_chunks
    assert initial_chunks >= 2

    # Now modify only the end (banana -> cherry), keeping part 1 completely identical
    part2_mod = " ".join(["cherry"] * 270)
    file_c.write_text(f"{part1}\n\n{part2_mod}", encoding="utf-8")

    client.embed_calls.clear()
    res2 = service.update_index()
    assert res2.status == "updated"
    assert res2.reused_embeddings >= 1  # part 1 reused!
    assert res2.new_embeddings >= 1     # part 2 new!

    # Verify that changing text AFTER character 80 does NOT falsely reuse embeddings
    # (Checking against chunk ID collision / text[:80] truncation bug)
    prefix_85_chars = "x" * 85
    file_d = service.settings.data_dir / "collision_test.txt"
    file_d.write_text(f"{prefix_85_chars} ORIGINAL_SUFFIX", encoding="utf-8")
    service.update_index()

    client.embed_calls.clear()
    # Same first 85 chars, different suffix
    file_d.write_text(f"{prefix_85_chars} CHANGED_SUFFIX", encoding="utf-8")
    res3 = service.update_index()
    assert res3.status == "updated"
    assert res3.new_embeddings == 1  # Must re-embed despite identical first 80 characters!


# Test 6: Deleting a source removes its searchable chunks
def test_deleting_source_removes_searchable_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    file_keep = service.settings.data_dir / "keep.txt"
    file_delete = service.settings.data_dir / "delete.txt"
    file_keep.write_text("Permanent guidelines on office equipment.", encoding="utf-8")
    file_delete.write_text("Confidential project codename Zebra details.", encoding="utf-8")

    service.update_index()
    assert any("Zebra" in c.text for c in service._chunks)

    # Delete file
    file_delete.unlink()
    res = service.update_index()
    assert res.status == "updated"
    assert len(res.deleted_files) == 1
    assert "delete.txt" in res.deleted_files[0]
    assert not any("Zebra" in c.text for c in service._chunks)


# Test 7: Changing a filename or moving a file does not create duplicate indexed copies
def test_moving_file_does_not_create_duplicate_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)

    # Subcase 1: Moving a file to another folder with same filename keeps title and reuses embeddings
    sub1 = service.settings.data_dir / "folder_alpha"
    sub2 = service.settings.data_dir / "folder_beta"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)
    file1 = sub1 / "guide.txt"
    file1.write_text("Unique text about quarterly sales goals.", encoding="utf-8")

    res1 = service.update_index()
    assert res1.total_files == 1
    assert res1.total_chunks == 1

    # Move file1 to sub2
    file1_moved = sub2 / "guide.txt"
    file1.rename(file1_moved)

    client.embed_calls.clear()
    res2 = service.update_index()
    assert res2.status == "updated"
    assert len(res2.deleted_files) == 1
    assert len(res2.new_files) == 1
    assert res2.total_files == 1
    assert res2.total_chunks == 1
    # Because content and title are identical, embedding was reused!
    assert res2.new_embeddings == 0
    assert res2.reused_embeddings == 1

    # Subcase 2: Renaming file to a new name updates title, but still never leaves duplicate copies
    file_renamed = sub2 / "renamed_guide.txt"
    file1_moved.rename(file_renamed)

    client.embed_calls.clear()
    res3 = service.update_index()
    assert res3.status == "updated"
    assert len(res3.deleted_files) == 1
    assert len(res3.new_files) == 1
    assert res3.total_files == 1
    assert res3.total_chunks == 1
    assert res3.new_embeddings == 1  # Title changed, so re-embedded with new title
    assert len(service._chunks) == 1  # No duplicate chunks!


# Test 8: Two files with the same name in different folders remain separate sources
def test_same_filename_in_different_folders_remain_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    sub1 = service.settings.data_dir / "dept_engineering"
    sub2 = service.settings.data_dir / "dept_sales"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    (sub1 / "handbook.txt").write_text("Engineering handbook: git flow and CI/CD pipelines.", encoding="utf-8")
    (sub2 / "handbook.txt").write_text("Sales handbook: CRM guidelines and commission structure.", encoding="utf-8")

    res = service.update_index()
    assert res.total_files == 2
    assert res.total_chunks == 2
    tracked = service.store.load_sources()
    assert tracked is not None
    assert len(tracked) == 2


# Test 9: Overlapping source roots do not duplicate chunks
def test_overlapping_source_roots_do_not_duplicate_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    sub = service.settings.data_dir / "nested"
    sub.mkdir(parents=True)
    (sub / "nested_doc.txt").write_text("Content in nested folder.", encoding="utf-8")

    # Mock source_roots to return both data_dir AND nested subfolder
    monkeypatch.setattr(service, "source_roots", lambda: [service.settings.data_dir, sub])

    res = service.update_index()
    assert res.total_files == 1
    assert res.total_chunks == 1
    assert len(service._chunks) == 1


# Test 10: Model or chunking configuration changes trigger appropriate invalidation or rebuilding
def test_model_configuration_change_triggers_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path, embed_model="model-v1")
    (service.settings.data_dir / "doc.txt").write_text("Knowledge base text.", encoding="utf-8")

    service.update_index()
    assert service.manifest.embedding_model == "model-v1"

    # Switch embedding model in settings
    service.settings = Settings(
        ollama_base_url=service.settings.ollama_base_url,
        chat_model=service.settings.chat_model,
        embed_model="model-v2-upgraded",
        data_dir=service.settings.data_dir,
        index_dir=service.settings.index_dir,
        top_k=service.settings.top_k,
        min_relevance=service.settings.min_relevance,
    )

    res = service.update_index()
    assert res.status == "rebuilt"
    assert "Embedding model changed" in res.message
    assert service.manifest.embedding_model == "model-v2-upgraded"


# Test 11: A parser error does not corrupt the existing index
def test_parser_error_does_not_corrupt_existing_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    good = service.settings.data_dir / "good.txt"
    good.write_text("Good document that parses fine.", encoding="utf-8")
    service.update_index()
    assert service.store.exists()
    assert len(service._chunks) == 1

    # Add a file that causes a parse error
    bad = service.settings.data_dir / "bad.docx"
    bad.write_text("not a real docx", encoding="utf-8")  # python-docx will fail

    res = service.update_index()
    # The good chunk is still in the active index
    assert len(service._chunks) == 1
    assert "bad.docx" in res.failed_files
    # Active index remains valid
    chunks, embeddings, manifest = service.store.load()
    assert len(chunks) == 1
    assert chunks[0].title == "good"


# Test 12: An embedding failure preserves the previous valid snapshot
def test_embedding_failure_preserves_previous_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    (service.settings.data_dir / "valid.txt").write_text("Valid text.", encoding="utf-8")
    service.update_index()

    # Now add new file, but make embedder fail
    (service.settings.data_dir / "new.txt").write_text("New text.", encoding="utf-8")
    client.fail_next_embed = True

    res = service.update_index()
    assert res.status == "failed"
    assert "previous valid index is still available" in res.message.lower()

    # Verify store load still works and has the previous 1 chunk
    chunks, embeddings, manifest = service.store.load()
    assert len(chunks) == 1
    assert chunks[0].title == "valid"


# Test 13: Reloading the application preserves the index and source-tracking state
def test_reloading_application_preserves_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service1, client = make_service(tmp_path)
    (service1.settings.data_dir / "doc.txt").write_text("Persistent content.", encoding="utf-8")
    service1.update_index()

    # Fresh service instance pointing to same directories
    service2 = RAGService(service1.settings, client=client)
    assert service2.store.exists()
    assert service2.manifest.chunk_count == 1

    client.embed_calls.clear()
    res = service2.update_index()
    assert res.status == "up_to_date"
    assert len(client.embed_calls) == 0


# Test 14: Media processing remains cached for unchanged files
def test_media_processing_remains_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "true")
    service, client = make_service(tmp_path)

    # Mock client.vision to track vision calls
    vision_calls = []

    def mock_vision(model, prompt, img_bytes):
        vision_calls.append(img_bytes)
        return "OCR TEXT: Emergency helpline number 112."

    client.vision = mock_vision

    img_path = service.settings.data_dir / "notice.png"
    img = Image.new("RGB", (100, 100), "white")
    img.save(img_path)

    res1 = service.update_index()
    assert len(vision_calls) == 1

    # Second run without changes: vision OCR must NOT be called again
    vision_calls.clear()
    client.embed_calls.clear()
    res2 = service.update_index()
    assert len(vision_calls) == 0
    assert len(client.embed_calls) == 0
    assert res2.status == "up_to_date"


# Test 15: Changing a relevant sidecar refreshes searchable metadata without regenerating embeddings
def test_sidecar_change_refreshes_metadata_without_reembedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    doc = service.settings.data_dir / "policy.txt"
    doc.write_text("Standard underwriting policy terms.", encoding="utf-8")

    sidecar = service.settings.data_dir / ".policy.txt.confluence.json"
    sidecar.write_text(
        json.dumps({
            "confluence_attachment": True,
            "attachment_id": "100",
            "attachment_version": "1",
            "page_id": "500",
            "url": "http://example.com/v1",
        }),
        encoding="utf-8",
    )

    res1 = service.update_index()
    assert service._chunks[0].source_url == "http://example.com/v1"

    # Modify sidecar URL to v2 without changing document text
    sidecar.write_text(
        json.dumps({
            "confluence_attachment": True,
            "attachment_id": "100",
            "attachment_version": "2",
            "page_id": "500",
            "url": "http://example.com/v2",
        }),
        encoding="utf-8",
    )

    client.embed_calls.clear()
    res2 = service.update_index()
    assert res2.status == "updated"
    assert service._chunks[0].source_url == "http://example.com/v2"
    # Embedding text was identical: zero calls to embedder!
    assert res2.new_embeddings == 0
    assert res2.reused_embeddings == 1
    assert len(client.embed_calls) == 0


# Test 16: Search after an incremental update finds newly added information and no longer retrieves deleted information
def test_search_after_incremental_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    f1 = service.settings.data_dir / "quantum.txt"
    f1.write_text("Quantum encryption protocol key exchange details.", encoding="utf-8")

    service.update_index()
    results1 = service.retrieve("quantum encryption")
    assert any("quantum" in r.chunk.text.lower() for r in results1)

    # Add fusion and delete quantum
    f2 = service.settings.data_dir / "fusion.txt"
    f2.write_text("Tokamak fusion reactor containment procedure.", encoding="utf-8")
    f1.unlink()

    service.update_index()
    results2 = service.retrieve("fusion reactor")
    assert any("fusion" in r.chunk.text.lower() for r in results2)

    results3 = service.retrieve("quantum encryption")
    assert not any("quantum" in r.chunk.text.lower() for r in results3)


# Test 17: An update containing additions, modifications, and deletions gives correct statistics
def test_combined_additions_modifications_deletions_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    service, client = make_service(tmp_path)
    fa = service.settings.data_dir / "a.txt"
    fb = service.settings.data_dir / "b.txt"
    fc = service.settings.data_dir / "c.txt"
    fa.write_text("Text A", encoding="utf-8")
    fb.write_text("Text B initial", encoding="utf-8")
    fc.write_text("Text C initial", encoding="utf-8")

    service.update_index()

    # Operations:
    # 1. fa remains untouched (unchanged)
    # 2. fb modified
    fb.write_text("Text B revised", encoding="utf-8")
    # 3. fc deleted
    fc.unlink()
    # 4. fd added
    fd = service.settings.data_dir / "d.txt"
    fd.write_text("Text D new", encoding="utf-8")

    res = service.update_index()
    assert res.status == "updated"
    assert len(res.new_files) == 1
    assert "d.txt" in res.new_files[0]
    assert len(res.modified_files) == 1
    assert "b.txt" in res.modified_files[0]
    assert len(res.deleted_files) == 1
    assert "c.txt" in res.deleted_files[0]
    assert len(res.unchanged_files) == 1
    assert "a.txt" in res.unchanged_files[0]
    assert res.total_files == 3


# Test 18: An existing index from the previous application version can be migrated or safely rebuilt
def test_legacy_index_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MEDIA_AUTO_PROCESS_ON_INDEX", "false")
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "index"
    data_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    doc = data_dir / "legacy_doc.txt"
    doc.write_text("Legacy document content.", encoding="utf-8")

    # Manually write a v1 index (chunks.json, embeddings.npy, manifest.json WITHOUT sources.json)
    chunk = Chunk(
        chunk_id="legacy123",
        text="Legacy document content.",
        title="legacy_doc",
        source_path="legacy_doc.txt",
        source_type="txt",
    )
    store = IndexStore(index_dir)
    raw_chunks_path = index_dir / "chunks.json"
    raw_embs_path = index_dir / "embeddings.npy"
    raw_man_path = index_dir / "manifest.json"

    raw_chunks_path.write_text(json.dumps([chunk.to_dict()]), encoding="utf-8")
    np.save(raw_embs_path, np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32))
    raw_man_path.write_text(
        json.dumps({
            "created_at": "2026-01-01T00:00:00Z",
            "embedding_model": "fake-embed",
            "chunk_count": 1,
            "source_count": 1,
            "errors": [],
        }),
        encoding="utf-8",
    )

    assert not store.has_sources()  # Truly a legacy v1 index

    client = TrackingFakeClient()
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        chat_model="fake-chat",
        embed_model="fake-embed",
        data_dir=data_dir,
        index_dir=index_dir,
        top_k=4,
        min_relevance=0.01,
    )
    service = RAGService(settings, client=client)

    # Calling update_index should migrate without error
    res = service.update_index()
    assert store.has_sources()  # Now upgraded with sources.json
    assert res.total_files == 1
    assert res.total_chunks == 1
    # Document content was identical, so zero new embeddings needed!
    assert res.new_embeddings == 0
