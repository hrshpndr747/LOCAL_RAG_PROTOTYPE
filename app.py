from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from local_rag import RAGService, Settings
from local_rag.ollama_client import OllamaClient, OllamaError
from local_rag.sql_tool import execute_read_only
from local_rag.table_parsing import TABLE_PARSER_VERSION


st.set_page_config(page_title="Local RAG Prototype", page_icon="🔎", layout="wide")
st.title("Local RAG knowledge assistant")
st.caption("Local Ollama model • hybrid retrieval • evidence citations • fictional demo data")

settings = Settings.from_env()


@st.cache_resource
def get_service(current_settings: Settings) -> RAGService:
    return RAGService(current_settings)


service = get_service(settings)

with st.sidebar:
    st.header("Prototype status")
    ollama = OllamaClient(settings.ollama_base_url)
    if ollama.is_running():
        st.success("Ollama is running")
        try:
            installed = ollama.installed_models()
            st.caption(f"Chat: {settings.chat_model}\n\nEmbedding: {settings.embed_model}")
            if not any(name.startswith(settings.chat_model) for name in installed):
                st.warning(f"Pull chat model: ollama pull {settings.chat_model}")
            if not any(name.startswith(settings.embed_model) for name in installed):
                st.warning(f"Pull embedding model: ollama pull {settings.embed_model}")
        except OllamaError as exc:
            st.warning(str(exc))
    else:
        st.error("Ollama is not running")
        st.code("ollama serve", language="bash")

    if service.store.exists():
        try:
            manifest = service.manifest
            st.info(f"Index: {manifest.chunk_count} chunks from {manifest.source_count} files")
            st.caption(f"Table parser: {TABLE_PARSER_VERSION}")
        except Exception as exc:
            st.warning(str(exc))
    else:
        st.warning("Index has not been built")

    if st.button("Update Index (Incremental)", type="primary", use_container_width=True, help="Scan local files for changes and re-embed only new or modified documents."):
        try:
            with st.spinner("Scanning files for changes and incrementally updating index…"):
                result = service.update_index()
            if result.status == "up_to_date":
                st.info(result.message)
            elif result.status in {"updated", "initial", "rebuilt"}:
                st.success(result.message)
                with st.expander("Update statistics", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("New files", len(result.new_files))
                        st.metric("Modified files", len(result.modified_files))
                        st.metric("Deleted files", len(result.deleted_files))
                        st.metric("Unchanged files", len(result.unchanged_files))
                    with col2:
                        st.metric("New embeddings", result.new_embeddings)
                        st.metric("Reused embeddings", result.reused_embeddings)
                        st.metric("Total chunks", result.total_chunks)
                        st.metric("Elapsed time", f"{result.elapsed_seconds:.2f}s")
                    if result.failed_files:
                        st.error("Failed files:\n" + "\n".join(f"- {k}: {v}" for k, v in result.failed_files.items()))
                    if result.errors:
                        st.warning("Errors / warnings:\n" + "\n".join(result.errors[:5]))
            elif result.status == "failed":
                st.error(result.message)
                if result.errors:
                    st.warning("\n".join(result.errors[:5]))
            st.rerun()
        except Exception as exc:
            st.error(f"Update failed. The previous valid index is still available. Details: {exc}")

    if st.button("Full Rebuild Index", use_container_width=True, help="Re-parse and re-embed every document from scratch."):
        try:
            with st.spinner("Processing OCR/video, reading files, and rebuilding complete index…"):
                manifest = service.build_index()
            st.success(f"Full index rebuilt: {manifest.chunk_count} chunks from {manifest.source_count} files.")
            if manifest.errors:
                st.warning("Some files were skipped:\n" + "\n".join(manifest.errors[:5]))
            st.rerun()
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")

    st.divider()
    with st.expander("Read-only SQLite explorer"):
        database = settings.data_dir / "02_Attachments" / "support_operations.db"
        sql = st.text_area("SELECT query", "SELECT status, COUNT(*) AS total FROM service_requests GROUP BY status ORDER BY total DESC")
        if st.button("Run safe query", use_container_width=True):
            try:
                result = execute_read_only(database, sql)
                st.dataframe([dict(zip(result.columns, row)) for row in result.rows], use_container_width=True)
                if result.truncated:
                    st.caption("Result was limited to 200 rows.")
            except Exception as exc:
                st.error(str(exc))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Evidence used"):
                for source in message["sources"]:
                    st.markdown(source)

question = st.chat_input("Ask a question about the demo data")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not service.store.exists():
            response = "Build the index from the sidebar first."
            st.warning(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            history = [
                {"role": item["role"], "content": item["content"]}
                for item in st.session_state.messages[:-1]
                if item["role"] in {"user", "assistant"}
            ]
            try:
                with st.spinner("Checking the sources…"):
                    answer = service.answer(question, history)
                st.markdown(answer.text)
                source_lines = []
                for number, result in enumerate(answer.results, 1):
                    source = result.chunk
                    label = f"**[S{number}] {html.escape(source.title)}** — `{html.escape(source.source_path)}`; {html.escape(source.locator)}"
                    if source.source_url:
                        label += f" — [open source]({source.source_url})"
                    label += f"  \nRelevance: {result.score:.3f}"
                    source_lines.append(label)
                if source_lines:
                    with st.expander("Evidence used", expanded=False):
                        for line in source_lines:
                            st.markdown(line)
                st.session_state.messages.append({"role": "assistant", "content": answer.text, "sources": source_lines})
            except Exception as exc:
                st.error(str(exc))
