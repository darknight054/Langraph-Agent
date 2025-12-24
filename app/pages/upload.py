"""Document upload and management pages."""

import tempfile
from pathlib import Path

import streamlit as st

from ingestion import IngestionPipeline


def render_upload_page():
    """Render the document upload page."""
    st.title("📤 Upload Documents")
    st.markdown("Upload PDF documents to ingest into the RAG system.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to process with OCR and store in the vector database.",
    )

    # Page range options
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input(
            "Start Page (optional)",
            min_value=1,
            value=1,
            help="First page to process (1-indexed)",
        )
    with col2:
        end_page = st.number_input(
            "End Page (optional)",
            min_value=0,
            value=0,
            help="Last page to process (0 = all pages)",
        )

    # Chunking strategy option
    strategy = st.radio(
        "Chunking Strategy",
        ["contextual", "semantic"],
        index=0,
        horizontal=True,
        help="Contextual: Better retrieval (uses LLM, slower). Semantic: Fast baseline.",
    )

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button("🚀 Start Ingestion", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = Path(tmp_file.name)

            # Progress container
            progress_container = st.empty()
            status_container = st.empty()

            def update_progress(stage: str, current: int, total: int):
                stage_names = {
                    "extracting_pages": "Extracting pages from PDF",
                    "ocr": "Running OCR on pages",
                    "chunking": "Chunking document",
                    "storing": "Storing in vector database",
                }
                stage_display = stage_names.get(stage, stage)

                if total > 0:
                    progress = current / total
                    progress_container.progress(
                        progress,
                        text=f"{stage_display}: {current}/{total}",
                    )
                else:
                    status_container.info(f"{stage_display}...")

            try:
                pipeline = IngestionPipeline(
                    chunking_strategy=strategy,
                )

                result = pipeline.ingest(
                    pdf_path=tmp_path,
                    start_page=start_page if start_page > 0 else None,
                    end_page=end_page if end_page > 0 else None,
                    progress_callback=update_progress,
                )

                progress_container.empty()
                status_container.empty()

                if result["success"]:
                    st.success("✅ Document ingested successfully!")
                    st.markdown(f"**Document ID:** `{result['document_id']}`")
                    st.markdown(f"**Pages processed:** {result['pages_processed']}")
                    st.markdown(f"**Chunks created:** {result['chunks_created']}")
                else:
                    st.error(f"❌ Ingestion failed: {result['error']}")

            except Exception as e:
                progress_container.empty()
                status_container.empty()
                st.error(f"❌ Error during ingestion: {e}")

            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


def render_manage_page():
    """Render the document management page."""
    st.title("📋 Manage Documents")
    st.markdown("View and manage ingested documents.")

    try:
        pipeline = IngestionPipeline()
        documents = pipeline.list_documents()

        if not documents:
            st.info("No documents ingested yet.")
            return

        st.markdown(f"**Total documents:** {len(documents)}")
        st.markdown(f"**Total chunks:** {pipeline.vectorstore.count()}")
        st.markdown("---")

        for doc in documents:
            with st.expander(
                f"📄 {doc['document_name']} ({doc['chunk_count']} chunks)",
                expanded=False,
            ):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.markdown(f"**Document ID:** `{doc['document_id']}`")

                with col2:
                    st.markdown(f"**Chunks:** {doc['chunk_count']}")

                with col3:
                    st.markdown(f"**Pages:** {doc['page_count']}")

                # Delete button
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{doc['document_id']}",
                    type="secondary",
                ):
                    deleted = pipeline.delete_document(doc["document_id"])
                    st.success(f"Deleted {deleted} chunks.")
                    st.rerun()

    except Exception as e:
        st.error(f"Error loading documents: {e}")

    st.markdown("---")

    # Search test
    st.subheader("🔍 Test Search")
    query = st.text_input("Enter a search query to test retrieval:")

    if query:
        try:
            results = pipeline.query(query, n_results=5)

            if results:
                st.markdown(f"**Found {len(results)} results:**")

                for i, result in enumerate(results, 1):
                    metadata = result.get("metadata", {})
                    distance = result.get("distance", 0)

                    with st.expander(
                        f"Result {i}: {metadata.get('document_name', 'Unknown')} "
                        f"(Page {metadata.get('page_number', 'N/A')}, "
                        f"Distance: {distance:.4f})",
                        expanded=(i == 1),
                    ):
                        st.markdown(result["text"][:500] + "...")
            else:
                st.info("No results found.")

        except Exception as e:
            st.error(f"Search error: {e}")
