"""Main Streamlit application entry point."""

import streamlit as st

from common import configure_logging, get_settings


def main():
    """Run the Streamlit application."""
    # Configure logging
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        format=settings.log_format,
    )

    # Page configuration
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("📚 RAG System")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        ["💬 Chat", "📤 Upload Documents", "📋 Manage Documents"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **About**

        This RAG system uses:
        - DeepSeek OCR for text extraction
        - LangGraph for agentic workflows
        - ChromaDB for vector storage
        - OpenAI GPT for Q&A
        """
    )

    # Route to selected page
    if page == "💬 Chat":
        from app.pages.chat import render_chat_page
        render_chat_page()
    elif page == "📤 Upload Documents":
        from app.pages.upload import render_upload_page
        render_upload_page()
    elif page == "📋 Manage Documents":
        from app.pages.upload import render_manage_page
        render_manage_page()


if __name__ == "__main__":
    main()
