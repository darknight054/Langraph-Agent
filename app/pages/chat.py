"""Chat interface page."""

import streamlit as st

from agents import ChatSession


def init_session_state():
    """Initialize session state for chat."""
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = ChatSession()

    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_page():
    """Render the chat interface page."""
    init_session_state()

    st.title("💬 Chat with Documents")
    st.markdown("Ask questions about your ingested documents.")

    # Check if documents are ingested
    try:
        from ingestion import IngestionPipeline
        pipeline = IngestionPipeline()
        doc_count = pipeline.vectorstore.count()

        if doc_count == 0:
            st.warning(
                "⚠️ No documents ingested yet. "
                "Please upload documents first using the Upload page."
            )
            return
        else:
            st.caption(f"📊 {doc_count} chunks available from ingested documents")

    except Exception as e:
        st.error(f"Error checking documents: {e}")

    st.markdown("---")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['document']}**, Page {source['page']}"
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_session.chat(prompt)

                    st.markdown(response.content)

                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "sources": response.sources,
                    })

                    # Show sources
                    if response.sources:
                        with st.expander("📚 Sources", expanded=False):
                            for source in response.sources:
                                st.markdown(
                                    f"- **{source['document']}**, Page {source['page']}"
                                )

                except Exception as e:
                    st.error(f"Error generating response: {e}")

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_session = ChatSession()
            st.rerun()
