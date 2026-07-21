import os
import io

import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.pdf_reader import read_pdf
from src.text_splitter import split_text
from src.vector_store import (
    clear_collection,
    get_collection_stats,
    get_uploaded_documents,
    remove_document,
    refresh_knowledge_base,
    search_chunks,
    store_chunks,
)
from src.rag import build_rag_prompt
from src.utils import ensure_project_structure, get_doc_stats

load_dotenv()
ensure_project_structure()

st.set_page_config(page_title="AI Knowledge Base Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Knowledge Base Assistant")
st.caption("Upload PDFs or TXT files to build a searchable knowledge base and ask questions grounded in your documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


def render_sidebar():
    """Render the sidebar with upload, stats, and knowledge-base management."""
    with st.sidebar:
        st.header("📚 Knowledge Base")
        st.markdown("Upload documents to build a searchable assistant.")

        uploaded_files = st.file_uploader(
            "Upload one or more PDF/TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed = []

            for index, uploaded_file in enumerate(uploaded_files, start=1):
                status_text.text(f"📖 Reading documents... {index}/{len(uploaded_files)}")
                if uploaded_file.name.lower().endswith(".pdf"):
                    text = read_pdf(uploaded_file)
                elif uploaded_file.name.lower().endswith(".txt"):
                    text = uploaded_file.getvalue().decode("utf-8", errors="replace")
                else:
                    continue

                if not text.strip():
                    continue

                status_text.text(f"✂ Splitting text... {index}/{len(uploaded_files)}")
                chunks = split_text(text)

                status_text.text(f"🧠 Creating embeddings... {index}/{len(uploaded_files)}")
                stored_ids = store_chunks(chunks, source_name=uploaded_file.name, text=text, page_count=get_doc_stats(text)["pages"])

                status_text.text(f"📦 Saving to ChromaDB... {index}/{len(uploaded_files)}")
                processed.append(uploaded_file.name)
                progress_bar.progress(index / len(uploaded_files))

            progress_bar.empty()
            status_text.empty()

            if processed:
                st.success("✅ Knowledge Base Updated")
                st.session_state.uploaded_docs = list(dict.fromkeys(st.session_state.uploaded_docs + processed))
            else:
                st.warning("No readable text found in the uploaded files.")

        st.divider()
        st.subheader("📊 Knowledge Base Stats")
        stats = get_collection_stats()
        st.metric("Uploaded documents", stats["document_count"])
        st.metric("Stored chunks", stats["chunk_count"])

        documents = get_uploaded_documents()
        if documents:
            st.subheader("📄 Uploaded Documents")
            for document in documents:
                with st.expander(document.get("name", "Unknown")):
                    st.write(f"Chunks: {document.get('chunk_count', 0)}")
                    st.write(f"Pages: {document.get('page_count', 1)}")
                    if st.button(f"Remove {document.get('name', 'document')}", key=f"remove_{document['name']}"):
                        remove_document(document["name"])
                        st.success(f"Removed {document['name']}")
                        st.rerun()

        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared.")
            st.rerun()

        if st.button("Clear Knowledge Base", use_container_width=True):
            clear_collection()
            st.success("Knowledge Base cleared.")
            st.rerun()

        if st.button("Refresh Knowledge Base", use_container_width=True):
            refresh_knowledge_base()
            st.success("Knowledge Base refreshed.")
            st.rerun()

        st.divider()
        st.caption("Current AI model: gemini-3.1-flash-lite")


render_sidebar()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL = "gemini-3.1-flash-lite"


def render_chat_history():
    """Render the stored chat history on every rerun."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


render_chat_history()

st.markdown("---")

prompt = st.chat_input("Ask anything about your uploaded documents...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("🔍 Searching Knowledge Base..."):
            relevant_results = search_chunks(prompt, n_results=5)

        context_chunks = [item["text"] for item in relevant_results]
        context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

        rag_prompt = build_rag_prompt(prompt, context_chunks)

        with st.spinner("🤖 Generating Response..."):
            response = client.models.generate_content(model=MODEL, contents=rag_prompt)

        answer = response.text.strip()

        similarity_scores = [item.get("distance", 0.0) for item in relevant_results]
        if similarity_scores:
            average_distance = sum(similarity_scores) / len(similarity_scores)
            if average_distance < 0.3:
                confidence = "🟢 High"
            elif average_distance < 0.6:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"
        else:
            confidence = "🔴 Low"

    except Exception as exc:
        answer = f"I couldn't generate a response right now. Error: {exc}"
        confidence = "🔴 Low"

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)

    st.markdown("**Sources Used**")
    if relevant_results:
        for index, item in enumerate(relevant_results, start=1):
            metadata = item.get("metadata", {}) or {}
            source_name = metadata.get("source_name", "Unknown")
            chunk_number = metadata.get("chunk_number", index)
            st.caption(f"{index}. {source_name} • Chunk {chunk_number}")
    else:
        st.caption("No sources available.")

    st.markdown(f"**Confidence:** {confidence}")

    chat_export = "\n\n".join(
        f"{message['role'].title()}: {message['content']}" for message in st.session_state.messages
    )
    st.download_button(
        label="Download Chat as TXT",
        data=chat_export,
        file_name="chat_export.txt",
        mime="text/plain",
    )