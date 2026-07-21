import json
import os
from datetime import datetime

import chromadb


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "knowledge_base"
METADATA_PATH = os.path.join(DB_PATH, "documents_metadata.json")


client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def _load_documents_metadata():
    """Load metadata for uploaded documents from disk."""
    if not os.path.exists(METADATA_PATH):
        return []

    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_documents_metadata(documents):
    """Persist document metadata to disk."""
    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(documents, handle, indent=2)


def _get_document_ids_for_source(source_name):
    """Fetch the stored chunk IDs for a given document name."""
    if not source_name:
        return []

    all_ids = collection.get(include=["metadatas"])["ids"]
    metadatas = collection.get(include=["metadatas"])["metadatas"]

    matching_ids = []
    for chunk_id, metadata in zip(all_ids, metadatas):
        if metadata and metadata.get("source_name") == source_name:
            matching_ids.append(chunk_id)

    return matching_ids


def store_chunks(chunks, source_name, text="", page_count=1):
    """Store text chunks, avoiding duplicates for the same document name."""
    if not chunks:
        return []

    existing_documents = _load_documents_metadata()
    if any(document["name"] == source_name for document in existing_documents):
        return []

    ids = [f"{source_name.replace(' ', '_')}_{index}" for index in range(len(chunks))]
    metadatas = []
    for index, chunk in enumerate(chunks):
        metadatas.append(
            {
                "source_name": source_name,
                "chunk_number": index + 1,
                "page_number": page_count,
                "stored_at": datetime.utcnow().isoformat(),
            }
        )

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    existing_documents.append(
        {
            "name": source_name,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "text": text,
            "stored_at": datetime.utcnow().isoformat(),
            "chunk_ids": ids,
        }
    )
    _save_documents_metadata(existing_documents)
    return ids


def search_chunks(query, n_results=5):
    """Search the collection for the most relevant text chunks."""
    if not query:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "text": document,
                "metadata": metadata or {},
                "distance": distance or 0.0,
            }
        )

    return retrieved


def get_uploaded_documents():
    """Return the list of stored documents metadata."""
    return _load_documents_metadata()


def get_collection_stats():
    """Return simple statistics about the collection."""
    metadata = _load_documents_metadata()
    all_ids = collection.get(include=["metadatas"])["ids"]
    return {
        "document_count": len(metadata),
        "chunk_count": len(all_ids),
        "ready": True,
    }


def clear_collection():
    """Clear all stored data from ChromaDB and metadata files."""
    stored_ids = collection.get(include=["metadatas"])["ids"]
    if stored_ids:
        collection.delete(ids=stored_ids)
    _save_documents_metadata([])


def remove_document(source_name):
    """Remove a single document and its related chunks."""
    matching_ids = _get_document_ids_for_source(source_name)
    if matching_ids:
        collection.delete(ids=matching_ids)

    documents = _load_documents_metadata()
    updated_documents = [doc for doc in documents if doc.get("name") != source_name]
    _save_documents_metadata(updated_documents)


def refresh_knowledge_base():
    """Rebuild the current knowledge base from the stored document text."""
    from src.text_splitter import split_text

    documents = _load_documents_metadata()
    if not documents:
        return []

    clear_collection()

    rebuilt_ids = []
    for document in documents:
        text = document.get("text", "")
        if not text:
            continue
        chunks = split_text(text)
        rebuilt_ids.extend(store_chunks(chunks, source_name=document["name"], text=text, page_count=document.get("page_count", 1)))

    return rebuilt_ids
