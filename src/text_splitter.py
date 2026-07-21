def split_text(text):
    """Split long text into overlapping chunks of roughly 500 characters."""
    if not text:
        return []

    chunk_size = 500
    overlap = 100
    step = chunk_size - overlap

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
