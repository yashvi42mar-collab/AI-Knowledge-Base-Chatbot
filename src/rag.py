def build_rag_prompt(question, context_chunks):
    """Build a compact retrieval-augmented prompt for Gemini."""
    context_text = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    return (
        "You are an AI Knowledge Base Assistant.\n\n"
        "Answer ONLY from the retrieved context.\n"
        'If the answer is unavailable, say: "I couldn\'t find that information in the uploaded documents."\n'
        "Never hallucinate.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}"
    )
