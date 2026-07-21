# AI Knowledge Base Assistant

A polished Streamlit application that lets you upload PDF and TXT documents, store their content in ChromaDB, and chat with a Gemini-powered assistant grounded in the uploaded knowledge base.

## Features
- Upload multiple PDF and TXT files
- Extract and chunk document text
- Store embeddings in ChromaDB
- Retrieve top relevant chunks for RAG
- Maintain chat history
- Display source references and confidence indicators
- Export chat history

## Tech Stack
- Streamlit
- Python
- Google Gemini API
- ChromaDB
- pypdf

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

## Folder Structure
```text
Project-1-Knowledge-Base-Chatbot/
app.py
requirements.txt
README.md
.env.example
data/
chroma_db/
src/
pdf_reader.py
text_splitter.py
vector_store.py
rag.py
utils.py
```

## Future Improvements
- Add document deletion from the UI
- Add richer source highlighting
- Support more document formats
