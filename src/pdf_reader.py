import io

from pypdf import PdfReader


def read_pdf(file):
    """Extract and return all text from a PDF file or bytes input."""
    if hasattr(file, "read"):
        data = file.read()
    else:
        data = file

    reader = PdfReader(io.BytesIO(data))
    text_pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(text for text in text_pages if text).strip()
