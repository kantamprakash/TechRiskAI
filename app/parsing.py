"""Extract text from uploaded BRD/SRD files and split it into LLM-sized chunks."""
from __future__ import annotations

import io
import re
from pathlib import Path

from .models import Chunk, SourceDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        import pymupdf

        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)
    if ext == ".docx":
        return _docx_text(data)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


def _docx_text(data: bytes) -> str:
    """Paragraphs and tables in document order; headings become markdown '#' lines."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(io.BytesIO(data))
    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            if style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                lines.append("#" * min(int(level), 6) + " " + text)
            else:
                lines.append(text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_document(filename: str, data: bytes, doc_type: str) -> SourceDocument:
    text = extract_text(filename, data)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(f"No text could be extracted from '{filename}' (scanned PDF? OCR is not supported yet).")
    return SourceDocument(name=filename, doc_type=doc_type, text=text)


_HEADING_RE = re.compile(
    r"^(#{1,6}\s+.+|\d+(\.\d+)*\.?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9 &/\-]{4,60})$"
)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body) pairs using markdown, numbered or ALL-CAPS headings."""
    sections: list[tuple[str, list[str]]] = [("(start)", [])]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) < 100 and _HEADING_RE.match(stripped):
            sections.append((stripped.lstrip("# ").strip(), [line]))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def chunk_document(doc: SourceDocument, chunk_chars: int, overlap: int) -> list[Chunk]:
    """Pack whole sections into chunks; split oversized sections with overlap."""
    chunks: list[Chunk] = []
    buf, buf_section = "", ""

    def flush():
        nonlocal buf, buf_section
        if buf.strip():
            chunks.append(Chunk(doc, len(chunks), buf_section, buf.strip()))
        buf, buf_section = "", ""

    for heading, body in _sections(doc.text):
        if len(body) > chunk_chars:
            flush()
            step = max(1, chunk_chars - overlap)
            for start in range(0, len(body), step):
                chunks.append(Chunk(doc, len(chunks), heading, body[start:start + chunk_chars]))
                if start + chunk_chars >= len(body):
                    break
            continue
        if len(buf) + len(body) + 2 > chunk_chars:
            flush()
        if not buf_section:
            buf_section = heading
        elif heading not in buf_section:
            buf_section = f"{buf_section}; {heading}"[:200]
        buf += body + "\n\n"
    flush()
    return chunks
