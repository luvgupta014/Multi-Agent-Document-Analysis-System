"""
Document loading and chunking.

Supports .txt, .pdf, and .docx files. Chunking uses sentence-aware splitting
with configurable overlap so context is preserved across chunk boundaries.
"""
from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class Document:

    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content.strip()
        self.metadata = metadata or {}
        self.doc_id = self.metadata.get("id", str(hash(content)))

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.doc_id}, content='{preview}')"

    def __len__(self):
        return len(self.content)


class DocumentLoader:

    @staticmethod
    def recursive_chunk_text(
        text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """
        Recursive character-level chunking.

        Tries to split on paragraph breaks first, then sentences, then words,
        then characters — whatever keeps each chunk under ``chunk_size``.
        This mirrors LangChain's RecursiveCharacterTextSplitter logic without
        the dependency.
        """
        separators = ["\n\n", "\n", ". ", " ", ""]

        def _split(txt: str, seps: List[str]) -> List[str]:
            if not txt.strip():
                return []
            if len(txt) <= chunk_size or not seps:
                return [txt]

            sep = seps[0]
            parts = txt.split(sep) if sep else list(txt)
            chunks: List[str] = []
            current: List[str] = []
            current_len = 0

            for part in parts:
                part_len = len(part) + len(sep)
                if current_len + part_len > chunk_size and current:
                    chunk_text = sep.join(current)
                    # recursively refine if still too big
                    chunks.extend(_split(chunk_text, seps[1:]))
                    # carry overlap
                    overlap_parts: List[str] = []
                    ol = 0
                    for p in reversed(current):
                        if ol + len(p) > overlap:
                            break
                        overlap_parts.insert(0, p)
                        ol += len(p) + len(sep)
                    current = overlap_parts
                    current_len = ol
                current.append(part)
                current_len += part_len

            if current:
                chunks.extend(_split(sep.join(current), seps[1:]))
            return chunks

        raw = _split(text, separators)
        # deduplicate consecutive identical chunks (can arise from overlap)
        seen: List[str] = []
        for c in raw:
            c = c.strip()
            if c and (not seen or c != seen[-1]):
                seen.append(c)
        return seen

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks on sentence boundaries.

        Paragraph breaks are preserved as [PARA] markers so the LLM
        can still detect topic shifts within a chunk.
        """
        # split on sentence boundaries, preserve paragraph markers for context
        sentences = text.replace("\n\n", " [PARA] ").split(". ")
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(". ".join(current_chunk) + ".")

                if overlap > 0 and len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

    @staticmethod
    def load_text_file(
        filepath: str, chunk_size: int = 500, strategy: str = "sentence_aware"
    ) -> List[Document]:
        """Load and chunk a plain-text (.txt) document."""
        path = Path(filepath)

        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return []

        text = path.read_text(encoding="utf-8")
        if strategy == "recursive":
            chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=chunk_size)
        else:
            chunks = DocumentLoader.chunk_text(text, chunk_size=chunk_size)

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    content=chunk,
                    metadata={
                        "id": f"{path.stem}_chunk_{i}",
                        "source": str(path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": path.name,
                        "file_type": "txt",
                    },
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from {path.name}")
        return documents

    @staticmethod
    def load_pdf_file(
        filepath: str, chunk_size: int = 500, strategy: str = "sentence_aware"
    ) -> List[Document]:
        """
        Load and chunk a PDF document.

        Requires ``pypdf`` (included in requirements.txt).
        Text is extracted page-by-page and then chunked the same way as .txt files.
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf not installed. Run: pip install pypdf")
            return []

        path = Path(filepath)

        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return []

        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not full_text.strip():
            logger.warning(f"No text extracted from {path.name} (may be a scanned/image PDF)")
            return []

        if strategy == "recursive":
            chunks = DocumentLoader.recursive_chunk_text(full_text, chunk_size=chunk_size)
        else:
            chunks = DocumentLoader.chunk_text(full_text, chunk_size=chunk_size)

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    content=chunk,
                    metadata={
                        "id": f"{path.stem}_chunk_{i}",
                        "source": str(path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": path.name,
                        "file_type": "pdf",
                        "page_count": page_count,
                    },
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from PDF: {path.name} ({page_count} pages)")
        return documents

    @staticmethod
    def load_docx_file(
        filepath: str, chunk_size: int = 500, strategy: str = "sentence_aware"
    ) -> List[Document]:
        """
        Load and chunk a Word (.docx) document.

        Requires ``python-docx``. Each paragraph is joined with a newline
        and then chunked using the same strategy as .txt and .pdf files.
        """
        try:
            from docx import Document as DocxDocument
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return []

        path = Path(filepath)

        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return []

        doc = DocxDocument(str(path))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        if not full_text.strip():
            logger.warning(f"No text extracted from {path.name}")
            return []

        if strategy == "recursive":
            chunks = DocumentLoader.recursive_chunk_text(full_text, chunk_size=chunk_size)
        else:
            chunks = DocumentLoader.chunk_text(full_text, chunk_size=chunk_size)

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc_obj = Document(
                    content=chunk,
                    metadata={
                        "id": f"{path.stem}_chunk_{i}",
                        "source": str(path),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "filename": path.name,
                        "file_type": "docx",
                    },
                )
                documents.append(doc_obj)

        logger.info(f"Loaded {len(documents)} chunks from DOCX: {path.name}")
        return documents

    @staticmethod
    def load_directory(
        dirpath: str,
        pattern: str = "*",
        chunk_size: int = 500,
        strategy: str = "sentence_aware",
    ) -> List[Document]:
        """
        Load all supported documents (.txt, .pdf, .docx) from a directory.

        The ``pattern`` argument is accepted for backwards compatibility
        but the method always scans for all supported extensions.
        """
        path = Path(dirpath)

        if not path.exists():
            logger.error(f"Directory not found: {dirpath}")
            return []

        all_docs: List[Document] = []

        for filepath in sorted(path.iterdir()):
            if not filepath.is_file():
                continue
            if filepath.suffix == ".txt":
                docs = DocumentLoader.load_text_file(str(filepath), chunk_size, strategy)
            elif filepath.suffix == ".pdf":
                docs = DocumentLoader.load_pdf_file(str(filepath), chunk_size, strategy)
            elif filepath.suffix == ".docx":
                docs = DocumentLoader.load_docx_file(str(filepath), chunk_size, strategy)
            else:
                continue
            all_docs.extend(docs)

        logger.info(f"Total: {len(all_docs)} document chunks from {dirpath}")
        return all_docs
