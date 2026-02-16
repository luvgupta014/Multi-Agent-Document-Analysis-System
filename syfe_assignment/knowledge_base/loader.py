from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger


class Document:

    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content.strip()
        self.metadata = metadata or {}
        self.doc_id = self.metadata.get('id', str(hash(content)))

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.doc_id}, content='{preview}')"

    def __len__(self):
        return len(self.content)


class DocumentLoader:

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        # split on sentence boundaries, preserve paragraph markers for context
        sentences = text.replace('\n\n', ' [PARA] ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')

                if overlap > 0 and len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    @staticmethod
    def load_text_file(filepath: str, chunk_size: int = 500) -> List[Document]:
        path = Path(filepath)

        if not path.exists():
            logger.error(f"File not found: {filepath}")
            return []

        text = path.read_text(encoding='utf-8')
        chunks = DocumentLoader.chunk_text(text, chunk_size=chunk_size)

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    content=chunk,
                    metadata={
                        'id': f"{path.stem}_chunk_{i}",
                        'source': str(path),
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'filename': path.name
                    }
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} chunks from {path.name}")
        return documents

    @staticmethod
    def load_directory(dirpath: str, pattern: str = "*.txt", chunk_size: int = 500) -> List[Document]:
        path = Path(dirpath)

        if not path.exists():
            logger.error(f"Directory not found: {dirpath}")
            return []

        all_docs = []

        for filepath in path.glob(pattern):
            if filepath.is_file():
                if filepath.suffix == '.txt':
                    docs = DocumentLoader.load_text_file(str(filepath), chunk_size)
                    all_docs.extend(docs)

        logger.info(f"Total: {len(all_docs)} documents from directory")
        return all_docs
