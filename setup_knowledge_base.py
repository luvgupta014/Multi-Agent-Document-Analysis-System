from rag_system.knowledge_base.loader import DocumentLoader
from rag_system.knowledge_base.vector_store import VectorStore
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()


def setup_knowledge_base():
    doc_dir = os.getenv("KB_DOC_DIR", "docs/sample_documents")
    persist_dir = os.getenv("KB_PERSIST_DIR", "./chroma_db")
    chunk_size = int(os.getenv("KB_CHUNK_SIZE", "500"))
    collection = os.getenv("SPECIALIST_COLLECTION", "climate_docs")
    force_reload = os.getenv("FORCE_RELOAD", "false").lower() == "true"

    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if not os.listdir(doc_dir):
        logger.error(f"No documents found in {doc_dir}")
        return

    documents = DocumentLoader.load_directory(
        dirpath=doc_dir,
        pattern="*.txt",
        chunk_size=chunk_size
    )

    if not documents:
        logger.error("No documents loaded")
        return

    logger.info(f"Loaded {len(documents)} document chunks")

    vector_store = VectorStore(
        collection_name=collection,
        persist_dir=persist_dir
    )

    if vector_store.collection.count() > 0:
        if force_reload:
            logger.info("Force reload enabled, clearing existing data")
            vector_store.clear()
        else:
            response = input("Collection already has data. Clear and reload? (y/n): ")
            if response.lower() == 'y':
                vector_store.clear()
            else:
                logger.info("Keeping existing data")
                return

    vector_store.add_documents(documents)

    test_queries = [
        "What causes climate change?",
        "How does renewable energy work?",
        "What can individuals do to help?"
    ]

    for query in test_queries:
        results = vector_store.search(query, top_k=2)
        logger.info(f"Query: '{query}' -> {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"  Result {i} (similarity={result['similarity']:.3f}): {result['content'][:100]}...")

    logger.info(f"Setup complete. {vector_store.collection.count()} documents indexed.")


if __name__ == "__main__":
    setup_knowledge_base()
