"""PDF ingestion pipeline for BimBam Buy RAG Agent.

Loads PDFs from docs/, chunks them, and creates a ChromaDB vector store.
"""

import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DOCS_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdfs(docs_dir: str) -> list:
    """Load all PDFs from the given directory."""
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}")
        return []

    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Add source metadata
            filename = os.path.basename(pdf_path)
            for doc in docs:
                doc.metadata["source"] = filename
            documents.extend(docs)
            print(f"Loaded {filename}: {len(docs)} pages")
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


def create_vectorstore(chunks: list, persist_dir: str) -> None:
    """Create ChromaDB vector store from chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Check if vectorstore already exists
    if os.path.exists(persist_dir):
        existing = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        count = existing._collection.count()
        if count > 0:
            print(f"Vector store already exists with {count} vectors. Skipping ingestion.")
            print("To re-ingest, delete the chroma_db/ directory first.")
            return

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Vector store created at {persist_dir} with {vectorstore._collection.count()} vectors")


def main():
    """Main ingestion pipeline."""
    print("=" * 50)
    print("BimBam Buy - PDF Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load PDFs
    documents = load_pdfs(DOCS_DIR)
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    if not chunks:
        print("No chunks created. Exiting.")
        return

    # Step 3: Create vector store
    create_vectorstore(chunks, CHROMA_DIR)

    print("=" * 50)
    print("Ingestion complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
