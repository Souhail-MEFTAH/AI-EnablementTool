#!/usr/bin/env python3
"""
insert_local.py
---------------
Crawl a directory of local files, extract text, and insert into ChromaDB.
Usage:
    python insert_local.py [<local_files_dir>]
If <local_files_dir> is omitted, defaults to ./local_files
"""

import sys
import logging
from pathlib import Path
from utils import (
    get_chroma_client,
    get_or_create_collection,
    add_documents_to_collection,
)
from agents.modules.loaders.local_loader import load_supported_file

# Simple container for page_content + metadata
class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata     = metadata

# Which file types to pick up
SUPPORTED_EXTENSIONS = [
    ".txt", ".md", ".csv", ".json",
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".mp4", ".mkv", ".avi"
]

def crawl_local_files(local_dir: Path) -> list[Document]:
    all_docs = []
    for file_path in local_dir.rglob("*"):
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                content = load_supported_file(file_path)
                if content:
                    all_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": str(file_path)}
                        )
                    )
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
    return all_docs

def main():
    # 1) Determine the folder to crawl
    if len(sys.argv) > 1:
        local_dir = Path(sys.argv[1]).resolve()
    else:
        local_dir = Path(__file__).parent / "local_files"

    if not local_dir.exists():
        print(f"[error] Directory not found: {local_dir}")
        sys.exit(1)

    # 2) Crawl & load
    docs = crawl_local_files(local_dir)
    if not docs:
        print("No valid documents found in local_files/")
        sys.exit(0)

    # 3) Connect to ChromaDB
    client     = get_chroma_client("./chroma_db")
    collection = get_or_create_collection(client, "local_files")

    # 4) Unpack for ChromaDB
    documents = [doc.page_content for doc in docs]
    metadatas = [doc.metadata     for doc in docs]
    ids        = [f"{doc.metadata['source']}_{i}" for i, doc in enumerate(docs)]

    # 5) Insert!
    add_documents_to_collection(
        collection=collection,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"Inserted {len(docs)} documents into vector store.")

if __name__ == "__main__":
    main()