import os
import pathlib
import hashlib
from typing import List, Dict, Any, Optional
import re
import pysbd # <-- Use the correct library, pysbd

import chromadb
from chromadb.utils import embedding_functions
from more_itertools import batched


def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(persist_directory)


def get_or_create_collection(
        client: chromadb.PersistentClient,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        distance_function: str = "cosine",
) -> chromadb.Collection:
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )

    try:
        return client.get_collection(name=collection_name)
    except:
        return client.create_collection(
            name=collection_name,
            embedding_function=embedding_func,
            metadata={"hnsw:space": distance_function}
        )


def add_documents_to_collection(
    collection: chromadb.Collection,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    batch_size: int = 0,      
) -> None:
    """
    Insert docs into ChromaDB in batches if batch_size>0, else all at once.
    """
    if batch_size and batch_size > 0:
        # iterate in batches
        for doc_batch, meta_batch, id_batch in zip(
            batched(documents, batch_size),
            batched(metadatas,  batch_size),
            batched(ids,        batch_size),
        ):
            collection.add(
                documents=list(doc_batch),
                metadatas=list(meta_batch),
                ids=list(id_batch),
            )
    else:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )


def query_collection(
        collection: chromadb.Collection,
        query_text: str,
        n_results: int = 5,
) -> List[Dict[str, Any]]:
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results


def format_results_as_context(results: Dict[str, Any]) -> str:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    context = []
    
    for doc, metadata in zip(documents, metadatas):
        # Add the real source URL to each retrieved document
        url = metadata.get("source")
        if url:
            context.append(f"{doc}\n\nSource URL: {url}")
        else:
            context.append(doc)

    return "\n\n---\n\n".join(context)


def chunk_text(text: str, max_sentences: int = 5, overlap: int = 2) -> list:
    """Split text into overlapping sentence chunks using the pysbd library."""
    try:
        # Initialize the segmenter for English
        seg = pysbd.Segmenter(language="en", clean=False)
        sentences = seg.segment(text)
    except Exception as e:
        # Fallback in case pysbd fails for some reason
        print(f"[warning] pysbd failed: {e}. Using regex-based fallback.")
        sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    start = 0

    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap

    return chunks


def embed_chunks(chunk_metadata_list: list, collection_name: str = "rag-content") -> None:
    """Embed chunks and store them in ChromaDB using existing logic."""
    client = get_chroma_client("chroma_db")
    collection = get_or_create_collection(client, collection_name)

    documents = []
    metadatas = []
    ids = []

    for i, (chunk, metadata) in enumerate(chunk_metadata_list):
        documents.append(chunk)
        metadatas.append(metadata)
        ids.append(f"{metadata.get('source', 'youtube')}_{i}")

    add_documents_to_collection(collection, documents, metadatas, ids)


# ============ NEW Features ============

def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_exists(collection, doc_hash: str) -> bool:
    try:
        results = collection.get(
            where={"doc_hash": doc_hash},
            limit=1
        )
        return len(results.get("ids", [])) > 0
    except Exception:
        return False


def log_collection_contents(collection, max_items: int = 10):
    try:
        items = collection.get(include=["metadatas", "documents"], limit=max_items)
        for i, doc_id in enumerate(items["ids"]):
            metadata = items["metadatas"][i]
            doc_snippet = items["documents"][i][:150].replace("\n", " ")
            print(f"#{i+1} | ID: {doc_id} | Source: {metadata.get('source', 'N/A')} | Preview: {doc_snippet}")
    except Exception as e:
        print(f"Error reading collection: {e}")