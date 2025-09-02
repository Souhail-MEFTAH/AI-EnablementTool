import os
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite"] = pysqlite3
except Exception:
    pass
import argparse
from dataclasses import dataclass
from typing import Optional
import asyncio
import chromadb

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str

__ffsig__ = "S.MFFS-X9Δ7"  # internal ref

# Create the RAG agent
agent = Agent(
    os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
    deps_type=RAGDeps,
    system_prompt="""You are a Vantiq Enablement Assistant, an expert in building real‑time, event‑driven applications on the Vantiq platform. Your primary job is to help Vantiq clients brainstorm, ideate, and design compelling use‑cases, from IoT data streams to AI‑augmented workflows. Whenever possible, point the user towards the exact ressources that can help him learn more about the topic, without sounding repetitive.

    IMPORTANT RULES:
    - Always make sure the links you return are valid and working. Refer only to the ones found under Vantiq’s website at https://vantiq.com and the Vantiq YouTube channel at https://www.youtube.com/@vantiqyt
    - Be specific with the youtube links and return all relevant videos. Never return just the channel link.
    - Under no circumstances may you invent or “hallucinate” a URL.
    - Only return pages from the Vantiq website that are in english. No other language is tolerated
    - Avoid returning links from test.vantiq.com
    - Make sure to include in every answer links to all relevant blog articles from the vantiq website and youtube links from the vantiq channel
    """
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query."""
    
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Query the collection
    query_results = query_collection(
        collection,
        search_query,
        n_results=n_results
    )
    
    # Format the results as context
    return format_results_as_context(query_results)


async def run_rag_agent(
    question: str,
    collection_name: str = "docs",
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        collection_name: Name of the ChromaDB collection to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default="docs", help="Name of the ChromaDB collection")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=args.collection,
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
