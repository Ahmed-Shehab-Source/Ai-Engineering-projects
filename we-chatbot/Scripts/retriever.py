from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# -----------------------------
# CONFIG
# -----------------------------

OPENROUTER_API_KEY = "openrouter_api_key_here"  # Replace with your OpenRouter API key

QDRANT_URL = "qdrant_url_here"  # Replace with your Qdrant URL

QDRANT_API_KEY = "qdrant_api_key_here"  # Replace with your Qdrant API key

COLLECTION_NAME = "we_chatbot"  # Replace with your Qdrant collection name

# -----------------------------
# Embeddings + Vector store (initialized once, reused across calls)
# -----------------------------

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
)


def retrieve(query: str, k: int = 5):
    """
    Retrieve the top-k most relevant chunks for a query, with similarity scores.

    Returns a list of dicts: [{"content": str, "metadata": dict, "score": float}, ...]
    Score is a similarity score from Qdrant (higher = more similar, since
    similarity_search_with_score returns relevance depending on distance metric —
    for cosine, this is typically in [0, 1] with 1 = identical).
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        })
    return chunks


def format_context(chunks):
    """Join retrieved chunk contents into a single context string for the LLM prompt."""
    return "\n\n---\n\n".join(c["content"] for c in chunks)


# -----------------------------
# Quick manual test
# -----------------------------
if __name__ == "__main__":
    query = "How do I renew my internet package?"
    chunks = retrieve(query, k=5)

    print(f"\nRetrieved {len(chunks)} chunks\n")
    for i, c in enumerate(chunks, start=1):
        print("=" * 80)
        print(f"Chunk {i} | score: {c['score']:.4f}")
        print("Source:", c["metadata"].get("source"))
        print()
        print(c["content"][:300])
        print()
