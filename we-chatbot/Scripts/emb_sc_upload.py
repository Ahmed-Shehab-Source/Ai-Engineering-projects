import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# -----------------------------
# PASTE YOUR KEYS HERE
# -----------------------------

OPENROUTER_API_KEY = "openrouter_api_key_here"  # Replace with your OpenRouter API key

QDRANT_URL = "qdrant_url_here"  # Replace with your Qdrant URL

QDRANT_API_KEY = "qdrant_api_key_here"  # Replace with your Qdrant API key

# -----------------------------
# Load chunks
# -----------------------------

with open("chunks.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

documents = [
    Document(
        page_content=item["content"],
        metadata=item.get("metadata", {})
    )
    for item in raw
]

print(f"Loaded {len(documents)} chunks")

# -----------------------------
# Embedding model
# -----------------------------

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# -----------------------------
# Upload to Qdrant Cloud
# -----------------------------

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="we_chatbot",
)

batch_size = 10

for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vector_store.add_documents(batch)
    print(f"Uploaded {min(i+batch_size, len(documents))}/{len(documents)}")

print("DONE!")