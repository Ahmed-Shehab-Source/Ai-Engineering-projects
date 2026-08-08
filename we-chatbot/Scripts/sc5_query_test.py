from openai import OpenAI, RateLimitError
import time
from retriever import retrieve, format_context

# -----------------------------
# CONFIG
# -----------------------------

OPENROUTER_API_KEY = "openrouter_api_key_here"  # Replace with your OpenRouter API key

# Using OpenRouter's free auto-router: picks whichever free model is
# currently available, so a single model being rate-limited doesn't block you.
LLM_MODEL = "openrouter/free"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """You are a helpful customer support assistant for WE (Telecom Egypt).
Answer the user's question using ONLY the information in the provided context.
If the context does not contain the answer, say you don't have that information
and suggest contacting WE customer service.
Respond in the same language as the user's question (Arabic or English).
Be concise and clear.
"""


def ask(query: str, k: int = 5, show_sources: bool = False):
    # 1. Retrieve relevant chunks
    chunks = retrieve(query, k=k)
    context = format_context(chunks)

    # 2. Build the prompt
    user_prompt = f"""Context:
{context}

Question:
{query}
"""

    # 3. Call the LLM (retry once on rate limit)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except RateLimitError as e:
            if attempt == 0:
                print("Rate limited, waiting 30s and retrying...")
                time.sleep(30)
            else:
                raise e

    answer = response.choices[0].message.content

    if show_sources:
        print("\n--- Retrieved sources ---")
        for c in chunks:
            print(f"  [{c['score']:.4f}] {c['metadata'].get('source')}")
        print("-------------------------\n")

    return answer


if __name__ == "__main__":
    query = "What is the SIP Trunk service and what does it offer?"
    answer = ask(query, k=5, show_sources=True)
    print("Answer:\n")
    print(answer)