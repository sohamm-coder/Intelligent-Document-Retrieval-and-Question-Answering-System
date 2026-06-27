from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize
from openai import OpenAI
import torch

client = OpenAI(api_key="")

# ---- DEVICE SETUP ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- LOAD MODEL ----
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

index = faiss.read_index("vector_index/sentence_index.faiss")

with open("vector_index/sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f]

with open("vector_index/sentence_meta.txt", "r", encoding="utf-8") as f:
    metadata = [line.strip() for line in f]

# === Set OpenAI key ===

# === RAG Function ===
def rag_answer(question, chunks):
    context = "\n".join(chunks)
    prompt = f"""
You are a helpful assistant. Based on the context below, answer the question clearly.

Context:
{context}

Question: {question}
If the answer is not directly in the text, say "Not found in the provided context."
Answer:
"""
    response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response.choices[0].message.content.strip()

# === CONTINUOUS QUERY LOOP ===
last_query = None

print("Enter your questions (type 'exit' to quit):")
while True:
    query = input("\n Question: ").strip()

    if query.lower() in {"exit", "quit"}:
        print("Exiting...")
        break

    if not query:
        print("Please enter a question.")
        continue

    if query == last_query:
        print("Thats the same question as before. Try asking something new.")
        continue

    last_query = query  # Update history

    query_vector = model.encode([query])
    query_vector = normalize(query_vector).astype("float32")

    D, I = index.search(query_vector, k=3)
    top_chunks = [sentences[i] for i in I[0]]

    print("\n" + "-" * 60)
    print(f"[Context Chunk]: {top_chunks[0]}")
    answer = rag_answer(query, top_chunks)
    print(f"[Answer]: {answer}")
    print("-" * 60)

