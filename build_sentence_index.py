import os
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import re

def split_into_sentences(text):
    # Fallback safe splitter — handles missing punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:  
        # No punctuation → hard-split by 200-character blocks
        sentences = [text[i:i+200] for i in range(0, len(text), 200)]
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def create_sliding_windows(sentences, window_size=3, stride=1):
    if len(sentences) < window_size:
        return [" ".join(sentences)]  # fallback single chunk
    windows = []
    for i in range(0, len(sentences) - window_size + 1, stride):
        window = " ".join(sentences[i:i+window_size])
        windows.append(window)
    return windows

# ----- MODEL -----
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SOURCE_DIRS = ["data/extracted_texts", "data/profession_summaries"]
OUTPUT_DIR = "vector_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sentences = []
metadata = []

for folder in SOURCE_DIRS:
    print(f"\n Reading folder: {folder}")
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(folder, fname)
        print(f"Processing {path}...")

        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        sents = split_into_sentences(raw)
        chunks = create_sliding_windows(sents, window_size=4, stride=1)

        print(f"   Sentences found: {len(sents)} | Chunks created: {len(chunks)}")

        sentences.extend(chunks)
        metadata.extend([fname] * len(chunks))

# ----- SAFETY CHECK -----
if len(sentences) == 0:
    raise ValueError("No sentences or chunks were extracted! Check folder paths or input text.")

# ----- EMBEDDINGS -----
print(f"\nEncoding {len(sentences)} chunks...")
embeddings = model.encode(sentences, show_progress_bar=True)
embeddings = normalize(embeddings).astype("float32")

# ----- FAISS INDEX -----
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ----- SAVE -----
faiss.write_index(index, os.path.join(OUTPUT_DIR, "sentence_index.faiss"))

with open(os.path.join(OUTPUT_DIR, "sentence_meta.txt"), "w", encoding="utf-8") as mf:
    mf.writelines(line + "\n" for line in metadata)

with open(os.path.join(OUTPUT_DIR, "sentences.txt"), "w", encoding="utf-8") as sf:
    sf.writelines(line + "\n" for line in sentences)

print(f"\n Indexed {len(sentences)} chunks successfully!")
