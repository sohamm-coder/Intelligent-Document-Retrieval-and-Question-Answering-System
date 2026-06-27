import os
from openai import OpenAI
from transformers import pipeline
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# === Device Info ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device set to:", device)

# === Config ===
client = OpenAI(api_key="")  # or use environment variable
INPUT_DIR = "data/extracted_texts"
OUTPUT_DIR = "data/profession_summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROFILES = ["student", "technical professional", "non-technical reader"]
USE_CHATGPT = True

# === Load summarizer (only works in main thread)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

def summarize_text_local(text):
    if len(text) < 100:
        return text.strip()
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summarized = summarizer(chunks, max_length=150, min_length=40, do_sample=False)
    return " ".join([s["summary_text"] for s in summarized])

def refine_for_profile(base_summary, profile):
    truncated = base_summary[:3000]
    prompt = f"""
You're an expert at writing summaries. Rewrite the following summary for a {profile}, adjusting tone and content accordingly.

Original Summary:
{truncated}

Profession-Specific Summary:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response.choices[0].message.content.strip()

def process_file(fname):
    if not fname.endswith(".txt"):
        return

    input_path = os.path.join(INPUT_DIR, fname)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f" Processing {fname}")
    base_summary = summarize_text_local(raw_text)
    print(f" Base summary length: {len(base_summary)} characters")

    for profile in PROFILES:
        out_fname = f"{os.path.splitext(fname)[0]}_{profile.replace(' ', '_')}.txt"
        out_path = os.path.join(OUTPUT_DIR, out_fname)
        if os.path.exists(out_path):
            print(f"Skipping (cached): {out_fname}")
            continue
        if USE_CHATGPT and len(base_summary) > 100:
            final_summary = refine_for_profile(base_summary, profile)
        else:
            final_summary = base_summary

        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(final_summary)

        print(f"Saved summary for {profile}: {out_fname}")

if __name__ == "__main__":
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    for fname in tqdm(files, desc="Processing files"):
        process_file(fname)
