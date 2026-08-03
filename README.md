# Intelligent Document Retrieval and Question Answering System

An AI-powered Retrieval-Augmented Generation (RAG) system that extracts text from digital and scanned PDF documents, generates intelligent summaries, builds a semantic vector index, and answers natural language questions using GPT-4.

The project combines OCR, document summarization, semantic search, vector embeddings, and large language models to enable accurate question answering over unstructured documents.

---

## Features

- Extracts text from both digital and scanned PDF documents
- OCR using Tesseract for image-based PDFs
- Digital text extraction using PyMuPDF
- Automatic text cleaning and duplicate removal
- AI-powered document summarization using Facebook BART
- Audience-specific summaries generated using OpenAI
  - Student
  - Technical Professional
  - Non-Technical Reader
- Semantic document search using Sentence Transformers
- FAISS vector indexing for efficient retrieval
- Retrieval-Augmented Generation (RAG) using GPT-4
- Sliding-window chunking for improved retrieval context
- GPU (CUDA) support when available

---

# System Architecture

```
                PDF Documents
                      │
          ┌───────────┴────────────┐
          │                        │
 Digital Text Extraction      OCR Extraction
    (PyMuPDF)               (Tesseract + pdf2image)
          │                        │
          └───────────┬────────────┘
                      │
             Text Cleaning
      Duplicate Removal & Normalization
                      │
                      ▼
          Audience-aware Summaries
      (BART + OpenAI GPT-3.5 Turbo)
                      │
                      ▼
        Sentence Chunking (Sliding Window)
                      │
                      ▼
      Sentence Transformer Embeddings
          all-MiniLM-L6-v2
                      │
                      ▼
             FAISS Vector Index
                      │
                      ▼
             User Question
                      │
        Sentence Transformer
          Query Embedding
                      │
                      ▼
          Top-k Semantic Retrieval
                      │
                      ▼
              GPT-4 RAG Prompt
                      │
                      ▼
          Context-aware Final Answer
```

---

# Technology Stack

### Programming

- Python

### AI / Machine Learning

- OpenAI GPT-4
- OpenAI GPT-3.5 Turbo
- Sentence Transformers
- Facebook BART
- Retrieval-Augmented Generation (RAG)

### Vector Database

- FAISS

### OCR

- Tesseract OCR
- pdf2image

### PDF Processing

- PyMuPDF (fitz)

### NLP

- HuggingFace Transformers

### Data Processing

- NumPy
- Scikit-learn

---

# Project Workflow

## Step 1 – Extract Text

The system processes PDF documents using two independent extraction methods.

### Digital PDFs

Text is extracted directly using PyMuPDF.

### Scanned PDFs

Images are generated from every page using pdf2image and passed through Tesseract OCR.

The outputs from both approaches are merged to maximize document coverage.

---

## Step 2 – Clean Text

The extracted content is

- normalized
- whitespace cleaned
- duplicate lines removed
- converted into a structured text file

---

## Step 3 – Generate Intelligent Summaries

The extracted document is summarized using Facebook BART.

The summary is then rewritten using OpenAI to generate versions for different audiences:

- Student
- Technical Professional
- Non-Technical Reader

---

## Step 4 – Build Semantic Index

The documents are split into overlapping sliding-window chunks.

Each chunk is converted into embeddings using

```
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are normalized and stored inside a FAISS vector index.

---

## Step 5 – Question Answering

When a user asks a question:

1. The question is converted into an embedding.
2. FAISS retrieves the most semantically relevant chunks.
3. Retrieved context is passed to GPT-4.
4. GPT-4 generates a grounded response using only the retrieved context.

---

# Folder Structure

```
data/
│
├── raw_pdfs/
│
├── extracted_texts/
│
├── profession_summaries/
│
vector_index/
│
├── sentence_index.faiss
├── sentence_meta.txt
└── sentences.txt

ocr_extract.py

generate_summaries.py

build_sentence_index.py

search_sentences.py
```

---

# Key Components

## OCR Pipeline

- Tesseract OCR
- pdf2image
- PyMuPDF

Combines OCR text and digital text into a unified representation.

---

## Document Summarization

Uses

- Facebook BART
- OpenAI GPT

to generate audience-aware summaries.

---

## Semantic Search

Uses

- Sentence Transformers
- FAISS

to retrieve relevant document chunks instead of keyword matching.

---

## Retrieval-Augmented Generation

GPT-4 receives only the retrieved document context, reducing hallucinations and improving response quality.

---

# Future Improvements

- Streamlit web interface
- Multi-document chat
- Citation generation
- Metadata filtering
- Hybrid keyword + semantic retrieval
- Incremental indexing
- Conversation memory
- Docker deployment
- Cloud deployment (AWS/Azure)

---

# Author

**Soham Mahajan**

MS Computer Science  
University of Massachusetts Dartmouth

GitHub:
https://github.com/sohamm-coder

LinkedIn:
https://www.linkedin.com/in/sohammahajan/
