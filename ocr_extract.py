import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  
import os
import re
from tqdm import tqdm

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\ProgramData\Tesseract-OCR\tesseract.exe"

INPUT_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/extracted_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = text.encode("ascii", errors="ignore").decode()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_digital(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return clean_text(text)
    except Exception as e:
        print(f" Digital extract completed: {e}")
        return None

def extract_text_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        text = ""
        for img in tqdm(images, desc=f"OCR {os.path.basename(pdf_path)}", unit="page"):
            text += pytesseract.image_to_string(img) + "\n"
        return clean_text(text)
    except Exception as e:
        print(f"OCR extract failed: {e}")
        return None

def extract_pdf(pdf_path, output_path):
    print(f"\n[📄] Processing: {os.path.basename(pdf_path)}")

    # Always extract digital text
    digital_text = extract_text_digital(pdf_path) or ""

    # Always extract OCR text (for headers, logos, scanned text)
    ocr_text = extract_text_ocr(pdf_path) or ""

    # Merge both
    merged = (digital_text + "\n" + ocr_text).strip()

    # Optional: Remove duplicate lines
    merged = "\n".join(list(dict.fromkeys(merged.split("\n"))))

    if merged.strip():
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged)
        print(f"Saved merged OCR + digital text → {output_path}")
    else:
        print(f"Failed to extract any text from: {pdf_path}")


# Loop through all PDFs
for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(".pdf"):
        pdf_path = os.path.join(INPUT_DIR, file)
        txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".txt")
        extract_pdf(pdf_path, txt_path)
