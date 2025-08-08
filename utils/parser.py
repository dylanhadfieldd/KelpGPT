# utils/parser.py

import os
import pdfplumber
import pandas as pd
from docx import Document

def parse_document(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return parse_pdf(filepath)
    elif ext == ".docx":
        return parse_docx(filepath)
    elif ext in [".csv", ".xlsx"]:
        return parse_csv(filepath)
    elif ext == ".txt":
        return parse_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def parse_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def parse_docx(filepath):
    doc = Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def parse_csv(filepath):
    df = pd.read_csv(filepath)
    return df.to_string(index=False)

def parse_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
