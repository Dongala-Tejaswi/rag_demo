import os

from fastapi import FastAPI
from pydantic import BaseModel

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

from transformers import pipeline


# ===============================
# FASTAPI APP
# ===============================

app = FastAPI(
    title="Resume RAG Chatbot",
    description="RAG chatbot for resume question answering",
    version="1.0"
)


# ===============================
# PDF PATH
# ===============================

pdf_path = os.path.join(
    os.path.dirname(__file__),
    "documents",
    "AI_ML_RESUME_DONGALA_Tejaswi (2).pdf"
)


# ===============================
# LOAD PDF
# ===============================

def load_pdf(file_path):

    with open(file_path, "rb") as f:
        header = f.read(5)

    if header != b"%PDF-":
        raise Exception("Invalid PDF file")

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


document_text = load_pdf(pdf_path)

print("PDF Loaded Successfully")


# ===============================
# TEXT CHUNKING
# ===============================

def split_text(
    text,
    chunk_size=120,
    overlap=30
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        chunks.append(chunk)

        start = end - overlap

    return chunks


chunks = split_text(document_text)

print(
    "Number of chunks:",
    len(chunks)
)


# ===============================
# EMBEDDING MODEL
# ===============================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    chunks
)

embeddings = np.array(
    embeddings
).astype("float32")


# Normalize for cosine similarity

faiss.normalize_L2(
    embeddings
)


print("Embeddings Created")


# ===============================
# CREATE FAISS DATABASE
# ===============================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)

print("FAISS Database Created")


# ===============================
# LOAD LLM
# ===============================

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

print("LLM Loaded Successfully")


# ===============================
# RETRIEVAL
# ===============================

def retrieve_context(
    question,
    top_k=2
):

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    faiss.normalize_L2(
        question_embedding
    )

    distances, ids = index.search(
        question_embedding,
        top_k
    )

    results = []

    for i in ids[0]:

        if i >= 0:
            results.append(
                chunks[i]
            )

    return "\n\n".join(results)


# ===============================
# RAG ANSWERING
# ===============================

def ask_question(question):

    context = retrieve_context(
        question
    )

    prompt = f"""
You are a resume assistant.

Answer only from the context.

Do not include unrelated information.

If the answer is not available, say:

Information not found in resume.

Context:

{context}

Question:

{question}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return response[0]["generated_text"]


# ===============================
# REQUEST MODEL
# ===============================

class QuestionRequest(BaseModel):

    question: str


# ===============================
# HOME ENDPOINT
# ===============================

@app.get("/")
def home():

    return {
        "message": "Resume RAG API is running",
        "status": "success"
    }


# ===============================
# ASK ENDPOINT
# ===============================

@app.post("/ask")
def ask(request: QuestionRequest):

    answer = ask_question(
        request.question
    )

    return {
        "question": request.question,
        "answer": answer
    }