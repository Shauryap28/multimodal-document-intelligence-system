"""
Phase 1 runner - wires the refactored modules into a Q&A loop.

Run:   python main.py
First run ingests + persists to data/chroma_db; later runs reload instantly.
Put your PDF at data/samples/sample.pdf (or change PDF_PATH below).
"""
import os
import sys

from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings
from backend.rag.embeddings import get_embeddings
from backend.rag.chunking import chunk_documents
from backend.rag.vectorstore import get_vectorstore, add_documents, count
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain
from backend.services.pipelines.text_pdf import load_text_pdf

PDF_PATH = "data/samples/sample.pdf"

if not settings.GOOGLE_API_KEY:
    sys.exit("Set GOOGLE_API_KEY in a .env file first "
             "(https://aistudio.google.com/apikey).")

embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)

# Persistence: only ingest when the store is empty.
if count(vectorstore) == 0:
    if not os.path.exists(PDF_PATH):
        sys.exit(f"Put a text PDF at {PDF_PATH} first.")
    print("Ingesting document (first run)...")
    docs = load_text_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    add_documents(vectorstore, chunks)
    print(f"  stored {len(chunks)} chunks")
else:
    print(f"Loaded existing store ({count(vectorstore)} chunks)")

retriever = get_retriever(vectorstore)

llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    temperature=0,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
)

qa_chain = build_qa_chain(retriever, llm)

print("\nReady. Ask questions about your PDF (blank line to quit).\n")
while True:
    q = input("Q: ").strip()
    if not q:
        break
    print("A:", qa_chain.invoke(q), "\n")