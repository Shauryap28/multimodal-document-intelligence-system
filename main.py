"""
Phase 1 CLI runner - wires the refactored modules into a Q&A loop.

Run:   python main.py
First run ingests + persists to data/chroma_db; later runs reload instantly.
Put your PDF at data/samples/sample.pdf (or change PDF_PATH below).

This runner uses Groq (Llama 3.3 70B) for text generation, via the
backend.rag.llm factory.
"""
import os
import sys

from backend.config import settings
from backend.rag.embeddings import get_embeddings
from backend.rag.chunking import chunk_documents
from backend.rag.vectorstore import get_vectorstore, add_documents, count
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain
from backend.rag.llm import get_llm
from backend.services.pipelines.text_pdf import load_text_pdf

PDF_PATH = "data/samples/sample.pdf"

if not settings.GROQ_API_KEY:
    sys.exit(
        "GROQ_API_KEY not set in .env. "
        "Get a free key at https://console.groq.com/keys"
    )

embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)

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
llm = get_llm("text")
qa_chain = build_qa_chain(retriever, llm)

print("\nReady. Ask questions about your PDF (blank line to quit).\n")
while True:
    q = input("Q: ").strip()
    if not q:
        break
    print("A:", qa_chain.invoke(q), "\n")