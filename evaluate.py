"""Evaluation suite for the RAG core.

Runs a fixed set of questions (with known answers) against a controlled,
in-memory corpus and reports two metrics, kept separate on purpose:

  - Context recall (retrieval): did the retrieved chunks CONTAIN the answer's
    supporting fact? Deterministic, no LLM needed.
  - Answer correctness (generation): does the final answer contain the expected
    fact? Normalized substring match (free; an LLM-as-judge could replace this
    later for fuzzier answers).

The corpus is plain text and lives in-memory, so the suite is free to run
repeatedly (no OCR/Vision calls) and never touches the main store. The runner
plugs into the EXISTING get_retriever / build_qa_chain unchanged - the point is
to evaluate the real pipeline, not a copy of it.

Each run makes one Groq call per test case (cheap/fast). Retrieval scoring is
fully local.

Run from project root:  python evaluate.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import settings
from backend.rag.embeddings import get_embeddings
from backend.rag.chunking import chunk_documents
from backend.rag.retriever import get_retriever
from backend.rag.chain import build_qa_chain
from backend.rag.llm import get_llm


# --- Controlled corpus ----------------------------------------------------

_CATEGORIES = ["Tools", "Electronics", "Garden", "Office"]


def _build_catalog() -> str:
    """100 products, categories cycling -> exactly 25 per category (by construction).
    Large enough (~6 chunks) that aggregation must span multiple chunks."""
    lines = ["Product Catalog."]
    for i in range(100):
        num = f"{i + 1:03d}"                 # zero-padded so 'Product 037' is unique
        cat = _CATEGORIES[i % 4]
        price = 20 + (i * 7) % 130
        stock = (i * 3) % 50
        lines.append(f"Product {num} - Category: {cat} - Price: ${price} - Stock: {stock}")
    return "\n".join(lines)


_HANDBOOK = (
    "Acme Corp Employee Handbook. "
    "The standard work week is 40 hours. "
    "Employees accrue 15 days of paid vacation per year. "
    "Health insurance is provided through BlueShield. "
    "The office is located at 123 Market Street, San Francisco."
)

_INVOICE = (
    "Invoice #5567. "
    "Vendor: Bright Landscaping. Vendor email: contact@brightland.com. "
    "Customer: Smith Enterprises. "
    "Subtotal: $500. Discount: $50. Tax: $40. Total: $490."
)

SAMPLE_DOCS = [
    Document(page_content=_HANDBOOK, metadata={"doc_name": "handbook.txt"}),
    Document(page_content=_build_catalog(), metadata={"doc_name": "catalog.txt"}),
    Document(page_content=_INVOICE, metadata={"doc_name": "invoice.txt"}),
]


# --- Test cases -----------------------------------------------------------
# ctx: a phrase that should appear in the RETRIEVED chunks (recall)
# ans: a phrase that should appear in the FINAL answer (correctness)
# Product 037 -> index 36 -> 36%4=0 -> Tools ; Product 008 -> index 7 -> 7%4=3 -> Office
TEST_CASES = [
    {"q": "How many vacation days do employees get?", "doc": "handbook.txt",
     "ctx": "15 days", "ans": "15", "cat": "lookup"},
    {"q": "Where is the office located?", "doc": "handbook.txt",
     "ctx": "123 Market Street", "ans": "123 Market Street", "cat": "lookup"},
    {"q": "How long is the standard work week?", "doc": "handbook.txt",
     "ctx": "40 hours", "ans": "40", "cat": "lookup"},
    {"q": "What category is Product 037 in?", "doc": "catalog.txt",
     "ctx": "Product 037", "ans": "Tools", "cat": "lookup"},
    {"q": "What category is Product 008 in?", "doc": "catalog.txt",
     "ctx": "Product 008", "ans": "Office", "cat": "lookup"},
    {"q": "How many products are in the Tools category?", "doc": "catalog.txt",
     "ctx": "Tools", "ans": "25", "cat": "aggregation"},
    {"q": "How many products are in the Electronics category?", "doc": "catalog.txt",
     "ctx": "Electronics", "ans": "25", "cat": "aggregation"},
    {"q": "What is the total?", "doc": "invoice.txt",
     "ctx": "490", "ans": "490", "cat": "lookup"},
    {"q": "What is the vendor's email?", "doc": "invoice.txt",
     "ctx": "contact@brightland.com", "ans": "contact@brightland.com", "cat": "lookup"},
    {"q": "What is the discount?", "doc": "invoice.txt",
     "ctx": "Discount", "ans": "50", "cat": "lookup"},
    {"q": "Who is the seller?", "doc": "invoice.txt",
     "ctx": "Bright Landscaping", "ans": "Bright Landscaping", "cat": "synonym"},
]


# --- Scoring --------------------------------------------------------------

def _normalize(s: str) -> str:
    return s.lower().replace("$", "").replace(",", "").strip()


def _contains(text: str, expected) -> bool:
    norm = _normalize(text)
    if isinstance(expected, str):
        expected = [expected]
    return any(_normalize(e) in norm for e in expected)


# --- Runner ---------------------------------------------------------------

def main():
    embeddings = get_embeddings()
    vs = Chroma(                       # in-memory: no persist_directory -> fresh each run
        collection_name="eval",
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": settings.DISTANCE_METRIC},
    )
    chunks = chunk_documents(SAMPLE_DOCS)
    vs.add_documents(chunks)
    llm = get_llm("text")

    print(f"\nCorpus: {len(SAMPLE_DOCS)} documents, {len(chunks)} chunks\n")
    print(f"{'category':<12} {'doc':<13} {'rec':^4} {'ans':^4}  question")
    print("-" * 84)

    results = []
    for c in TEST_CASES:
        retriever = get_retriever(vs, doc_name=c["doc"])
        retrieved = " ".join(d.page_content for d in retriever.invoke(c["q"]))
        recall = _contains(retrieved, c["ctx"])

        answer = build_qa_chain(retriever, llm).invoke(c["q"])
        correct = _contains(answer, c["ans"])

        results.append((c, recall, correct, answer))
        print(f"{c['cat']:<12} {c['doc']:<13} "
              f"{'Y' if recall else 'N':^4} {'Y' if correct else 'N':^4}  {c['q']}")

    n = len(results)
    rec = sum(1 for _, r, _, _ in results if r)
    ans = sum(1 for _, _, a, _ in results if a)
    print("\n=== Summary ===")
    print(f"Context recall:     {rec}/{n}  ({round(100 * rec / n)}%)")
    print(f"Answer correctness: {ans}/{n}  ({round(100 * ans / n)}%)")

    print("\nBy category:")
    for cat in sorted({c["cat"] for c, _, _, _ in results}):
        rows = [(r, a) for c, r, a, _ in results if c["cat"] == cat]
        cr = sum(1 for r, _ in rows if r)
        ca = sum(1 for _, a in rows if a)
        print(f"  {cat:<12} recall {cr}/{len(rows)}   answer {ca}/{len(rows)}")

    misses = [(c, answer) for c, _, correct, answer in results if not correct]
    if misses:
        print("\nAnswer misses (expected vs got):")
        for c, answer in misses:
            got = " ".join(answer.split())[:90]
            print(f"  [{c['cat']}] {c['q']}")
            print(f"      expected {c['ans']!r}  ::  got {got!r}")
    print()


if __name__ == "__main__":
    main()