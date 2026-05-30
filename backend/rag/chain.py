"""LCEL RAG chain.

A small LangChain Expression Language (LCEL) pipeline that:
  1. Takes the user's question as input
  2. Sends it to the retriever to fetch relevant chunks
  3. Formats those chunks into a single context string
  4. Drops both into a ChatPromptTemplate (system + human roles)
  5. Calls the LLM
  6. Parses the response to a plain string

Composing this as a Runnable means later phases (chat memory, reranker,
citations) plug in as additional pipe steps instead of rewrites.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a document. "
    "Use ONLY the context below to answer. "
    "If the answer is not in the context, say you don't know - "
    "do not invent facts."
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def _format_docs(docs) -> str:
    """Join retrieved chunks into one context string with separators."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def build_qa_chain(retriever, llm):
    """Wire retriever + prompt + LLM into a single Runnable."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    return (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )