"""POST /query - the core endpoint.

Wraps the Phase 7 conversational chain unchanged: it builds a retriever scoped
to `scope` (per-document filtering), converts the client's chat history into
LangChain messages (windowed server-side, since the window policy belongs with
the logic), runs the chain, and returns the answer plus its source labels.
"""
from fastapi import APIRouter, Depends
from langchain_core.messages import HumanMessage, AIMessage

from backend.api import deps
from backend.api.schemas import QueryRequest, QueryResponse
from backend.config import settings
from backend.rag.retriever import get_retriever, get_hybrid_retriever
from backend.rag.chain import (
    build_conversational_chain_with_sources, format_sources,
)

router = APIRouter(tags=["query"])


def _to_messages(chat_history, window_turns):
    """Last `window_turns` turns (user+assistant pairs) -> LangChain messages."""
    recent = chat_history[-(window_turns * 2):]
    messages = []
    for turn in recent:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    vectorstore=Depends(deps.vectorstore),
    llm=Depends(deps.text_llm),
):
    doc_filter = req.scope or None  # "" or None -> all documents
    build = get_hybrid_retriever if settings.HYBRID_SEARCH else get_retriever
    retriever = build(vectorstore, doc_name=doc_filter)
    chain = build_conversational_chain_with_sources(retriever, llm)
    history = _to_messages(req.chat_history, settings.HISTORY_WINDOW)
    result = chain.invoke({"question": req.question, "chat_history": history})
    return QueryResponse(
        answer=result["answer"],
        sources=format_sources(result["docs"]),
    )