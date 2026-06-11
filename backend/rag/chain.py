"""LCEL RAG chains.

- build_qa_chain: returns just the answer string (used by the CLI and the eval
  harness, which don't need sources).
- build_qa_chain_with_sources: returns {"answer", "docs", "question"} so the UI
  can show which documents/pages an answer was grounded in (Phase 6 citations).
- build_condense_question_chain / build_conversational_chain_with_sources:
  Phase 7 conversation memory - reformulate a follow-up into a standalone
  question using chat history, then run the unchanged Phase 6 chain on it.

Both read each retrieved Document's page_content (NOT its metadata), so anything
that must be answerable has to live in page_content. The system prompt stays
grounded but recognizes reasonable synonyms across domains.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch,
)


_SYSTEM_PROMPT = (
    "You answer questions about a document using only the information in the "
    "context below.\n"
    "The question may use different words than the document for the same thing. "
    "Recognize reasonable synonyms and equivalent phrasings, and answer if the "
    "information is present in any form - even if the exact word in the question "
    "does not appear in the text. For example, a question about the 'owner' or "
    "'seller' can be answered from text that says 'vendor'; a question about the "
    "'writer' from text that says 'author'; a question about the 'talk' from a "
    "'transcript'. When you rely on such an equivalent, briefly note the term the "
    "document actually uses.\n"
    "If the information is genuinely not present in the context, say you don't "
    "know. Never invent facts, and do not assume equivalences that are not "
    "reasonable."
)


def _make_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_qa_chain(retriever, llm):
    """Returns a chain that produces just the answer string."""
    prompt = _make_prompt()
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def build_qa_chain_with_sources(retriever, llm):
    """Returns a chain that produces {"docs", "question", "answer"}.

    Retrieve once, keep the docs, generate the answer from them, and return both
    - so the UI can cite the sources the answer was grounded in.
    """
    prompt = _make_prompt()

    # docs = retrieved chunks; question passes through unchanged
    retrieve = RunnableParallel(docs=retriever, question=RunnablePassthrough())

    # add `answer` to the dict WITHOUT dropping docs/question
    generate = RunnablePassthrough.assign(
        answer=(
            RunnableLambda(lambda x: {
                "context": _format_docs(x["docs"]),
                "question": x["question"],
            })
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return retrieve | generate


def format_sources(docs) -> list[str]:
    """Distinct 'doc_name (page N)' labels from retrieved docs, for display.

    Collapses multiple chunks of the same page into one entry.
    """
    seen: list[str] = []
    for d in docs:
        name = d.metadata.get("doc_name", "unknown")
        page = d.metadata.get("page_number")
        label = f"{name} (page {page})" if page else name
        if label not in seen:
            seen.append(label)
    return seen


# --- Phase 7: conversation memory (history-aware retrieval) ------------------

_CONDENSE_SYSTEM_PROMPT = (
    "Given the conversation so far and a follow-up question, rewrite the "
    "follow-up as a STANDALONE question that can be understood without the "
    "conversation. Resolve pronouns and references (e.g. 'it', 'that', 'the "
    "document', 'its') using the history. Do NOT answer the question - only "
    "rewrite it. If the question is already standalone, return it unchanged."
)


def _make_condense_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", _CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])


def build_condense_question_chain(llm):
    """Returns a chain: {"question", "chat_history"} -> standalone question (str).

    Short-circuits when there is no history (turn 1): the first question is
    passed straight through, so no extra LLM call is spent rewriting a question
    that is already standalone.
    """
    prompt = _make_condense_prompt()
    rewrite = prompt | llm | StrOutputParser()
    return RunnableBranch(
        # no history -> return the question unchanged (no LLM call)
        (lambda x: not x.get("chat_history"), RunnableLambda(lambda x: x["question"])),
        # otherwise -> rewrite the follow-up into a standalone question
        rewrite,
    )


def build_conversational_chain_with_sources(retriever, llm):
    """Conversation-aware chain with citations.

    Input:  {"question": str, "chat_history": list[BaseMessage]}
    Output: {"docs", "question", "answer"} - same shape as Phase 6.

    Reformulates the follow-up into a standalone query (using chat_history),
    then runs the UNCHANGED Phase 6 retrieve+answer+sources chain on the
    resulting string. Memory lives entirely in the reformulation step; the
    answer prompt still sees only {context, question}.
    """
    condense = build_condense_question_chain(llm)   # {question, chat_history} -> str
    qa = build_qa_chain_with_sources(retriever, llm)  # str -> {docs, question, answer}
    return condense | qa