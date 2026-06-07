"""LCEL RAG chain: prompt -> LLM -> StrOutputParser, fed by an MMR retriever.

The chain reads each retrieved Document's page_content (NOT its metadata), so
anything that must be answerable has to live in page_content.

System prompt: stays grounded ("use only the context", "never invent facts")
to prevent hallucination, but allows recognizing reasonable synonyms. The
principle is stated generally and illustrated with cross-domain examples (an
invoice term and a document term) so it applies to any input type - not just
invoices - while still benefiting from concrete examples.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def build_qa_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )