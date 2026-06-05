"""LCEL RAG chain: prompt -> LLM -> StrOutputParser, fed by an MMR retriever.

The chain reads each retrieved Document's page_content (NOT its metadata), so
anything that must be answerable has to live in page_content.

System prompt note: it stays grounded ("use only the context", "never invent
facts") to prevent hallucination, but explicitly allows recognizing reasonable
synonyms (vendor/seller/owner; customer/buyer/client). Without that, the model
was refusing to answer "owner details" from text that said "vendor" - a
generation-strictness failure, not a retrieval failure.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


_SYSTEM_PROMPT = (
    "You answer questions about a document using only the information in the "
    "context below.\n"
    "The question may use different words than the document for the same thing. "
    "Recognize reasonable equivalents and answer if the information is present in "
    "any form - for example 'vendor', 'seller', 'supplier', 'company', and 'owner' "
    "may all refer to the business that issued an invoice, and 'customer', 'buyer', "
    "and 'client' may all refer to its recipient. When you rely on such an "
    "equivalent, briefly note the term the document actually uses.\n"
    "If the information is genuinely not present in the context, say you don't know. "
    "Never invent facts, and do not assume equivalences that are not reasonable."
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