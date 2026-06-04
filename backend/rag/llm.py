"""LLM factory.

Picks the right provider for each task:

    "text"   -> Groq (Llama 3.3 70B). Fast, free-tier-friendly. Used for the
                text RAG chain across Phases 1, 2, and YouTube.

    "vision" -> Gemini 2.5 Flash. Multimodal. Used in Phase 3 for invoices,
                forms, images, and complex/handwritten scans.

The rest of the codebase calls get_llm(task) and does not import providers
directly. Switching providers later is a one-file change here.

Note: text and vision use SEPARATE output-token budgets. Per-query answers
(text) stay short; image descriptions (vision) are a one-time ingestion cost
and get a larger budget so they can be complete.
"""
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.config import settings


def get_llm(task: str = "text"):
    if task == "text":
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com/keys "
                "and add it to your .env file."
            )
        return ChatGroq(
            model=settings.GROQ_TEXT_MODEL,
            temperature=0,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
            api_key=settings.GROQ_API_KEY,
        )

    if task == "vision":
        if not settings.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com/apikey "
                "and add it to your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_VISION_MODEL,
            temperature=0,
            max_output_tokens=settings.VISION_MAX_OUTPUT_TOKENS,
        )

    raise ValueError(f"Unknown LLM task: {task!r}. Use 'text' or 'vision'.")