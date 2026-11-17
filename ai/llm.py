import logging
from langchain_groq import ChatGroq
from app_config import get_groq_api_key

logger = logging.getLogger(__name__)


class LLMManager:
    """
    Centralized LLM manager with lazy initialization.
    Creates reusable Groq models on demand for the application.
    """

    def __init__(self):
        self._fast_llm = None
        self._reasoning_llm = None

    def get_fast_llm(self):
        """Returns the fast inference model (llama-3.1-8b-instant)."""
        api_key = get_groq_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or Streamlit secrets."
            )
        if self._fast_llm is None:
            self._fast_llm = ChatGroq(
                groq_api_key=api_key,
                model="llama-3.1-8b-instant",
                temperature=0,
            )
        return self._fast_llm

    def get_reasoning_llm(self):
        """Returns the deep reasoning model (llama-3.3-70b-versatile)."""
        api_key = get_groq_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env file or Streamlit secrets."
            )
        if self._reasoning_llm is None:
            self._reasoning_llm = ChatGroq(
                groq_api_key=api_key,
                model="llama-3.3-70b-versatile",
                temperature=0,
            )
        return self._reasoning_llm


llm_manager = LLMManager()
