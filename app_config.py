import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def get_groq_api_key() -> str:
    """
    Safely retrieves the GROQ_API_KEY from environment variables or Streamlit secrets.
    Never crashes the application or halts execution if secrets are missing or unreadable.
    """
    # 1. First priority: Environment variable (e.g. from .env file or OS environment)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and str(api_key).strip():
        return str(api_key).strip()

    # 2. Second priority: Streamlit secrets (safely wrapped against all exceptions)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                secret_key = st.secrets.get("GROQ_API_KEY", "")
                if isinstance(secret_key, str) and secret_key.strip():
                    return secret_key.strip()
            except Exception as e:
                logger.debug("Streamlit secrets could not be read: %s", e)
    except Exception as e:
        logger.debug("Streamlit not available for secrets retrieval: %s", e)

    return ""


def is_groq_available() -> bool:
    """Returns True if a valid non-empty GROQ_API_KEY is configured."""
    key = get_groq_api_key()
    return bool(key and key.strip())


# Module-level constant maintained for backward compatibility
GROQ_API_KEY = get_groq_api_key()
