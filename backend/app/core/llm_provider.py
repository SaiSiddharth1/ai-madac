"""
LLM Provider Abstraction Layer.
Allows seamless switching between OpenAI, Google Gemini, Anthropic, or local models
without changing agent implementation code.
"""

import logging
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Instantiate and return a LangChain LLM instance based on configuration.

    Args:
        provider: 'openai', 'google', or 'anthropic' (defaults to settings.LLM_PROVIDER)
        model_name: specific model name (defaults to settings.LLM_MODEL)
        temperature: randomness (default 0.0 for deterministic agent routing/SQL)

    Returns:
        BaseChatModel instance.
    """
    selected_provider = (provider or settings.LLM_PROVIDER).lower()
    selected_model = model_name or settings.LLM_MODEL

    logger.debug(f"Initializing LLM: provider='{selected_provider}', model='{selected_model}', temp={temperature}")

    if selected_provider == "openai":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "sk-your-openai-key":
            logger.warning("OPENAI_API_KEY is not configured properly.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
        )

    elif selected_provider in ("google", "gemini"):
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not configured.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=selected_model or "gemini-1.5-flash",
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    elif selected_provider == "anthropic":
        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY is not configured.")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=selected_model or "claude-3-5-sonnet-20240620",
            temperature=temperature,
            api_key=settings.ANTHROPIC_API_KEY,
        )

    else:
        logger.warning(f"Unknown provider '{selected_provider}', defaulting to OpenAI")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
        )
