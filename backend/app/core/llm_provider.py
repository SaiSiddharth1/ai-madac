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
    configured_provider = (provider or settings.LLM_PROVIDER or "").lower()

    # If OpenAI is configured/requested but key is missing while Google API key exists, switch to Google
    if configured_provider == "openai" and (not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "sk-your-openai-key") and settings.GOOGLE_API_KEY:
        logger.info("OPENAI_API_KEY is not configured, but GOOGLE_API_KEY is present. Using Google Gemini provider.")
        selected_provider = "google"
    elif configured_provider in ("google", "gemini"):
        selected_provider = "google"
    elif configured_provider == "anthropic":
        selected_provider = "anthropic"
    elif configured_provider == "openai":
        selected_provider = "openai"
    else:
        # Default to google if GOOGLE_API_KEY is present, else openai
        selected_provider = "google" if settings.GOOGLE_API_KEY else "openai"

    selected_model = model_name or settings.LLM_MODEL
    logger.debug(f"Initializing LLM: provider='{selected_provider}', model='{selected_model}', temp={temperature}")

    if selected_provider in ("google", "gemini"):
        if not settings.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY is not configured.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = selected_model
        if not target_model or "gpt" in target_model or target_model in ("gemini-1.5-flash", "gemini-3.8-flash"):
            target_model = "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            max_retries=2,
            timeout=15,
        )

    elif selected_provider == "openai":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "sk-your-openai-key":
            logger.warning("OPENAI_API_KEY is not configured properly.")
        from langchain_openai import ChatOpenAI
        target_model = selected_model if (selected_model and "gemini" not in selected_model) else "gpt-4o-mini"
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
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
        logger.warning(f"Unknown provider '{selected_provider}', falling back to Google Gemini if available")
        if settings.GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=temperature,
                google_api_key=settings.GOOGLE_API_KEY,
                max_retries=2,
                timeout=15,
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=settings.OPENAI_API_KEY,
        )

