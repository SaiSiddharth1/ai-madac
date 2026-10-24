"""
Conversational Chat Router for AI-MADAC.
Routes user messages between CHAT (conversational assistant) and DATASET (analytical pipeline).
"""

import json
import logging
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_provider import get_llm

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """You are the Conversational Assistant for AI-MADAC, an autonomous data analytics platform.
Your task is to classify whether the user's message is a pure conversational message (CHAT) or a data analysis question (DATASET).

Classification Rules:
1. CHAT: Pure greetings, pleasantries, small talk, questions about your identity or capabilities, greetings with no data request, expressions of gratitude, or polite departures.
   Examples:
   - "hi", "hello", "hey", "hii", "hiii"
   - "good morning", "good afternoon", "good evening"
   - "how are you?", "how are you doing?", "what's up?"
   - "thanks", "thank you", "thanks a lot"
   - "bye", "goodbye", "see you later"
   - "who are you?", "what can you do?", "what is AI-MADAC?", "help"

2. DATASET: Any question or instruction requesting data retrieval, filtering, aggregation, calculation, metrics, columns, trends, charts, or analysis — EVEN IF it starts with a casual greeting or polite words.
   Examples:
   - "Hi, what is the total sales?" -> DATASET
   - "Hello, show me sales by category as a bar chart." -> DATASET
   - "Hey, what is the average profit?" -> DATASET
   - "Show monthly sales as a line chart." -> DATASET
   - "What are the columns?" -> DATASET
   - "How many rows are in this dataset?" -> DATASET

OUTPUT FORMAT:
If CHAT:
Provide a warm, polite, and helpful conversational response ready to assist with dataset analysis.
Return JSON:
{
  "mode": "CHAT",
  "response": "<your friendly conversational response>"
}

If DATASET:
Return JSON:
{
  "mode": "DATASET"
}
"""

# Regex for pure conversational greetings without data keywords
PURE_CHAT_PATTERNS = [
    r"^\s*hi+\s*[\.!\?]*\s*$",
    r"^\s*hello+\s*[\.!\?]*\s*$",
    r"^\s*hey+\s*[\.!\?]*\s*$",
    r"^\s*good\s*(?:morning|afternoon|evening|day)\s*[\.!\?]*\s*$",
    r"^\s*how\s+are\s+you(?:\s+doing)?\s*[\.!\?]*\s*$",
    r"^\s*what'?s\s+up\s*[\.!\?]*\s*$",
    r"^\s*thanks?(?:\s+(?:you|a lot|so much))?\s*[\.!\?]*\s*$",
    r"^\s*thank\s+you(?:\s+(?:so much|very much))?\s*[\.!\?]*\s*$",
    r"^\s*(?:bye|goodbye|see\s+you(?:\s+later)?)\s*[\.!\?]*\s*$",
    r"^\s*who\s+are\s+you\s*[\.!\?]*\s*$",
    r"^\s*what\s+can\s+you\s+do\s*[\.!\?]*\s*$",
    r"^\s*what\s+is\s+ai[-_\s]*madac\s*[\.!\?]*\s*$",
    r"^\s*help\s*[\.!\?]*\s*$",
]

# Keywords indicating analytical/dataset intent
DATASET_INDICATORS = [
    "sales", "revenue", "profit", "quantity", "price", "unitprice", "discount", "cost",
    "order", "customer", "product", "category", "country", "state", "city", "region",
    "date", "month", "year", "day", "week", "time", "trend",
    "chart", "plot", "graph", "histogram", "scatter", "bar", "line", "pie", "box",
    "total", "sum", "average", "mean", "avg", "min", "max", "count", "rows", "columns",
    "top", "bottom", "highest", "lowest", "distribution", "versus", "vs", "by", "per",
    "dataset", "table", "data", "records", "filter", "show", "calculate", "find"
]


def _get_fallback_chat_response(q_lower: str) -> str:
    """Natural fallback responses when LLM is unavailable."""
    if any(re.search(p, q_lower) for p in (r"how\s+are\s+you", r"what'?s\s+up")):
        return "I'm doing well! I'm ready to help you analyze your dataset. Feel free to ask questions or request charts."
    if any(re.search(p, q_lower) for p in (r"what\s+can\s+you\s+do", r"who\s+are\s+you", r"what\s+is\s+ai[-_\s]*madac", r"help")):
        return "I am AI-MADAC's conversational data science assistant. I can analyze your dataset, answer questions, calculate metrics, find trends, and create visualizations."
    if any(re.search(p, q_lower) for p in (r"thanks?", r"thank\s+you")):
        return "You're welcome! Let me know if you need any other analysis or visualizations."
    if any(re.search(p, q_lower) for p in (r"bye", r"goodbye", r"see\s+you")):
        return "Goodbye! Come back whenever you want to analyze your data."
    return "Hi! How can I help you with your data today? You can ask questions, calculate metrics, or request charts."


async def classify_chat_or_dataset(question: str, schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Classify whether a user's question is a pure conversational message (CHAT)
    or a dataset query (DATASET), using Gemini with robust deterministic fallback.
    """
    q_clean = question.strip()
    q_lower = q_clean.lower()

    # 1. Check if message has any clear dataset indicator keywords
    words = set(re.findall(r"[a-z0-9_]+", q_lower))
    has_dataset_keyword = any(ind in words for ind in DATASET_INDICATORS)
    
    # Check if any schema column names are in the query
    if schema and isinstance(schema, dict):
        col_names = [str(c.get("name", "")).lower() for c in schema.get("columns", []) if c.get("name")]
        if any(cn in q_lower for cn in col_names if len(cn) > 2):
            has_dataset_keyword = True

    # Fast match for pure greetings
    is_pure_greeting = any(bool(re.search(p, q_lower)) for p in PURE_CHAT_PATTERNS)

    if is_pure_greeting and not has_dataset_keyword:
        # It's definitely CHAT. Try Gemini for dynamic response, fallback to natural response.
        try:
            llm = get_llm(temperature=0.7)
            messages = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=q_clean),
            ]
            resp = await llm.ainvoke(messages)
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
            if parsed.get("mode") == "CHAT":
                return {"mode": "CHAT", "response": parsed.get("response", _get_fallback_chat_response(q_lower))}
        except Exception as e:
            logger.info(f"Gemini chat response fallback used: {e}")
            return {"mode": "CHAT", "response": _get_fallback_chat_response(q_lower)}

    # If it has dataset keywords, it's DATASET
    if has_dataset_keyword:
        return {"mode": "DATASET", "response": None}

    # For ambiguous or general statements, consult Gemini
    try:
        llm = get_llm(temperature=0.0)
        messages = [
            SystemMessage(content=CHAT_SYSTEM_PROMPT),
            HumanMessage(content=f"User Message: {q_clean}"),
        ]
        resp = await llm.ainvoke(messages)
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        mode = parsed.get("mode", "DATASET")
        if mode == "CHAT":
            return {"mode": "CHAT", "response": parsed.get("response", _get_fallback_chat_response(q_lower))}
        return {"mode": "DATASET", "response": None}
    except Exception as e:
        logger.warning(f"Gemini intent router failed: {e}. Falling back based on indicators.")
        if is_pure_greeting:
            return {"mode": "CHAT", "response": _get_fallback_chat_response(q_lower)}
        return {"mode": "DATASET", "response": None}
