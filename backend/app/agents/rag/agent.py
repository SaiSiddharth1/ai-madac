"""
RAG Agent.
Retrieves semantic knowledge, relevant schema column descriptions, and domain context from Vector Store.
"""

import logging
from typing import Any

from app.agents.state import AgentState
from app.core.vector_store import vector_store_service

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent responsible for semantic document retrieval and context augmentation.
    """

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute vector similarity search for user question context.
        """
        question = state.get("question", "")
        dataset_id = state.get("dataset_id")

        logger.info(f"RAG Agent retrieving context for query: '{question}'")

        try:
            docs = vector_store_service.similarity_search(
                query=question, k=3, dataset_id=dataset_id
            )
            retrieved_chunks = [d.page_content for d in docs]

            agents_used = state.get("agents_used", []) + ["rag_agent"]

            return {
                "rag_result": {
                    "retrieved_context": retrieved_chunks,
                    "document_count": len(docs),
                },
                "agents_used": agents_used,
            }
        except Exception as e:
            logger.warning(f"RAG Agent execution error: {e}")
            return {"rag_result": None}


# Singleton instance
rag_agent = RAGAgent()
