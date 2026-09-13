"""
Workflow Event Manager for real-time Agent status reporting via Server-Sent Events (SSE).
"""

import asyncio
import logging
from typing import AsyncGenerator, Dict, Any

logger = logging.getLogger(__name__)


class WorkflowEventManager:
    """
    Manages active event queues for executing agent queries to support real-time SSE.
    """

    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def register(self, query_id: str) -> asyncio.Queue:
        """Register a new query_id queue."""
        queue = asyncio.Queue()
        self._queues[query_id] = queue
        logger.debug(f"Registered workflow queue for query: {query_id}")
        return queue

    def unregister(self, query_id: str):
        """Unregister and cleanup queue."""
        if query_id in self._queues:
            del self._queues[query_id]
            logger.debug(f"Unregistered workflow queue for query: {query_id}")

    async def emit(self, query_id: str, agent: str, status: str, message: str, **kwargs):
        """Emit a workflow event to the registered query queue."""
        if query_id in self._queues:
            event = {
                "query_id": query_id,
                "agent": agent,
                "status": status,
                "message": message,
                **kwargs
            }
            await self._queues[query_id].put(event)
            logger.debug(f"Workflow Event Emitted [{query_id}]: {agent} -> {status} ({message})")


# Singleton instance
workflow_event_manager = WorkflowEventManager()
