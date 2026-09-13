"""
Vector Store Service.
Handles vector embedding and similarity search for dataset metadata, columns, and domain knowledge.
Uses ChromaDB for local persistent vector indexing.
"""

import logging
import os
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Vector Store manager for RAG knowledge indexing and retrieval.
    """

    def __init__(self):
        self.embeddings = None
        self._vector_store = None

    def _get_embeddings(self):
        if self.embeddings is None:
            # Fallback to default OpenAI embeddings or fake embeddings if unconfigured
            try:
                self.embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
            except Exception:
                from langchain_community.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=1536)
        return self.embeddings

    def get_vector_store(self, collection_name: str = "dataset_knowledge") -> Chroma:
        """
        Get or initialize ChromaDB vector store instance.
        """
        persist_directory = os.path.join(settings.DATA_DIR, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)

        return Chroma(
            collection_name=collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=persist_directory,
        )

    def index_dataset_schema(self, dataset_id: str, dataset_name: str, schema_info: dict[str, Any]):
        """
        Index a dataset's column names, types, and sample values into the vector store.
        """
        try:
            vs = self.get_vector_store()
            docs = []
            for col in schema_info.get("columns", []):
                col_name = col.get("name")
                dtype = col.get("dtype")
                samples = ", ".join(col.get("sample_values", []))

                text_content = f"Dataset: {dataset_name} | Column: {col_name} | Dtype: {dtype} | Samples: {samples}"
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "dataset_id": str(dataset_id),
                        "dataset_name": dataset_name,
                        "column_name": col_name,
                    },
                )
                docs.append(doc)

            if docs:
                vs.add_documents(docs)
                logger.info(f"Indexed {len(docs)} column document(s) in vector store for dataset {dataset_id}")
        except Exception as e:
            logger.warning(f"Vector store indexing skipped or failed: {e}")

    def similarity_search(self, query: str, k: int = 3, dataset_id: str | None = None) -> list[Document]:
        """
        Perform similarity search for a query string.
        """
        try:
            vs = self.get_vector_store()
            filter_dict = {"dataset_id": str(dataset_id)} if dataset_id else None
            results = vs.similarity_search(query, k=k, filter=filter_dict)
            return results
        except Exception as e:
            logger.warning(f"Vector store similarity search failed: {e}")
            return []

    def delete_dataset_schema(self, dataset_id: str) -> None:
        """Remove a dataset's indexed schema when the dataset is deleted."""
        try:
            vs = self.get_vector_store()
            vs.delete(where={"dataset_id": str(dataset_id)})
            logger.info("Removed vector-store entries for dataset %s", dataset_id)
        except Exception as e:
            # Vector retrieval is additive functionality; inability to clean it
            # up must not prevent a user deleting their primary dataset records.
            logger.warning("Vector-store cleanup skipped or failed: %s", e)


# Singleton instance
vector_store_service = VectorStoreService()
