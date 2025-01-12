from typing import Any, Iterable, List, Literal, Optional, Tuple, Self

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
import turbopuffer as tpuf
import uuid


class TurbopufferVectorStore(VectorStore):
    """VectorStore backed by Turbopuffer."""

    _store = None
    _embedding: Embeddings

    def __init__(
        self,
        embedding: Embeddings,
        namespace: str,
        api_key: str,
    ) -> None:
        if not namespace:
            raise ValueError("namespace must be provided")
        if not api_key:
            raise ValueError("api_key must be provided")
        # TODO: support URLs
        self._namespace = tpuf.Namespace(namespace, api_key)
        self._embedding = embedding

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        namespace: str = "",
        api_key: str = "",
        **kwargs: Any,
    ) -> Self:
        _self: TurbopufferVectorStore = cls(
            embedding=embedding,
            namespace=namespace,
            api_key=api_key,
        )
        _self.add_texts(texts, metadatas, **kwargs)
        return _self

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        namespace: str = "",
        api_key: str = "",
        **kwargs: Any,
    ) -> Self:
        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        return cls.from_texts(texts, embedding, metadatas, namespace, api_key, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self._embedding.embed_documents(list(texts))
        ids = [str(uuid.uuid4().hex) for _ in texts]
        attributes = {
            "text": texts,
        }
        for key in metadatas[0].keys():
            items = [meta.get(key, None) for meta in metadatas]
            attributes[key] = items
        self._namespace.upsert(ids, embeddings, attributes)
        return ids

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self.add_texts(
            [document.page_content for document in documents],
            [document.metadata for document in documents],
            **kwargs,
        )

    def similarity_search_with_score_by_vector(
        self,
        query_vector: List[float],
        k: int = 10,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        results = self._namespace.query(
            vector=query_vector,
            distance_metric=distance_metric,
            top_k=k,
            include_attributes=["text"],
        )
        return [
            (
                Document(
                    page_content=res.attributes["text"],
                ),
                res.dist,
            )
            for res in results
        ]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
        **kwargs: Any,
    ) -> List[Document]:
        return [
            doc
            for doc, _score in self.similarity_search_with_score_by_vector(
                embedding, k, distance_metric, **kwargs
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query_vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            query_vector, k, distance_metric, **kwargs
        )

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        distance_metric: Literal[
            "cosine_distance", "euclidean_squared"
        ] = "cosine_distance",
        **kwargs: Any,
    ) -> List[Document]:
        query_vector = self._embedding.embed_query(query)
        return [
            doc
            for doc, _score in self.similarity_search_with_score_by_vector(
                query_vector, k, distance_metric, **kwargs
            )
        ]
