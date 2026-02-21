from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
from tenacity import retry, stop_after_attempt, wait_exponential

from app.configs import settings
from app.logging import logger

logger = logger.getChild("vector_store")


class VectorStore:
    _embeddings: OpenAIEmbeddings = None
    _vector_stores: dict = {}

    @classmethod
    def get_client(cls) -> QdrantClient:
        return QdrantClient(
            location=settings.QDRANT_LOCATION,
            port=settings.QDRANT_REST_PORT,
            grpc_port=settings.QDRANT_GRPC_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=True,
            prefer_grpc=settings.QDRANT_PREFER_GRPC,
            timeout=60,
        )

    @classmethod
    def get_embeddings(cls) -> OpenAIEmbeddings:
        if cls._embeddings is None:
            cls._embeddings = OpenAIEmbeddings(
                model=settings.OPENAI_EMBEDDING_MODEL,
                dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS,
                disallowed_special=(),
            )

        return cls._embeddings

    @classmethod
    def get_vector_store(
        cls, collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION
    ) -> QdrantVectorStore:
        client = cls.get_client()

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=cls.get_embeddings(),
        )

        return vector_store

    @classmethod
    def get_retriever(
        cls,
        collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION,
        search_type: str = "similarity",
        search_kwargs: dict = None,
    ) -> VectorStoreRetriever:
        search_kwargs = search_kwargs if search_kwargs else {}

        if search_kwargs.get("filter"):
            search_kwargs["filter"] = convert_to_qdrant_filter(search_kwargs["filter"])

        return cls.get_vector_store(collection_name).as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

    @classmethod
    def delete_by_project_id(
        cls,
        project_id: str,
        collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION,
    ):

        logger.info(f"Deleting vectors from Qdrant for project_id: {project_id}")
        client = cls.get_client()

        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.project_id", match=MatchValue(value=project_id)
                )
            ]
        )

        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=delete_filter),
            wait=True,
        )

    @classmethod
    def delete_by_flag_id(
        cls, flag_id: str, collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION
    ):
        logger.info(f"Deleting vectors from Qdrant for flag_id: {flag_id}")
        client = cls.get_client()

        delete_filter = Filter(
            must=[
                FieldCondition(key="metadata.flag_id", match=MatchValue(value=flag_id))
            ]
        )

        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=delete_filter),
            wait=True,
        )

    @classmethod
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.75, min=2, max=10),
        reraise=True,
    )
    def add_documents(
        cls,
        docs: list[Document],
        collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION,
    ) -> list[str]:
        return cls.get_vector_store(collection_name).add_documents(docs)

    @classmethod
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.75, min=2, max=10),
        reraise=True,
    )
    def check_flag_id_exists(
        cls, flag_id: str, collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION
    ) -> bool:
        """
        Check if any vector data exists in Qdrant for a given flag_id.
        Returns True if found, False otherwise.
        """
        logger.info(f"Checking if vectors exist for flag_id: {flag_id}")

        client = cls.get_client()

        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.flag_id",
                    match=MatchValue(value=flag_id),
                )
            ]
        )

        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=2,
        )

        if points:
            logger.info(f"✅ Vector data found for flag_id: {flag_id}")
            return True
        else:
            logger.info(f"❌ No vector data found for flag_id: {flag_id}")
            return False

    @classmethod
    def get_all_by_filter(
        cls,
        filters: dict,
        collection_name: str = settings.QDRANT_EMBEDDING_COLLECTION,
        limit: int = 100,
    ) -> list[Document]:
        """Retrieve all documents matching the given filters using scroll.

        This does not perform similarity search - it returns all matching documents.
        """
        client = cls.get_client()
        scroll_filter = convert_to_qdrant_filter(filters)

        all_documents = []
        offset = None

        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                metadata = point.payload.get("metadata", {})
                page_content = point.payload.get("page_content", "")
                all_documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

            if offset is None:
                break

        return all_documents


def convert_to_qdrant_filter(filters: dict, metadata_prefix="metadata"):
    conditions = []

    for key, value in filters.items():
        field_key = f"{metadata_prefix}.{key}" if metadata_prefix else key

        if isinstance(value, list):
            conditions.append(
                FieldCondition(
                    key=field_key,
                    match=MatchAny(any=value),
                )
            )
        else:
            conditions.append(
                FieldCondition(key=field_key, match=MatchValue(value=value))
            )

    return Filter(must=conditions)


def initialize_qdrant():
    logger.info("Initializing Qdrant")
    client = VectorStore.get_client()

    if not client.collection_exists(settings.QDRANT_EMBEDDING_COLLECTION):
        logger.info("Creating Qdrant collection")
        client.create_collection(
            collection_name=settings.QDRANT_EMBEDDING_COLLECTION,
            vectors_config=VectorParams(
                size=settings.OPENAI_EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
                on_disk=True,
                hnsw_config=HnswConfigDiff(m=32, ef_construct=256, on_disk=False),
            ),
            on_disk_payload=True,
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True,
                )
            ),
        )

        logger.info("Creating flag id index")
        client.create_payload_index(
            collection_name=settings.QDRANT_EMBEDDING_COLLECTION,
            field_name="metadata.flag_id",
            field_type=PayloadSchemaType.UUID,
            wait=True,
        )

        logger.info("Creating project id index")
        client.create_payload_index(
            collection_name=settings.QDRANT_EMBEDDING_COLLECTION,
            field_name="metadata.project_id",
            field_type=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    else:
        logger.info("Qdrant collection already exists")

    logger.info("Qdrant initialization complete")
