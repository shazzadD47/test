from app.configs import settings
from app.core.vector_store import VectorStore
from app.v3.endpoints.get_title_summery.utils.embedding import insert_cost_document
from app.v3.endpoints.get_title_summery.utils.metadata import metadata_to_str


class Document:
    def __init__(self, page_content, metadata):
        self.id = None
        self.page_content = page_content
        self.metadata = metadata


def process_metadata_and_save(
    project_id: str, flag_id: str, user_id: str, extra_metadata: dict
):
    processed_extra_metadata = metadata_to_str(
        extra_metadata,
        ignore_keys=["reference", "link"],
    )
    paper_meta_info = {
        "project_id": project_id,
        "flag_id": flag_id,
        "user_id": user_id,
        "file_type": "document",
        "supplementary_id": None,
    }
    chunks = [
        Document(content, paper_meta_info) for content in processed_extra_metadata
    ]

    embedding_token = len(chunks)
    cost_per_1000_tokens = settings.COST_PER_1000_TOKENS
    cost_per_single_tokens = cost_per_1000_tokens / 1000
    embedding_cost = embedding_token * cost_per_single_tokens
    insert_cost_document(user_id, embedding_token, embedding_cost)

    VectorStore.add_documents(chunks)

    return {
        "status": "success",
        "flag_id": flag_id,
        "project_id": project_id,
        "user_id": user_id,
        "supplementary_id": None,
        "chunks_embedded": len(chunks),
    }
