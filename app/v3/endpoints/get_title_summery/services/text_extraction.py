import os
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException, status
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from libsbml import SBMLReader, writeSBMLToString
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter

from app.configs import settings
from app.core.vector_store import VectorStore
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.utils.embedding import insert_cost_document

SUPPORTED_EXTENSIONS = {
    ".m",
    ".r",
    ".R",
    ".py",
    ".json",
    ".csv",
    ".tsv",
    ".txt",
    ".js",
    ".md",
}
XML_EXTENSIONS = {".xml", ".sbml"}
EXCEL_EXTENSIONS = {".xls", ".xlsx"}

LANGUAGE_MAP = {
    ".m": "matlab",
    ".r": "r",
    ".R": "r",
    ".py": "python",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".xml": "xml",
    ".sbml": "xml",
    ".txt": "text",
    ".js": "javascript",
    ".md": "markdown",
    ".xls": "excel",
    ".xlsx": "excel",
}


def extract_text_generic(file_location, encoding="utf-8"):
    """
    Generic text/code extraction using LangChain TextLoader.
    """
    try:
        loader = TextLoader(file_location, encoding=encoding)
        data = loader.load()
        texts = [str(doc) for doc in data]
        return data, texts
    except Exception as e:
        logger.exception(f"Text extraction failed for {file_location}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract text: {e}",
        )


def extract_text_from_xml_file(file_location):
    """
    Extracts text and XML content (supports SBML parsing).
    """
    try:
        loader = TextLoader(file_location)
        data = loader.load()

        reader = SBMLReader()
        document = reader.readSBML(file_location)
        sbml_as_string = writeSBMLToString(document)
        return data, sbml_as_string
    except Exception as e:
        logger.exception(f"XML/SBML extraction failed for {file_location}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


def extract_text_from_excel(file_location: str):
    """
    Extracts structured text from Excel sheets.
    Each row becomes a chunk, and headers are repeated in each row.
    Supports multi-sheet Excel.

    Note: For very large Excel files (100k+ rows), this may consume
    significant memory.
    """
    try:
        excel_data = pd.ExcelFile(file_location)
        all_docs = []
        combined_texts = []

        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(excel_data, sheet_name=sheet_name)
            df = df.fillna("")  # Replace NaN with empty string

            headers = df.columns.tolist()
            sheet_texts = []  # store rows only for this sheet

            for _, row in df.iterrows():
                row_dict = dict(zip(headers, row.tolist()))
                row_text = "\n".join(
                    [f"{k}: {v}" for k, v in row_dict.items() if str(v).strip()]
                )
                chunk_text = f"Sheet: {sheet_name}\n{row_text}"
                sheet_texts.append(chunk_text)
                combined_texts.append(chunk_text)

            # store each sheet as a Document
            all_docs.append(Document(page_content="\n\n".join(sheet_texts)))

        return all_docs, combined_texts

    except Exception as e:
        logger.exception(f"Excel extraction failed for {file_location}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract Excel data: {e}",
        )


def ingest_text_txt(
    data, project_id, flag_id, file_type, user_id, supp_id, is_excel=False
):
    """
    Ingest plain text (e.g., TXT or Excel) into the VectorStore.
    Excel files are handled row-wise without SentenceSplitter.
    """
    if not is_excel:
        text_splitter = SentenceSplitter.from_defaults(
            chunk_size=settings.EMBEDDING_CHUNK_SIZE,
            chunk_overlap=settings.EMBEDDING_CHUNK_OVERLAP,
        )

    all_chunks = []
    for doc in data:
        text_content = doc.page_content
        if is_excel:
            # For Excel, treat each row as one chunk
            chunks = text_content.split("\n\n")
            all_chunks.extend(chunks)
        else:
            if len(text_content.strip()) >= 100:
                chunks = text_splitter.split_text(text_content)
                all_chunks.extend(chunks)
            else:
                all_chunks.append(text_content)

    # Embedding and cost
    embedding_token = sum(len(chunk.split()) for chunk in all_chunks)
    cost = (settings.COST_PER_1000_TOKENS / 1000) * embedding_token
    insert_cost_document(user_id, embedding_token, cost)

    langchain_docs = []
    for chunk in all_chunks:
        metadata = {
            "project_id": project_id,
            "flag_id": flag_id,
            "user_id": user_id,
            "file_type": file_type,
            "supplementary_id": supp_id if supp_id else None,
        }
        langchain_docs.append(
            Document(id=str(uuid4()), page_content=chunk, metadata=metadata)
        )

    VectorStore.add_documents(langchain_docs)


def ingest_text(data, language, project_id, flag_id, file_type, user_id, supp_id):
    """
    Ingest structured text or code files using CodeSplitter.
    """
    text_splitter = CodeSplitter.from_defaults(language=language)

    all_chunks = []
    for doc in data:
        text_content = doc.page_content
        if len(text_content.strip()) >= 100:
            chunks = text_splitter.split_text(text_content)
            all_chunks.extend(chunks)
        else:
            all_chunks.append(text_content)

    embedding_token = sum(len(chunk.split()) for chunk in all_chunks)
    cost = (settings.COST_PER_1000_TOKENS / 1000) * embedding_token
    insert_cost_document(user_id, embedding_token, cost)

    langchain_docs = []
    for chunk in all_chunks:
        metadata = {
            "project_id": project_id,
            "flag_id": flag_id,
            "user_id": user_id,
            "file_type": file_type,
            "supplementary_id": supp_id if supp_id else None,
        }
        langchain_docs.append(
            Document(id=str(uuid4()), page_content=chunk, metadata=metadata)
        )

    VectorStore.add_documents(langchain_docs)


def text_fileprocessing(file_location, project_id, flag_id, user_id, supp_id):
    """
    Handles text, code, XML, and Excel file ingestion end-to-end.
    """
    file_extension = os.path.splitext(file_location)[-1].lower()
    language = LANGUAGE_MAP.get(file_extension, "text")

    if file_extension in SUPPORTED_EXTENSIONS:
        encoding = "latin-1" if file_extension == ".m" else "utf-8"
        data, response = extract_text_generic(file_location, encoding=encoding)

    elif file_extension in XML_EXTENSIONS:
        data, response = extract_text_from_xml_file(file_location)

    elif file_extension in EXCEL_EXTENSIONS:
        data, response = extract_text_from_excel(file_location)

    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file_extension}",
        )

    file_type = "code"
    if file_extension in [".txt", ".xls", ".xlsx"]:
        ingest_text_txt(
            data,
            project_id,
            flag_id,
            file_type,
            user_id,
            supp_id,
            is_excel=(file_extension in [".xls", ".xlsx"]),
        )
    else:
        ingest_text(data, language, project_id, flag_id, file_type, user_id, supp_id)

    return response
