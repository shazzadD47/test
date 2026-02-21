import concurrent.futures
import json
import os

import pdf2doi
from fastapi import HTTPException, status
from feedparser.util import FeedParserDict
from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfparser import PDFParser

from app.configs import settings
from app.core.celery.app import celery_app
from app.v3.endpoints.get_title_summery.exceptions import (
    DoiExtractionFailed,
    EncryptionError,
)
from app.v3.endpoints.get_title_summery.logging import logger
from app.v3.endpoints.get_title_summery.services.text_extraction import (
    text_fileprocessing,
)
from app.v3.endpoints.get_title_summery.utils.payload_utils import (
    fix_paper_metadata_response,
)
from app.v3.endpoints.get_title_summery.utils.utils import (
    clean_jats_tags,
)

pdf2doi.config.set("verbose", False)
pdf2doi.config.set("separator", os.sep)


def extract_data_from_pdf_preprocess(
    file_location: str,
    flag_id: str,
    project_id: str,
    user_id: str,
    file_id: str = None,
    file_extension: str = None,
    image_uploaded: bool = False,
):
    logger.info(
        f"Extracting data from {file_location} "
        f"with file_extension: {file_extension} "
        f"and image_uploaded: {image_uploaded}"
    )
    if file_extension in [".pdf", ".ipynb"]:
        if not image_uploaded:
            celery_app.send_task(
                "convert_pdf_to_image_task",
                (file_location, flag_id, project_id, None, file_id),
            )

        result = celery_app.send_task(
            "process_layout_chunks",
            (
                file_location,
                project_id,
                flag_id,
                user_id,
                None,
                file_id,
                file_extension,
            ),
        )
        task_id = result.id
    else:
        result = celery_app.send_task(
            "extract_text_task",
            (file_location, project_id, flag_id, user_id, None),
        )
        task_id = result.id

    try:
        metadata = extract_title_doi_abstract(
            file_location, project_id, flag_id, user_id
        )

        if isinstance(metadata, tuple):
            metadata, _ = metadata
    except Exception as e:
        logger.exception(
            f"[flag_id: {flag_id}] Error extracting title, abstract and DOI"
            f"Error: {str(e)}"
        )
        metadata = {
            "title": "",
            "abstract": "",
            "DOI": "",
            "DOI_URL": "",
            "extra_metadata": {},
        }

    metadata["flag_id"] = flag_id
    metadata["paper_exists"] = False
    if "ai_metadata" not in metadata:
        metadata["ai_metadata"] = {"task_id": task_id}
    else:
        metadata["ai_metadata"]["task_id"] = task_id

    return metadata


def extract_data_from_file_preprocess_sup(
    file_location: str,
    flag_id: str,
    supp_id: str,
    project_id: str,
    user_id: str,
    file_extension: str,
    image_uploaded: bool = False,
):
    supp_flag_id = f"{flag_id}-supplementary-{supp_id}"
    logger.info(
        f"Extracting data from {file_location} "
        f"with file_extension: {file_extension} "
        f"and image_uploaded: {image_uploaded}"
    )
    if file_extension in [".pdf", ".ipynb"]:
        if not image_uploaded:
            celery_app.send_task(
                "convert_pdf_to_image_task",
                (file_location, flag_id, project_id, supp_id, None),
            )
        celery_app.send_task(
            "process_layout_chunks",
            (
                file_location,
                project_id,
                flag_id,
                user_id,
                supp_id,
                None,
                file_extension,
            ),
        )

    else:
        celery_app.send_task(
            "extract_text_task",
            (file_location, project_id, flag_id, user_id, supp_id),
        )

    response = {
        "flag_id": supp_flag_id,
    }

    return response


def extract_title_doi_abstract(
    file_path: str, project_id: str, flag_id: str, user_id: str, file_id: str = None
):
    extra_metadata = {}
    title, abstract, doi, doi_url = "", "", "", ""

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(pdf2doi.pdf2doi_singlefile, file_path)
            try:
                metadata = future.result(timeout=settings.PDF_DOI_TIME_OUT)
            except concurrent.futures.TimeoutError:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                return {
                    "title": "",
                    "abstract": "",
                    "DOI": "",
                    "DOI_URL": "",
                    "extra_metadata": {},
                }

        if not metadata.get("identifier"):
            raise DoiExtractionFailed()

        if isinstance(metadata["validation_info"], FeedParserDict):
            extra_metadata = metadata["validation_info"]
        else:
            extra_metadata = json.loads(metadata["validation_info"])

        celery_app.send_task(
            "embedding_meta_task",
            (project_id, flag_id, user_id, extra_metadata),
        )

        title = clean_jats_tags(extra_metadata.get("title", ""))
        title = " ".join(title.split())

        abstract = clean_jats_tags(extra_metadata.get("abstract", ""))
        abstract = " ".join(abstract.split())

        if not abstract:
            abstract = clean_jats_tags(extra_metadata.get("summary", ""))
            abstract = " ".join(abstract.split())

        doi = extra_metadata.get("DOI", metadata.get("identifier", ""))
        doi_url = extra_metadata.get("URL", extra_metadata.get("link", ""))

        if not doi or not doi_url:
            logger.error("DOI or DOI URL not found")
            logger.error(
                f"Extract DOI info, Val info type: {type(metadata['validation_info'])}"
            )
            logger.error(f"Metadata: {metadata}")
            logger.error(f"Extra Metadata: {json.dumps(extra_metadata, indent=2)}")

    except DoiExtractionFailed:
        try:
            with open(file_path, "rb") as file:
                parser = PDFParser(file)
                document = PDFDocument(parser)
                title = document.info[0].get("Title", "")
        except (PDFPasswordIncorrect, PDFEncryptionError):
            raise EncryptionError()

    except Exception:
        logger.exception(
            f"[flag_id: {flag_id}] Error while extracting DOI and metadata"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error while extracting DOI and metadata",
        )

    response = {
        "title": title,
        "abstract": abstract,
        "DOI": doi,
        "DOI_URL": doi_url,
        "extra_metadata": extra_metadata,
    }
    logger.info(f"Metadata response: {response}")
    response = fix_paper_metadata_response(response)
    logger.info(f"Fixed metadata resposne: {response}")

    return response


def extract_text(
    file_location: str,
    project_id: str,
    flag_id: str,
    user_id: str,
    supp_id: str = None,
):
    logger.info(
        f"Extracting text from {file_location} "
        f"with project_id: {project_id} "
        f"and flag_id: {flag_id} "
        f"and user_id: {user_id} "
        f"and supp_id: {supp_id}"
    )
    try:
        text_fileprocessing(file_location, project_id, flag_id, user_id, supp_id)
    except Exception as e:
        logger.exception(
            f"[flag_id: {flag_id}] Error in extracting text from {file_location}"
        )

        raise HTTPException(status_code=500, detail=str(e))
