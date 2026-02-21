import mimetypes
import os
from uuid import uuid4

from boto3 import session

from app.configs import settings
from app.logging import logger


def upload_files_to_storage(filepath: str, upload_path: str):
    session_ = session.Session()
    object_key = f"{upload_path}/{os.path.basename(filepath)}"
    client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )
    try:
        with open(filepath, "rb") as file_object:
            endpoint = client.upload_fileobj(
                file_object,
                settings.S3_SPACES_BUCKET,
                object_key,
                ExtraArgs={
                    "ContentType": mimetypes.guess_type(filepath)[0],
                    "ACL": "public-read",
                },
            )
            logger.info(f"{filepath} uploaded, used endpoint {endpoint}")
    except Exception as e:
        logger.exception(f"Error in uploading {filepath}, {e}")


def upload_files_to_storage_by_name(filepath: str, upload_path: str, filename: str):
    client = session.Session().client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )

    object_key = f"{upload_path}/{filename}"
    try:
        with open(filepath, "rb") as file_object:
            client.upload_fileobj(
                file_object,
                settings.S3_SPACES_BUCKET,
                object_key,
                ExtraArgs={
                    "ContentType": mimetypes.guess_type(filepath)[0],
                    "ACL": "public-read",
                },
            )
    except Exception as e:
        logger.exception(f"Error in uploading {filepath}: {e}")


def download_files_from_storage(
    object_key: str,
    download_path: str,
):
    """
    Downloads a file from an S3 bucket.
    Args:
    object_key: str, key of the file in the S3 bucket
    download_path: str, local path to save the downloaded file
    """
    try:
        session_ = session.Session()
        s3_client = session_.client(
            "s3",
            endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
            aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
            aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
        )

        # Download the file
        s3_client.download_file(settings.S3_SPACES_BUCKET, object_key, download_path)
        logger.debug(f"File downloaded successfully to {download_path}")

    except Exception as e:
        logger.exception(f"Error in downloading {object_key}: {e}")


def delete_bucket_object(
    object_key: str,
):
    session_ = session.Session()
    s3_client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )
    s3_client.delete_object(
        Bucket=settings.S3_SPACES_BUCKET,
        Key=object_key,
    )


def upload_fileobject_to_storage(file_object, upload_path: str, extension: str):
    session_ = session.Session()
    object_key = f"{upload_path}/{uuid4()}-{uuid4()}.{extension.lower()}"
    client = session_.client(
        "s3",
        region_name=settings.S3_SPACES_REGION,
        endpoint_url=settings.S3_SPACES_ENDPOINT_URL,
        aws_access_key_id=settings.S3_SPACES_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SPACES_SECRET_KEY,
    )
    try:
        endpoint = client.upload_fileobj(
            file_object,
            settings.S3_SPACES_BUCKET,
            object_key,
            ExtraArgs={
                "ContentType": f"file/{extension.lower()}",
                "ACL": "public-read",
            },
        )
        logger.info(f"file object type:{extension} uploaded, used endpoint {endpoint}")
    except Exception as e:
        logger.exception(f"Error in uploading file object type:{extension}, {e}")

    file_url = f"{settings.S3_SPACES_PUBLIC_BASE_URL}/{object_key}"

    return object_key, file_url
