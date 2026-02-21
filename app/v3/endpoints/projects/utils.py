import asyncio
from functools import partial

from botocore.exceptions import ClientError

from app.v3.endpoints.projects.configs import settings as project_settings
from app.v3.endpoints.projects.exceptions import StorageError
from app.v3.endpoints.projects.logging import logger


async def delete_s3_files(
    flag_ids: list[str],
    s3_client,
    bucket_name: str,
    max_retries: int = project_settings.S3_MAX_RETRIES,
) -> None:
    """
    Delete files from S3 Storage with best practices implementation.

    Args:
        flag_ids: List of flag IDs to process
        s3_client: Initialized S3 client
        bucket_name: Name of the storage bucket
        max_retries: Maximum number of retry attempts
    """

    async def delete_object_with_retry(key: str) -> None:
        """Delete a single object with retry logic"""
        for attempt in range(max_retries):
            try:
                await loop.run_in_executor(
                    None, partial(s3_client.delete_object, Bucket=bucket_name, Key=key)
                )
                logger.debug(f"Successfully deleted object: {key}")
                return
            except Exception as e:
                error_code = (
                    e.response.get("Error", {}).get("Code", "")
                    if isinstance(e, ClientError)
                    else ""
                )
                if error_code == "NoSuchKey":
                    logger.info(f"Object not found: {key}. Hence not deleting.")
                    return
                elif error_code == "429":  # Rate limit exceeded
                    wait_time = (3**attempt) * project_settings.S3_RETRY_BASE_DELAY
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
                elif error_code == "404":
                    logger.info(f"Object not found: {key}. Hence not deleting.")
                    return
                elif attempt == max_retries - 1:
                    logger.exception(
                        f"Failed to delete object {key} after {max_retries} attempts"
                    )
                    raise StorageError()
                else:
                    wait_time = (2**attempt) * project_settings.S3_RETRY_BASE_DELAY
                    await asyncio.sleep(wait_time)

    async def list_objects_with_retry(prefix: str) -> list[dict]:
        """List objects with retry logic"""
        objects = []
        continuation_token = None

        while True:
            list_params = {
                "Bucket": bucket_name,
                "Prefix": prefix,
                "MaxKeys": 1000,  # S3 maximum
            }
            if continuation_token:
                list_params["ContinuationToken"] = continuation_token

            for attempt in range(max_retries):
                try:
                    response = await loop.run_in_executor(
                        None, partial(s3_client.list_objects_v2, **list_params)
                    )
                    break
                except Exception:
                    if attempt == max_retries - 1:
                        logger.exception(
                            f"Failed to list objects for {prefix} "
                            f"after {max_retries} attempts"
                        )
                        raise StorageError()
                    wait_time = (2**attempt) * project_settings.S3_RETRY_BASE_DELAY
                    await asyncio.sleep(wait_time)

            if "Contents" in response:
                objects.extend(response["Contents"])

            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break

            # Small pause to prevent rate limiting
            await asyncio.sleep(project_settings.S3_RATE_LIMIT_PAUSE)

        return objects

    async def delete_path(path: str) -> None:
        """Delete all objects under a specific path"""
        try:
            objects = await list_objects_with_retry(path)

            if not objects:
                logger.info(f"No objects found under path: {path}. Hence not deleting.")
                return

            semaphore = asyncio.Semaphore(project_settings.S3_MAX_CONCURRENT_OPERATIONS)

            async def delete_with_semaphore(obj: dict) -> None:
                async with semaphore:
                    await delete_object_with_retry(obj["Key"])
                    await asyncio.sleep(project_settings.S3_RATE_LIMIT_PAUSE)

            delete_tasks = [delete_with_semaphore(obj) for obj in objects]

            # Execute deletions with controlled concurrency
            await asyncio.gather(*delete_tasks)

        except Exception as e:
            logger.exception(f"Error deleting path {path}: {str(e)}")
            raise StorageError()

    loop = asyncio.get_running_loop()

    for flag_id in flag_ids:
        try:
            paths = [f"documents/pages/{flag_id}/", f"documents/tables/{flag_id}/"]

            for path in paths:
                await delete_path(path)
                logger.debug(f"Successfully processed path: {path}")

        except Exception as e:
            logger.exception(f"Error processing flag_id {flag_id}: {str(e)}")
            raise StorageError()
