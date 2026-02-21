from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def test_app():
    """Create a test FastAPI app for testing endpoints"""
    app = FastAPI()
    return app


@pytest.fixture
def client_with_projects_router(test_app):
    """Create a test client with projects router"""
    from app.v3.endpoints.projects.routers import router

    test_app.include_router(router)
    return TestClient(test_app)


@pytest.fixture
def mock_database_operations():
    """Mock all database operations"""
    with patch(
        "app.v3.endpoints.projects.services.select_with_retry"
    ) as mock_select, patch(
        "app.v3.endpoints.projects.services.delete_with_retry"
    ) as mock_delete:
        yield {"select_with_retry": mock_select, "delete_with_retry": mock_delete}


@pytest.fixture
def mock_storage_operations():
    """Mock all storage operations"""
    with patch(
        "app.v3.endpoints.projects.services.delete_s3_files"
    ) as mock_s3_delete, patch(
        "app.v3.endpoints.projects.services.VectorStore"
    ) as mock_vector_store:
        yield {"delete_s3_files": mock_s3_delete, "vector_store": mock_vector_store}


@pytest.fixture
def mock_all_external_deps():
    """Mock all external dependencies for projects services"""
    with patch(
        "app.v3.endpoints.projects.services.select_with_retry"
    ) as mock_select, patch(
        "app.v3.endpoints.projects.services.delete_with_retry"
    ) as mock_delete, patch(
        "app.v3.endpoints.projects.services.delete_s3_files"
    ) as mock_s3_delete, patch(
        "app.v3.endpoints.projects.services.VectorStore"
    ) as mock_vector_store, patch(
        "app.v3.endpoints.projects.services.logger"
    ) as mock_logger, patch(
        "app.v3.endpoints.projects.services.s3_client"
    ) as mock_s3_client, patch(
        "app.v3.endpoints.projects.services.settings"
    ) as mock_settings, patch(
        "app.v3.endpoints.projects.services.asyncio.gather"
    ) as mock_gather, patch(
        "app.v3.endpoints.projects.services.asyncio.to_thread"
    ) as mock_to_thread:

        # Setup default mock behavior
        mock_select.return_value = []
        mock_delete.return_value = True
        # Make delete_s3_files itself an AsyncMock
        mock_s3_delete.return_value = None
        mock_s3_delete.side_effect = None
        mock_vector_store.delete_by_project_id = Mock()
        mock_vector_store.delete_by_flag_id = Mock()
        mock_settings.S3_SPACES_BUCKET = "test-bucket"

        async def mock_gather_impl(*awaitables):
            """Mock gather that executes all awaitables and handles exceptions"""
            results = []
            for awaitable in awaitables:
                try:
                    if hasattr(awaitable, "__await__"):
                        result = await awaitable
                    elif callable(awaitable):
                        result = awaitable()
                    else:
                        result = awaitable
                    results.append(result)
                except Exception as e:
                    # Re-raise the original exception instead of wrapping it
                    raise e
            return results

        async def mock_to_thread_impl(func, *args, **kwargs):
            """Mock to_thread that executes function and preserves exceptions"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Re-raise the original exception
                raise e

        mock_gather.side_effect = mock_gather_impl
        mock_to_thread.side_effect = mock_to_thread_impl

        yield {
            "select_with_retry": mock_select,
            "delete_with_retry": mock_delete,
            "delete_s3_files": mock_s3_delete,
            "vector_store": mock_vector_store,
            "logger": mock_logger,
            "s3_client": mock_s3_client,
            "settings": mock_settings,
            "gather": mock_gather,
            "to_thread": mock_to_thread,
        }


@pytest.fixture
def mock_database_models():
    """Mock all database models"""
    with patch("app.v3.endpoints.projects.services.FileDetails") as mock_file, patch(
        "app.v3.endpoints.projects.services.FigureDetails"
    ) as mock_figure, patch(
        "app.v3.endpoints.projects.services.PageDetails"
    ) as mock_page, patch(
        "app.v3.endpoints.projects.services.TableDetails"
    ) as mock_table:

        yield {
            "FileDetails": mock_file,
            "FigureDetails": mock_figure,
            "PageDetails": mock_page,
            "TableDetails": mock_table,
        }


@pytest.fixture
def mock_dependencies_validation():
    """Mock dependency validation functions"""
    with patch("app.core.dependencies.validate_flag_id") as mock_validate_flag:
        mock_validate_flag.return_value = "valid-flag-id"
        yield {"validate_flag_id": mock_validate_flag}


@pytest.fixture
def mock_asyncio_operations():
    """Mock asyncio operations to run synchronously in tests"""
    with patch(
        "app.v3.endpoints.projects.services.asyncio.gather"
    ) as mock_gather, patch(
        "app.v3.endpoints.projects.services.asyncio.to_thread"
    ) as mock_to_thread:

        async def mock_gather_impl(*awaitables):
            """Mock gather that executes all awaitables and handles exceptions"""
            results = []
            for awaitable in awaitables:
                try:
                    if hasattr(awaitable, "__await__"):
                        result = await awaitable
                    elif callable(awaitable):
                        result = awaitable()
                    else:
                        result = awaitable
                    results.append(result)
                except Exception as e:
                    # Re-raise the original exception instead of wrapping it
                    raise e
            return results

        async def mock_to_thread_impl(func, *args, **kwargs):
            """Mock to_thread that executes function and preserves exceptions"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Re-raise the original exception
                raise e

        mock_gather.side_effect = mock_gather_impl
        mock_to_thread.side_effect = mock_to_thread_impl

        yield {"gather": mock_gather, "to_thread": mock_to_thread}
