import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status

from app.v3.endpoints.projects.exceptions import (
    DatabaseError,
    StorageError,
    UnexpectedError,
)
from app.v3.endpoints.projects.services import (
    delete_flag_and_storage_service,
    delete_project_and_storage_service,
)


class TestDeleteProjectAndStorageEndpoint:
    """Tests for DELETE /projects/{project_id}/ endpoint"""

    @patch("app.v3.endpoints.projects.routers.delete_project_and_storage_service")
    def test_delete_project_success(self, mock_service, client_with_projects_router):
        """Test successful project deletion"""
        mock_service.return_value = AsyncMock()

        response = client_with_projects_router.delete("/projects/test-project-123/")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert response.content == b""
        mock_service.assert_called_once_with("test-project-123")

    @patch("app.v3.endpoints.projects.routers.delete_project_and_storage_service")
    def test_delete_project_database_error(
        self, mock_service, client_with_projects_router
    ):
        """Test project deletion with database error"""
        mock_service.side_effect = DatabaseError()

        response = client_with_projects_router.delete("/projects/test-project-123/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_service.assert_called_once_with("test-project-123")

    @patch("app.v3.endpoints.projects.routers.delete_project_and_storage_service")
    def test_delete_project_storage_error(
        self, mock_service, client_with_projects_router
    ):
        """Test project deletion with storage error"""
        mock_service.side_effect = StorageError()

        response = client_with_projects_router.delete("/projects/test-project-123/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_service.assert_called_once_with("test-project-123")

    @patch("app.v3.endpoints.projects.routers.delete_project_and_storage_service")
    def test_delete_project_unexpected_error(
        self, mock_service, client_with_projects_router
    ):
        """Test project deletion with unexpected error"""
        mock_service.side_effect = UnexpectedError()

        response = client_with_projects_router.delete("/projects/test-project-123/")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_service.assert_called_once_with("test-project-123")

    def test_delete_project_invalid_method(self, client_with_projects_router):
        """Test using wrong HTTP method"""
        response = client_with_projects_router.get("/projects/test-project-123/")

        assert response.status_code != status.HTTP_204_NO_CONTENT

    @patch("app.v3.endpoints.projects.routers.delete_project_and_storage_service")
    def test_delete_project_with_special_characters(
        self, mock_service, client_with_projects_router
    ):
        """Test project deletion with special characters in project_id"""
        mock_service.return_value = AsyncMock()

        project_ids = [
            "project-with-dashes",
            "project_with_underscores",
            "project123",
            "PROJECT-UPPER-CASE",
        ]

        for project_id in project_ids:
            response = client_with_projects_router.delete(f"/projects/{project_id}/")
            assert response.status_code == status.HTTP_204_NO_CONTENT
            mock_service.assert_called_with(project_id)


class TestDeleteFlagAndStorageEndpoint:
    """Tests for DELETE /projects/{project_id}/file/{flag_id} endpoint"""

    @patch("app.v3.endpoints.projects.routers.delete_flag_and_storage_service")
    def test_delete_flag_success(self, mock_service, test_app):
        """Test successful flag deletion"""
        from fastapi.testclient import TestClient

        from app.v3.endpoints.projects.routers import router

        mock_service.return_value = AsyncMock()
        flag_id = str(uuid.uuid4())  # Generate a valid UUID v4

        test_app.include_router(router)
        client = TestClient(test_app)

        try:
            response = client.delete(f"/projects/test-project-123/file/{flag_id}")

            assert response.status_code == status.HTTP_204_NO_CONTENT
            assert response.content == b""
            mock_service.assert_called_once_with("test-project-123", flag_id)
        finally:
            # Clean up the override
            test_app.dependency_overrides.clear()

    @patch("app.v3.endpoints.projects.routers.delete_flag_and_storage_service")
    def test_delete_flag_database_error(self, mock_service, test_app):
        """Test flag deletion with database error"""
        from fastapi.testclient import TestClient

        from app.v3.endpoints.projects.routers import router

        mock_service.side_effect = DatabaseError()
        flag_id = str(uuid.uuid4())

        test_app.include_router(router)
        client = TestClient(test_app)

        try:
            response = client.delete(f"/projects/test-project-123/file/{flag_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            mock_service.assert_called_once_with("test-project-123", flag_id)
        finally:
            # Clean up the override
            test_app.dependency_overrides.clear()

    @patch("app.v3.endpoints.projects.routers.delete_flag_and_storage_service")
    def test_delete_flag_storage_error(self, mock_service, test_app):
        """Test flag deletion with storage error"""
        from fastapi.testclient import TestClient

        from app.v3.endpoints.projects.routers import router

        mock_service.side_effect = StorageError()
        flag_id = str(uuid.uuid4())  # Generate a valid UUID v4

        test_app.include_router(router)
        client = TestClient(test_app)

        try:
            response = client.delete(f"/projects/test-project-123/file/{flag_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            mock_service.assert_called_once_with("test-project-123", flag_id)
        finally:
            # Clean up the override
            test_app.dependency_overrides.clear()

    @patch("app.v3.endpoints.projects.routers.delete_flag_and_storage_service")
    def test_delete_flag_unexpected_error(self, mock_service, test_app):
        """Test flag deletion with unexpected error"""
        from fastapi.testclient import TestClient

        from app.v3.endpoints.projects.routers import router

        mock_service.side_effect = UnexpectedError()
        flag_id = str(uuid.uuid4())  # Generate a valid UUID v4

        test_app.include_router(router)
        client = TestClient(test_app)

        try:
            response = client.delete(f"/projects/test-project-123/file/{flag_id}")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            mock_service.assert_called_once_with("test-project-123", flag_id)
        finally:
            # Clean up the override
            test_app.dependency_overrides.clear()


class TestDeleteServicesUnit:
    """Unit tests for delete service functions"""

    @pytest.mark.asyncio
    async def test_delete_project_service_success(self, mock_all_external_deps):
        """Test delete_project_and_storage_service success scenario"""
        # Setup mocks
        mock_all_external_deps["select_with_retry"].return_value = [
            "flag1",
            "flag2",
            "flag3",
        ]
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].return_value = None

        # Execute service
        await delete_project_and_storage_service("test-project-123")

        # Verify database queries
        mock_all_external_deps["select_with_retry"].assert_called_once()

        # Verify delete operations (4 models * 1 call each)
        assert mock_all_external_deps["delete_with_retry"].call_count == 4

        # Verify S3 deletion
        mock_all_external_deps["delete_s3_files"].assert_called_once()

        # Verify vector store deletion
        mock_all_external_deps[
            "vector_store"
        ].delete_by_project_id.assert_called_once_with("test-project-123")

    @pytest.mark.asyncio
    async def test_delete_project_service_no_documents(self, mock_all_external_deps):
        """Test delete_project_and_storage_service when no documents found"""
        # Setup: no flag_ids found
        mock_all_external_deps["select_with_retry"].return_value = []

        # Execute service - should return early
        await delete_project_and_storage_service("test-project-123")

        # Verify it returns early without further operations
        mock_all_external_deps["logger"].info.assert_called_with(
            "No documents found for project_id: test-project-123"
        )

    @pytest.mark.asyncio
    async def test_delete_project_service_database_failure(
        self, mock_all_external_deps
    ):
        """Test delete_project_and_storage_service with database deletion failure"""
        # Setup mocks
        mock_all_external_deps["select_with_retry"].return_value = ["flag1", "flag2"]
        mock_all_external_deps["delete_with_retry"].return_value = (
            False  # Database deletion fails
        )

        # Execute and expect DatabaseError
        with pytest.raises(DatabaseError):
            await delete_project_and_storage_service("test-project-123")

    @pytest.mark.asyncio
    async def test_delete_project_service_storage_failure(self, mock_all_external_deps):
        """Test delete_project_and_storage_service with storage deletion failure"""
        # Setup mocks
        mock_all_external_deps["select_with_retry"].return_value = ["flag1", "flag2"]
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].side_effect = (
            StorageError()
        )  # S3 deletion fails

        # Execute and expect StorageError
        with pytest.raises(StorageError):
            await delete_project_and_storage_service("test-project-123")

    @pytest.mark.asyncio
    async def test_delete_project_service_unexpected_failure(
        self, mock_all_external_deps
    ):
        """Test delete_project_and_storage_service with unexpected failure"""
        # Setup: unexpected exception during flag_ids retrieval
        mock_all_external_deps["select_with_retry"].side_effect = Exception(
            "Unexpected database error"
        )

        # Execute and expect UnexpectedError
        with pytest.raises(UnexpectedError):
            await delete_project_and_storage_service("test-project-123")

    @pytest.mark.asyncio
    async def test_delete_flag_service_success(self, mock_all_external_deps):
        """Test delete_flag_and_storage_service success scenario"""
        # Setup mocks
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].return_value = None

        # Execute service
        await delete_flag_and_storage_service("test-project-123", "test-flag-456")

        # Verify delete operations (4 models * 1 call each)
        assert mock_all_external_deps["delete_with_retry"].call_count == 4

        # Verify S3 deletion
        mock_all_external_deps["delete_s3_files"].assert_called_once()

        # Verify vector store deletion
        mock_all_external_deps[
            "vector_store"
        ].delete_by_flag_id.assert_called_once_with("test-flag-456")

    @pytest.mark.asyncio
    async def test_delete_flag_service_database_failure(self, mock_all_external_deps):
        """Test delete_flag_and_storage_service with database deletion failure"""
        # Setup mocks
        mock_all_external_deps["delete_with_retry"].return_value = (
            False  # Database deletion fails
        )

        # Execute and expect DatabaseError
        with pytest.raises(DatabaseError):
            await delete_flag_and_storage_service("test-project-123", "test-flag-456")

    @pytest.mark.asyncio
    async def test_delete_flag_service_storage_failure(self, mock_all_external_deps):
        """Test delete_flag_and_storage_service with storage deletion failure"""
        # Setup mocks
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].side_effect = (
            StorageError()
        )  # S3 deletion fails

        # Execute and expect StorageError
        with pytest.raises(StorageError):
            await delete_flag_and_storage_service("test-project-123", "test-flag-456")

    @pytest.mark.asyncio
    async def test_delete_flag_service_unexpected_failure(self, mock_all_external_deps):
        """Test delete_flag_and_storage_service with unexpected failure"""
        # Setup: unexpected exception during delete_with_retry
        mock_all_external_deps["delete_with_retry"].side_effect = Exception(
            "Unexpected error"
        )

        # Execute and expect DatabaseError (inner exception handler converts it)
        with pytest.raises(DatabaseError):
            await delete_flag_and_storage_service("test-project-123", "test-flag-456")

    @pytest.mark.asyncio
    async def test_delete_flag_service_truly_unexpected_failure(
        self, mock_all_external_deps
    ):
        """Test delete_flag_and_storage_service with failure outside
        database operations"""
        # Setup: unexpected exception during asyncio.gather itself
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["gather"].side_effect = RuntimeError(
            "Unexpected runtime error"
        )

        # Execute and expect UnexpectedError (outer exception handler)
        with pytest.raises(UnexpectedError):
            await delete_flag_and_storage_service("test-project-123", "test-flag-456")


class TestDeleteServicesEdgeCases:
    """Edge case tests for delete services"""

    @pytest.mark.asyncio
    async def test_delete_project_with_duplicate_flag_ids(self, mock_all_external_deps):
        """Test delete project service handles duplicate flag_ids correctly"""
        # Setup: duplicate flag_ids in result
        mock_all_external_deps["select_with_retry"].return_value = [
            "flag1",
            "flag1",
            "flag2",
            "flag2",
            "flag3",
        ]
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].return_value = None

        # Execute service
        await delete_project_and_storage_service("test-project-123")

        # Verify S3 deletion called with deduplicated flag_ids
        called_flag_ids = mock_all_external_deps["delete_s3_files"].call_args[0][0]
        assert set(called_flag_ids) == {"flag1", "flag2", "flag3"}
        assert len(called_flag_ids) == 3  # Deduplicated

    @pytest.mark.asyncio
    async def test_delete_project_with_empty_project_id(self, mock_all_external_deps):
        """Test delete project service with empty project_id"""
        mock_all_external_deps["select_with_retry"].return_value = []

        # Should not raise an exception, just return early
        await delete_project_and_storage_service("")

    @pytest.mark.asyncio
    async def test_delete_flag_with_empty_ids(self, mock_all_external_deps):
        """Test delete flag service with empty IDs"""
        mock_all_external_deps["delete_with_retry"].return_value = True
        mock_all_external_deps["delete_s3_files"].return_value = None

        # Should not raise an exception
        await delete_flag_and_storage_service("", "")

        # Verify operations still attempted
        assert mock_all_external_deps["delete_with_retry"].call_count == 4

    @pytest.mark.asyncio
    async def test_delete_project_partial_database_failure(
        self, mock_all_external_deps
    ):
        """Test delete project service with partial database deletion failure"""
        # Setup: some deletions succeed, some fail
        mock_all_external_deps["select_with_retry"].return_value = ["flag1", "flag2"]
        mock_all_external_deps["delete_with_retry"].side_effect = [
            True,
            False,
            True,
            True,
        ]  # Second deletion fails

        # Execute and expect DatabaseError
        with pytest.raises(DatabaseError):
            await delete_project_and_storage_service("test-project-123")

    @pytest.mark.asyncio
    async def test_delete_flag_partial_database_failure(self, mock_all_external_deps):
        """Test delete flag service with partial database deletion failure"""
        # Setup: some deletions succeed, some fail
        mock_all_external_deps["delete_with_retry"].side_effect = [
            True,
            True,
            False,
            True,
        ]  # Third deletion fails

        # Execute and expect DatabaseError
        with pytest.raises(DatabaseError):
            await delete_flag_and_storage_service("test-project-123", "test-flag-456")
