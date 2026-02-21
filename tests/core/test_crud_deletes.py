from unittest.mock import Mock, patch

import psycopg2
import psycopg2.errors
import pytest
from psycopg2.extensions import TransactionRollbackError
from sqlalchemy.exc import (
    DisconnectionError,
    OperationalError,
)
from sqlalchemy.sql import Select

from app.core.database.crud import (
    delete_with_retry,
    insert_single_with_retry,
    select_with_retry,
)


class TestDeleteWithRetryCornerCases:
    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_empty_filter_conditions(
        self, mock_logger, mock_get_db_session
    ):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        result = delete_with_retry(MockModel, {})

        assert result is True
        mock_session.execute.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_none_filter_value(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": None}

        result = delete_with_retry(MockModel, filter_conditions)

        assert result is True

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_zero_rows_affected(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 999}

        result = delete_with_retry(MockModel, filter_conditions)

        assert result is True

    @patch("app.core.database.crud.get_db_session")
    def test_delete_with_model_without_tablename(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a class without __tablename__
        class InvalidModel:
            id = Mock()

        filter_conditions = {"id": 1}

        # This should fail during the delete operation, not during model access
        result = delete_with_retry(InvalidModel, filter_conditions)

        # The function should return False due to the error
        assert result is False


class TestAllExceptionTypes:
    @patch("app.core.database.crud.get_db_session")
    def test_select_with_admin_shutdown_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = psycopg2.errors.AdminShutdown(
            "admin shutdown"
        )
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_connection_exception(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = psycopg2.errors.ConnectionException(
            "connection error"
        )
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_connection_failure(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = psycopg2.errors.ConnectionFailure(
            "connection failure"
        )
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_interface_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = psycopg2.InterfaceError("interface error")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None

    @patch("app.core.database.crud.get_db_session")
    def test_delete_with_disconnection_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = DisconnectionError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_model = Mock()
        mock_model.__tablename__ = "test_table"
        mock_model.id = Mock()

        result = delete_with_retry(mock_model, {"id": 1}, max_retries=1)

        assert result is False


class TestRetryExhaustionScenarios:
    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_select_retry_exhaustion_logging(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = OperationalError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=2)

        assert result is None
        # The logger should be called, but we need to check the call more flexibly
        mock_logger.exception.assert_called()
        call_args = mock_logger.exception.call_args[0][0]
        assert "Database query failed after 2 retries" in call_args

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_retry_exhaustion_logging(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = OperationalError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        result = delete_with_retry(MockModel, {"id": 1}, max_retries=3)

        assert result is False
        # Check that logger was called with appropriate message
        mock_logger.exception.assert_called()
        call_args = mock_logger.exception.call_args[0][0]
        assert "Database delete failed after 3 retries" in call_args

    @patch("app.core.database.crud.get_db_session")
    def test_insert_retry_with_max_attempts_reached(self, mock_get_db_session):
        mock_session = Mock()
        # Simulate failure on all attempts
        mock_session.add.side_effect = [
            OperationalError("test", "test", "test"),
            OperationalError("test", "test", "test"),
            OperationalError("test", "test", "test"),
        ]
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        with pytest.raises(OperationalError):
            insert_single_with_retry(mock_item, max_retries=2)


class TestDeleteWithRetry:
    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_success(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 2
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        result = delete_with_retry(MockModel, filter_conditions)

        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_logger.info.assert_called_once_with(
            "Successfully deleted 2 rows from test_table"
        )

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_multiple_conditions(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer, String
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)
            name = Column(String)

        filter_conditions = {"id": 1, "name": "test"}

        result = delete_with_retry(MockModel, filter_conditions)

        assert result is True
        mock_session.execute.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_nonexistent_column(self, mock_logger, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1, "nonexistent_column": "value"}

        result = delete_with_retry(MockModel, filter_conditions)

        assert result is True

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.configure_retry")
    def test_delete_with_custom_retry_params(
        self, mock_configure_retry, mock_get_db_session
    ):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_retry_decorator = Mock()
        mock_configure_retry.return_value = mock_retry_decorator

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        delete_with_retry(
            MockModel, filter_conditions, max_retries=5, min_wait=2, max_wait=20
        )

        mock_configure_retry.assert_called_once_with(
            max_retries=5, min_wait=2, max_wait=20
        )

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_database_error_returns_false(
        self, mock_logger, mock_get_db_session
    ):
        mock_session = Mock()
        mock_session.execute.side_effect = OperationalError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        result = delete_with_retry(MockModel, filter_conditions, max_retries=1)

        assert result is False

        mock_logger.exception.assert_called()
        call_args = mock_logger.exception.call_args[0][0]
        assert "Database delete failed after 1 retries" in call_args

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_delete_with_unexpected_error_returns_false(
        self, mock_logger, mock_get_db_session
    ):
        mock_session = Mock()
        mock_session.execute.side_effect = ValueError("unexpected error")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        result = delete_with_retry(MockModel, filter_conditions, max_retries=1)

        assert result is False
        # Check that logger was called with appropriate message
        mock_logger.exception.assert_called()
        call_args = mock_logger.exception.call_args[0][0]
        assert "Unexpected error during database delete" in call_args

    @patch("app.core.database.crud.get_db_session")
    def test_delete_with_psycopg2_errors(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = psycopg2.OperationalError("connection error")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        result = delete_with_retry(MockModel, filter_conditions, max_retries=1)

        assert result is False

    @patch("app.core.database.crud.get_db_session")
    def test_delete_with_transaction_rollback_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.execute.side_effect = TransactionRollbackError("rollback")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        # Create a proper SQLAlchemy model mock
        from sqlalchemy import Column, Integer
        from sqlalchemy.orm import declarative_base

        Base = declarative_base()

        class MockModel(Base):
            __tablename__ = "test_table"
            id = Column(Integer, primary_key=True)

        filter_conditions = {"id": 1}

        result = delete_with_retry(MockModel, filter_conditions, max_retries=1)

        assert result is False
