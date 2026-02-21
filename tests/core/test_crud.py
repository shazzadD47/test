from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import (
    OperationalError,
    SQLAlchemyError,
)
from sqlalchemy.sql import Select

from app.core.database.crud import (
    ResultType,
    configure_retry,
    insert_batch_with_retry,
    insert_single_with_retry,
    select_with_retry,
)


class TestResultType:
    def test_result_type_enum_values(self):
        assert ResultType.SCALAR.value == "scalar"
        assert ResultType.SCALAR_ONE.value == "scalar_one"
        assert ResultType.SCALAR_ALL.value == "scalar_all"
        assert ResultType.ROW_ONE.value == "row_one"
        assert ResultType.ROW_ALL.value == "row_all"


class TestConfigureRetry:
    def test_configure_retry_with_default_params(self):
        retry_decorator = configure_retry()
        assert retry_decorator is not None

    def test_configure_retry_with_custom_params(self):
        retry_decorator = configure_retry(max_retries=5, min_wait=2, max_wait=20)
        assert retry_decorator is not None

    @patch("app.core.database.crud.retry")
    def test_configure_retry_parameters(self, mock_retry):
        configure_retry(max_retries=5, min_wait=2, max_wait=20)
        mock_retry.assert_called_once()


class TestInsertSingleWithRetry:
    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_success(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        insert_single_with_retry(mock_item)

        mock_session.add.assert_called_once_with(mock_item)
        mock_session.commit.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.configure_retry")
    def test_insert_single_with_custom_retries(
        self, mock_configure_retry, mock_get_db_session
    ):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        mock_item = Mock()

        mock_retry_decorator = Mock()
        mock_configure_retry.return_value = mock_retry_decorator

        insert_single_with_retry(mock_item, max_retries=5)

        mock_configure_retry.assert_called_once_with(max_retries=5)

    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_database_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.add.side_effect = OperationalError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        with pytest.raises(OperationalError):
            insert_single_with_retry(mock_item, max_retries=1)


class TestInsertBatchWithRetry:
    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_success(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_items = [Mock(), Mock(), Mock()]

        insert_batch_with_retry(mock_items)

        mock_session.bulk_save_objects.assert_called_once_with(mock_items)
        mock_session.commit.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.configure_retry")
    def test_insert_batch_with_custom_retries(
        self, mock_configure_retry, mock_get_db_session
    ):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        mock_items = [Mock(), Mock()]

        mock_retry_decorator = Mock()
        mock_configure_retry.return_value = mock_retry_decorator

        insert_batch_with_retry(mock_items, max_retries=3)

        mock_configure_retry.assert_called_once_with(max_retries=3)

    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_database_error(self, mock_get_db_session):
        mock_session = Mock()
        mock_session.bulk_save_objects.side_effect = SQLAlchemyError("test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_items = [Mock()]

        with pytest.raises(SQLAlchemyError):
            insert_batch_with_retry(mock_items, max_retries=1)


class TestSelectWithRetry:
    @patch("app.core.database.crud.get_db_session")
    def test_select_with_scalar_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = "test_value"
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.SCALAR)

        assert result == "test_value"
        mock_result.scalar.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_scalar_one_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.first.return_value = "first_value"
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.SCALAR_ONE)

        assert result == "first_value"
        mock_result.scalars.assert_called_once()
        mock_scalars.first.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_scalar_all_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = ["value1", "value2"]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.SCALAR_ALL)

        assert result == ["value1", "value2"]
        mock_result.scalars.assert_called_once()
        mock_scalars.all.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_row_one_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.first.return_value = {"col1": "value1"}
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.ROW_ONE)

        assert result == {"col1": "value1"}
        mock_result.first.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_row_all_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.all.return_value = [{"col1": "value1"}, {"col1": "value2"}]
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.ROW_ALL)

        assert result == [{"col1": "value1"}, {"col1": "value2"}]
        mock_result.all.assert_called_once()

    def test_select_with_invalid_query_type(self):
        invalid_query = "SELECT * FROM table"

        with pytest.raises(ValueError, match="Query must be a Select object"):
            select_with_retry(invalid_query)

    def test_select_with_invalid_result_type(self):
        mock_query = Mock(spec=Select)

        # Test with invalid result type - should raise ValueError
        with pytest.raises(ValueError, match="Invalid result type"):
            select_with_retry(mock_query, "invalid_result_type")

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.configure_retry")
    def test_select_with_custom_retry_params(
        self, mock_configure_retry, mock_get_db_session
    ):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = "test"
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_retry_decorator = Mock(return_value=lambda: "test")
        mock_configure_retry.return_value = mock_retry_decorator

        mock_query = Mock(spec=Select)

        select_with_retry(
            mock_query, ResultType.SCALAR, max_retries=5, min_wait=2, max_wait=15
        )

        mock_configure_retry.assert_called_once_with(
            max_retries=5, min_wait=2, max_wait=15
        )

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_select_with_database_error_returns_none(
        self, mock_logger, mock_get_db_session
    ):
        mock_session = Mock()
        mock_session.execute.side_effect = OperationalError("test", "test", "test")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None
        mock_logger.exception.assert_called()

    @patch("app.core.database.crud.get_db_session")
    @patch("app.core.database.crud.logger")
    def test_select_with_unexpected_error_returns_none(
        self, mock_logger, mock_get_db_session
    ):
        mock_session = Mock()
        mock_session.execute.side_effect = ValueError("unexpected error")
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, max_retries=1)

        assert result is None
        mock_logger.exception.assert_called()


class TestInsertSingleWithRetryCornerCases:
    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_with_none_item(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        insert_single_with_retry(None)

        mock_session.add.assert_called_once_with(None)
        mock_session.commit.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_with_zero_retries(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        insert_single_with_retry(mock_item, max_retries=0)

        mock_session.add.assert_called_once_with(mock_item)
        mock_session.commit.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_with_negative_retries(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        insert_single_with_retry(mock_item, max_retries=-1)

        mock_session.add.assert_called_once_with(mock_item)

    @patch("app.core.database.crud.get_db_session")
    def test_insert_single_with_very_large_retries(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        insert_single_with_retry(mock_item, max_retries=1000)

        mock_session.add.assert_called_once_with(mock_item)


class TestInsertBatchWithRetryCornerCases:
    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_with_empty_list(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        insert_batch_with_retry([])

        mock_session.bulk_save_objects.assert_called_once_with([])
        mock_session.commit.assert_called_once()

    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_with_none_list(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        insert_batch_with_retry(None)

        mock_session.bulk_save_objects.assert_called_once_with(None)

    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_with_single_item(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_item = Mock()

        insert_batch_with_retry([mock_item])

        mock_session.bulk_save_objects.assert_called_once_with([mock_item])

    @patch("app.core.database.crud.get_db_session")
    def test_insert_batch_with_none_items_in_list(self, mock_get_db_session):
        mock_session = Mock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        items = [None, None, Mock()]

        insert_batch_with_retry(items)

        mock_session.bulk_save_objects.assert_called_once_with(items)


class TestSelectWithRetryCornerCases:
    @patch("app.core.database.crud.get_db_session")
    def test_select_with_none_query_result(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.SCALAR)

        assert result is None

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_empty_result_list(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_scalars = Mock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, ResultType.SCALAR_ALL)

        assert result == []

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_boundary_retry_values(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalars().all.return_value = ["test"]
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        # Test with zero retries - should still work
        result = select_with_retry(mock_query, max_retries=0)
        assert result == ["test"]

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_zero_wait_times(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalars().all.return_value = ["test"]
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        result = select_with_retry(mock_query, min_wait=0, max_wait=0)

        assert result == ["test"]

    @patch("app.core.database.crud.get_db_session")
    def test_select_with_invalid_wait_times(self, mock_get_db_session):
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalars().all.return_value = ["test"]
        mock_session.execute.return_value = mock_result
        mock_get_db_session.return_value.__enter__.return_value = mock_session

        mock_query = Mock(spec=Select)

        # min_wait > max_wait should still work (tenacity handles this)
        result = select_with_retry(mock_query, min_wait=10, max_wait=1)

        assert result == ["test"]
