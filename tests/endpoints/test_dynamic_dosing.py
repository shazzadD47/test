from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.v3.endpoints.dynamic_dosing.prompt import build_dynamic_dosing_instruction
from app.v3.endpoints.dynamic_dosing.routers import router
from app.v3.endpoints.dynamic_dosing.schemas import DynamicDosingRequest
from app.v3.endpoints.dynamic_dosing.tasks import (
    extract_dynamic_dosing,
    format_dynamic_dosing_output,
)


def _build_request_payload() -> dict:
    return {
        "payload": {
            "project_id": "proj_1",
            "paper_id": "paper_1",
            "image_url": ["https://example.com/figure.png"],
            "table_template": [
                {
                    "name": "GROUP",
                    "description": "Treatment group",
                    "d_type": "string",
                    "c_type": "general",
                    "literal_options": None,
                },
                {
                    "name": "ROUTE",
                    "description": "Route of administration",
                    "d_type": "string",
                    "c_type": "general",
                    "literal_options": None,
                },
                {
                    "name": "ARM_TIME",
                    "description": "Dosing times",
                    "d_type": "list[float]",
                    "c_type": "array",
                    "literal_options": None,
                },
                {
                    "name": "AMT",
                    "description": "Dose amounts",
                    "d_type": "list[float]",
                    "c_type": "array",
                    "literal_options": None,
                },
            ],
        },
        "metadata": {"request_id": "req-1"},
    }


class TestDynamicDosingSchema:
    def test_table_template_alias_and_group_root_enforced(self):
        payload = _build_request_payload()
        payload["payload"]["table_tamplate"] = payload["payload"].pop("table_template")

        # Intentionally mark a non-GROUP label as root; validator should downgrade it.
        payload["payload"]["table_tamplate"][1]["c_type"] = "root"

        request = DynamicDosingRequest.model_validate(payload)
        template = request.payload.table_template

        group_field = next(field for field in template if field.name == "GROUP")
        route_field = next(field for field in template if field.name == "ROUTE")

        assert group_field.c_type == "root"
        assert route_field.c_type == "general"

    def test_missing_group_raises_validation_error(self):
        payload = _build_request_payload()
        payload["payload"]["table_template"] = [
            field
            for field in payload["payload"]["table_template"]
            if field["name"] != "GROUP"
        ]

        with pytest.raises(ValidationError):
            DynamicDosingRequest.model_validate(payload)

    def test_legacy_flat_shape_with_table_structure_is_accepted(self):
        payload = {
            "project_id": "proj_1",
            "paper_id": "paper_1",
            "table_structure": [
                {
                    "name": "GROUP",
                    "description": "Treatment group",
                    "d_type": "string",
                    "c_type": "root",
                    "literal_options": "None",
                },
                {
                    "name": "ROUTE",
                    "description": "Route of administration",
                    "d_type": "string",
                    "c_type": "general",
                    "literal_options": "None",
                },
            ],
            "metadata": {"request_id": "legacy-req"},
        }

        request = DynamicDosingRequest.model_validate(payload)
        assert request.payload.project_id == "proj_1"
        assert request.payload.paper_id == "paper_1"
        assert len(request.payload.table_template) == 2
        assert request.payload.table_template[0].c_type == "root"
        assert request.payload.table_template[0].literal_options is None
        assert request.metadata["request_id"] == "legacy-req"


class TestDynamicDosingRouter:
    @patch(
        "app.v3.endpoints.dynamic_dosing.services."
        "extract_dynamic_dosing_task.apply_async"
    )
    def test_dynamic_dosing_starts_background_task(self, mock_apply_async):
        mock_apply_async.return_value = Mock(id="task-123")

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        payload = _build_request_payload()
        response = client.post("/meta-analysis/dosing-table/dynamic/", json=payload)

        assert response.status_code == 200
        body = response.json()

        assert (
            body["message"] == "Dynamic dosing extraction process started in Background"
        )
        assert body["ai_metadata"]["task_id"] == "task-123"

        task_kwargs = mock_apply_async.call_args.kwargs["kwargs"]
        assert task_kwargs["paper_id"] == "paper_1"
        table_template = task_kwargs["table_template"]

        group_field = next(
            field for field in table_template if field["name"] == "GROUP"
        )
        assert group_field["c_type"] == "root"
        assert all(
            field["name"] == "GROUP" or field["c_type"] != "root"
            for field in table_template
        )


class TestDynamicDosingFormatter:
    def test_format_dynamic_dosing_output_explodes_array_fields(self):
        table_template = [
            {
                "name": "GROUP",
                "description": "Treatment group",
                "d_type": "string",
                "c_type": "root",
            },
            {
                "name": "ROUTE",
                "description": "Route",
                "d_type": "string",
                "c_type": "general",
            },
            {
                "name": "ARM_TIME",
                "description": "Times",
                "d_type": "list[float]",
                "c_type": "array",
            },
            {
                "name": "AMT",
                "description": "Dose",
                "d_type": "list[float]",
                "c_type": "array",
            },
        ]

        output = format_dynamic_dosing_output(
            data=[
                {
                    "GROUP": "Semaglutide",
                    "ROUTE": "sc",
                    "ARM_TIME": "0, 4, 8",
                    "AMT": "0.25, 0.5, 1",
                }
            ],
            table_template=table_template,
        )

        assert len(output) == 3
        assert [row["GROUP"] for row in output] == [
            "Semaglutide",
            "Semaglutide",
            "Semaglutide",
        ]
        assert [row["ROUTE"] for row in output] == ["sc", "sc", "sc"]
        assert [row["ARM_TIME"] for row in output] == [0.0, 4.0, 8.0]
        assert [row["AMT"] for row in output] == [0.25, 0.5, 1.0]


class TestDynamicDosingPromptMode:
    @patch("app.v3.endpoints.dynamic_dosing.tasks.execute_general_extraction")
    def test_with_figure_uses_figure_aware_instruction(self, mock_ge):
        mock_ge.return_value = {"payload": [], "metadata": {"status": "success"}}
        table_template = [
            {
                "name": "GROUP",
                "description": "Treatment group",
                "d_type": "string",
                "c_type": "root",
            }
        ]

        extract_dynamic_dosing(
            project_id="proj_1",
            paper_id="paper_1",
            table_template=table_template,
            metadata={},
            image_url="https://example.com/figure.png",
        )

        ge_inputs = mock_ge.call_args.args[0]
        assert "Figure-aware mode" in ge_inputs["custom_instruction"]

    @patch("app.v3.endpoints.dynamic_dosing.tasks.execute_general_extraction")
    def test_no_figure_uses_document_mode_instruction(self, mock_ge):
        mock_ge.return_value = {"payload": [], "metadata": {"status": "success"}}
        table_template = [
            {
                "name": "GROUP",
                "description": "Treatment group",
                "d_type": "string",
                "c_type": "root",
            }
        ]

        extract_dynamic_dosing(
            project_id="proj_1",
            paper_id="paper_1",
            table_template=table_template,
            metadata={},
            image_url=None,
        )

        ge_inputs = mock_ge.call_args.args[0]
        assert "Document-only mode" in ge_inputs["custom_instruction"]


class TestDynamicDosingDynamicPrompt:
    def test_dynamic_prompt_includes_user_defined_fields_without_hardcoded_rules(self):
        instruction = build_dynamic_dosing_instruction(
            table_template=[
                {
                    "name": "GROUP",
                    "description": "Arm identifier",
                    "d_type": "string",
                    "c_type": "root",
                    "literal_options": None,
                },
                {
                    "name": "NEW_PK_WINDOW",
                    "description": "PK collection window",
                    "d_type": "string",
                    "c_type": "general",
                    "literal_options": None,
                },
                {
                    "name": "SPECIAL_SCORE",
                    "description": "Dynamic score from protocol",
                    "d_type": "float",
                    "c_type": "general",
                    "literal_options": None,
                },
            ],
            has_figure=False,
        )

        assert "NEW_PK_WINDOW" in instruction
        assert "SPECIAL_SCORE" in instruction
        assert "Numeric fields (SPECIAL_SCORE)" in instruction
        # AMT-specific rule should not appear when AMT is not requested.
        assert "placebo is explicitly stated" not in instruction
