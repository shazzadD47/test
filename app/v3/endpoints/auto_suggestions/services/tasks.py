from celery import shared_task

from app.core.event_bus import BackendEventEnumType, send_to_backend
from app.v3.endpoints import Status
from app.v3.endpoints.auto_suggestions.configs import settings
from app.v3.endpoints.auto_suggestions.constants import (
    AISelectionInputTypes,
    DatabaseFigureTypes,
)
from app.v3.endpoints.auto_suggestions.logging import celery_logger
from app.v3.endpoints.auto_suggestions.prompts import (
    FIGURE_AUTO_SUGGESTION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
)
from app.v3.endpoints.auto_suggestions.schemas import (
    AutoFigureConnectionAnnotation,
    AutoFigureSuggestion,
    FigureSearchCriteria,
    ImageSelection,
    ImageSuggested,
)
from app.v3.endpoints.auto_suggestions.services.ai_selection import (
    AIAutoSuggestionGenerator,
)
from app.v3.endpoints.auto_suggestions.services.helpers.data_fetch import (
    get_sub_figure_all_data,
)
from app.v3.endpoints.auto_suggestions.services.keyword_gen import (
    KeyWordGenerator,
)
from app.v3.endpoints.auto_suggestions.services.score_gen import (
    ConfidenceScoreGenerator,
)


@shared_task(
    name="auto_figure_suggestion_task",
    bind=True,
    max_retries=0,
    default_retry_delay=10,
)
def auto_figure_suggestion_task(
    self,
    table_id: str,
    file_id: str,
    flag_id: str,
    table_structure: dict,
    project_id: str,
    metadata: dict,
) -> dict:
    try:
        inputs = table_structure.get("inputs")
        if not inputs:
            error_message = (
                "No inputs found in table structure, skipping auto figure suggestion"
            )
            celery_logger.exception(error_message)
            output = {
                "message": error_message,
                "data": None,
                "status": Status.FAILED.value,
                "metadata": metadata,
            }
            send_to_backend(BackendEventEnumType.AUTO_INPUT_SUGGESTION, output)
            return output

        celery_logger.info(
            f"Auto figure suggestion task started for table_id: {table_id}"
        )
        all_data: list[AutoFigureConnectionAnnotation] = get_sub_figure_all_data(
            project_id=project_id, flag_id=flag_id
        )
        celery_logger.info("All data extracted for auto figure suggestion")

        output_suggestions: list[AutoFigureSuggestion] = []
        for user_data in inputs:
            # Filter input type
            user_input_type: str = user_data["typeLabel"].lower()
            user_input_original_type: str = user_data["typeLabel"]
            if user_input_type not in settings.SELECTION_TYPE:
                continue

            user_input_type = (
                DatabaseFigureTypes.PLOT
                if user_input_type == AISelectionInputTypes.CHART
                else user_input_type
            )
            input_type = (
                DatabaseFigureTypes.IMAGE
                if user_input_type == AISelectionInputTypes.IMAGE
                else [user_input_type]
            )
            if user_input_original_type.lower() == AISelectionInputTypes.CHART:
                total_count = len(all_data)

                other_count_before = sum(
                    1
                    for fig in all_data
                    if fig.get("chartType")
                    and fig["chartType"].lower() in ("other", "others")
                )
                filtered_all_data = [
                    fig
                    for fig in all_data
                    if fig.get("chartType")
                    and fig["chartType"].lower() not in ("other", "others")
                ]
                filtered_count = len(filtered_all_data)
                other_count_after = sum(
                    1
                    for fig in filtered_all_data
                    if fig.get("chartType")
                    and fig["chartType"].lower() in ("other", "others")
                )
                celery_logger.info(
                    f"[AutoFigure] InputType={user_input_original_type} | "
                    f"Total={total_count}, OtherBefore={other_count_before}, "
                    f"AfterFilter={filtered_count}, OtherAfter={other_count_after}"
                )

                # Optional: log IDs of removed charts (first 5 only)
                removed_ids = [
                    fig["id"]
                    for fig in all_data
                    if fig.get("chartType")
                    and fig["chartType"].lower() in ("other", "others")
                ]
                celery_logger.debug(
                    f"[AutoFigure] Removed 'other' IDs (up to 5): {removed_ids[:5]}"
                )

            else:
                filtered_all_data = all_data
                celery_logger.info(
                    f"[AutoFigure] InputType={user_input_original_type} | "
                    f"Total={len(all_data)}, Others allowed, No filtering applied"
                )

            # Get input data

            user_requirement: str = user_data["description"]
            keyword_extraction_prompt: str = KEYWORD_EXTRACTION_PROMPT
            input_id: str = user_data["id"]

            keyword_generator = KeyWordGenerator(
                user_requirement=user_requirement,
                keyword_extraction_prompt=keyword_extraction_prompt,
                output_schema=FigureSearchCriteria,
            )
            celery_logger.info("Keyword generation completed")

            ai_suggestion_generator = AIAutoSuggestionGenerator(
                all_data=filtered_all_data,
                keywords_extraction_info=keyword_generator.keywords_extraction_info,
                user_requirement=user_requirement,
                selection_type=input_type,
                figure_auto_suggestion_prompt=FIGURE_AUTO_SUGGESTION_PROMPT,
            )
            llm_suggestion_output: list[ImageSelection] = (
                ai_suggestion_generator.get_llm_suggestion_output()
            )
            celery_logger.info(
                f"LLM auto figure suggestion generated: {llm_suggestion_output}"
            )

            confidence_score_generator = ConfidenceScoreGenerator(
                keywords_extraction_info=keyword_generator.keywords_extraction_info,
                auto_suggestion_llm_output=llm_suggestion_output,
            )
            suggested_image_data: list[ImageSuggested] = (
                confidence_score_generator.get_confidence_score_output()
            )
            celery_logger.info("Auto figure suggestion score generation completed")

            # Generate output suggestions
            for suggested_image in suggested_image_data:
                temp_suggestion_data = {
                    "id": suggested_image["image_number"],
                    "type": user_input_original_type,
                    "keyword": suggested_image["keyword_matches"],
                    "confidence_score": round(suggested_image["confidence_score"], 3),
                    "input_id": input_id,
                }
                output_suggestions.append(temp_suggestion_data)

        output_suggestion_data = {
            "table_id": table_id,
            "file_id": file_id,
            "project_id": project_id,
            "suggestions": output_suggestions,
        }

        output = {
            "message": "Auto Figure Suggestion Generated",
            "data": output_suggestion_data,
            "status": Status.SUCCESS.value,
            "metadata": metadata,
        }
        celery_logger.info(f"Auto Figure Suggestion completed for flag_id: {flag_id}")
        send_to_backend(BackendEventEnumType.AUTO_INPUT_SUGGESTION, output)
        return output

    except Exception as e:
        message = f"Error when tried to generate auto figure suggestion: {e}"
        celery_logger.exception(message)
        output = {
            "message": message,
            "data": None,
            "status": Status.FAILED.value,
            "metadata": metadata,
        }
        send_to_backend(BackendEventEnumType.AUTO_INPUT_SUGGESTION, output)
        return output
