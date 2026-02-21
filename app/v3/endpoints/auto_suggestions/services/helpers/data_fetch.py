from sqlalchemy import select

from app.configs import settings
from app.core.database.crud import (
    ResultType,
    select_with_retry,
)
from app.core.database.models import (
    FigureDetails,
    TableDetails,
)
from app.v3.endpoints.auto_suggestions.constants import (
    AutoConnectionSkipWords,
    DatabaseFigureTypes,
)
from app.v3.endpoints.auto_suggestions.schemas import (
    AutoFigureConnectionAnnotation,
)


def get_sub_figure_all_data(
    project_id: str, flag_id: str
) -> list[AutoFigureConnectionAnnotation]:
    bucket_path = settings.S3_SPACES_PUBLIC_BASE_URL

    # Query FigureDetails
    figure_query = select(FigureDetails).where(
        FigureDetails.project_id == project_id,
        FigureDetails.flag_id == flag_id,
    )
    figure_details = (
        select_with_retry(figure_query, result_type=ResultType.SCALAR_ALL) or []
    )

    # Query TableDetails
    table_query = select(TableDetails).where(
        TableDetails.project_id == project_id,
        TableDetails.flag_id == flag_id,
    )
    table_details = (
        select_with_retry(table_query, result_type=ResultType.SCALAR_ALL) or []
    )

    all_data = []
    # extract plot data
    parent_figure_ids = {
        f.parent_figure_id for f in figure_details if f.parent_figure_id
    }
    for data in figure_details:
        if data.figure_id in parent_figure_ids:
            continue
        id = str(data.figure_id)
        x_top, y_top, x_bottom, y_bottom = data.converted_bbox_coordinates
        x = round(x_top)
        y = round(y_top)
        width = abs(round(x_bottom) - round(x_top))
        height = abs(round(y_bottom) - round(y_top))
        pageNo = data.page_number
        imgSrc = data.bucket_path
        type = DatabaseFigureTypes.PLOT

        caption = data.caption
        caption_filter_words = ["caption"]
        if caption in caption_filter_words:
            caption = None

        summary = data.summary
        footnote = data.footnote
        footnote_filter_words = ["footnote"]
        if footnote in footnote_filter_words:
            footnote = None

        temp_data = {
            "id": id,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "pageNo": pageNo,
            "imgSrc": f"{bucket_path}{imgSrc}",
            "type": type,
            "caption": caption,
            "description": summary,
            "footnote": footnote,
        }
        all_data.append(temp_data)

    # extract table data
    for data in table_details:
        id = str(data.table_id)
        x_top, y_top, x_bottom, y_bottom = data.converted_bbox_coordinates
        x = round(x_top)
        y = round(y_top)
        width = abs(round(x_bottom) - round(x_top))
        height = abs(round(y_bottom) - round(y_top))
        pageNo = data.page_number
        imgSrc = data.bucket_path
        type = DatabaseFigureTypes.TABLE

        caption = data.caption
        caption_filter_words = AutoConnectionSkipWords.CAPTION_FILTER_WORDS
        if caption in caption_filter_words:
            caption = None

        summary = data.summary
        footnote = data.footnote
        footnote_filter_words = AutoConnectionSkipWords.FOOTNOTE_FILTER_WORDS
        if footnote in footnote_filter_words:
            footnote = None

        temp_data = {
            "id": id,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "pageNo": pageNo,
            "imgSrc": f"{bucket_path}{imgSrc}",
            "type": type,
            "caption": caption,
            "description": summary,
            "footnote": footnote,
        }
        all_data.append(temp_data)

    return all_data
