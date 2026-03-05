from fastapi import APIRouter

from app.v3.endpoints.agent_chat.routers import router as agent_chat_router
from app.v3.endpoints.auto_suggestions.routers import (
    router as auto_figure_suggestion_router,
)
from app.v3.endpoints.autofill.routers import router as autofill_router
from app.v3.endpoints.case_study_rag.routers import router as case_study_rag
from app.v3.endpoints.column_standardization.routers import (
    router as column_standardization_router,
)
from app.v3.endpoints.covariate_extraction.routers import router as covariate_router
from app.v3.endpoints.delete.delete import router as delete_router
from app.v3.endpoints.dosing_table.routers import router as dosing_table_router
from app.v3.endpoints.dynamic_dosing.routers import router as dynamic_dosing_router
from app.v3.endpoints.extraction_templates.routers import (
    router as extraction_templates_router,
)
from app.v3.endpoints.general_extraction.routers import (
    router as general_extraction_router,
)
from app.v3.endpoints.get_paper_labels.routers import router as get_paper_labels_router
from app.v3.endpoints.get_title_summery.routers import router as title_summary_router
from app.v3.endpoints.iterative_autofill.routers import (
    router as iterative_autofill_router,
)
from app.v3.endpoints.merging.routers import router as merging_router
from app.v3.endpoints.plot_digitizer.routers import router as plot_digitizer_router
from app.v3.endpoints.projects.routers import router as project_router
from app.v3.endpoints.rag_chat.routers import router as improved_rag
from app.v3.endpoints.report_generator.routers import router as report_generator_router
from app.v3.endpoints.tag_extraction.routers import router as tag_extraction_router
from app.v3.endpoints.tasks.routers import router as tasks_router
from app.v3.endpoints.text2graph.graph_prompt_test import graph_prompt_router
from app.v3.endpoints.text2graph.text_to_graph_3 import text_to_graph_router_3
from app.v3.endpoints.unit_standardization.routers import (
    router as unit_standardization_router,
)

api_router_v3 = APIRouter(prefix="/v3")

api_router_v3.include_router(autofill_router)
api_router_v3.include_router(dosing_table_router)
api_router_v3.include_router(dynamic_dosing_router)
api_router_v3.include_router(title_summary_router)
api_router_v3.include_router(project_router)
api_router_v3.include_router(text_to_graph_router_3)
api_router_v3.include_router(delete_router)
api_router_v3.include_router(graph_prompt_router)
api_router_v3.include_router(improved_rag)
api_router_v3.include_router(agent_chat_router)
api_router_v3.include_router(
    report_generator_router
)  # Add the new report generator router
api_router_v3.include_router(case_study_rag)
api_router_v3.include_router(plot_digitizer_router)
api_router_v3.include_router(covariate_router)
api_router_v3.include_router(get_paper_labels_router)
api_router_v3.include_router(iterative_autofill_router)
api_router_v3.include_router(tasks_router)
api_router_v3.include_router(general_extraction_router)
api_router_v3.include_router(auto_figure_suggestion_router)
api_router_v3.include_router(unit_standardization_router)
api_router_v3.include_router(column_standardization_router)
api_router_v3.include_router(tag_extraction_router)
api_router_v3.include_router(merging_router)
api_router_v3.include_router(extraction_templates_router)
