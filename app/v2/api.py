from fastapi import APIRouter

from app.v2.endpoints.equation_extraction.equation_extract2 import equation_extract
from app.v2.endpoints.plot_digitizer.get_plot_data import plot_digitization_router
from app.v2.endpoints.table_extraction.routers import api_csv_embedding_router

api_router_v2 = APIRouter(prefix="/v2")

api_router_v2.include_router(api_csv_embedding_router)
api_router_v2.include_router(equation_extract)
api_router_v2.include_router(plot_digitization_router)
