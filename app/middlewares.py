import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.logging import logger


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        start_time = time.time()
        response = await call_next(request)
        processing_time = (time.time() - start_time) * 1000  # in milliseconds

        client_host = request.client.host
        client_port = request.url.port
        request_method = request.method
        request_url = request.url

        log_data = {
            "client_ip": client_host,
            "client_port": client_port,
            "method": request_method,
            "url": str(request_url),
            "time": processing_time,
            "status_code": response.status_code,
        }

        logger.info(log_data, extra=log_data)

        return response
