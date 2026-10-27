"""Basic latency and error monitoring: one log line per API request with status and duration,
and a Server-Timing header so the browser's network tab shows the same number."""
import logging
import time

logger = logging.getLogger("players.requests")
SLOW_MS = 1000


class RequestTimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.perf_counter()
        response = self.get_response(request)
        ms = (time.perf_counter() - start) * 1000
        if request.path.startswith("/api/"):
            response["Server-Timing"] = f"app;dur={ms:.0f}"
            level = logging.WARNING if response.status_code >= 500 or ms > SLOW_MS else logging.INFO
            logger.log(level, "%s %s %s %.0fms", request.method, request.path, response.status_code, ms)
        return response
