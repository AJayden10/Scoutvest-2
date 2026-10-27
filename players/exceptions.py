"""One error shape for every API failure: {"detail": "...", "code": "..."} plus the HTTP status.

DRF already handles its own exceptions (404s, validation errors); this adds a `code` to them and turns
anything unexpected into a logged 500 with the same shape, instead of Django's HTML error page."""
import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger("players.api")


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        data = response.data
        if isinstance(data, dict) and "detail" in data:
            code = getattr(data["detail"], "code", None) or getattr(exc, "default_code", "error")
            response.data = {"detail": str(data["detail"]), "code": str(code)}
        else:  # field validation errors: keep the per-field messages
            response.data = {"detail": "invalid request", "code": "invalid", "errors": data}
        return response

    view = context.get("view")
    logger.exception("Unhandled API error in %s", view.__class__.__name__ if view else "unknown view")
    return Response({"detail": "Internal server error", "code": "server_error"}, status=500)
