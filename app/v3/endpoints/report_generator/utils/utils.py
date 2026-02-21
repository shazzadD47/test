def is_valid_report_request(request: dict | str) -> bool:
    """Validate report generation request."""
    if isinstance(request, str):
        return False

    required_fields = ["query", "project_id"]

    for field in required_fields:
        if field not in request or not request[field]:
            return False

    generation_type = request.get("generation_type", "ai_assistant")
    if generation_type not in ["ai_assistant", "ai_edit", "ai_insights"]:
        return False

    return True
