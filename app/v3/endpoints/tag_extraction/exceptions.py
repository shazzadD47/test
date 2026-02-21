from fastapi import HTTPException, status


class TagExtractionTaskFailedException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start tag extraction task",
        )


class TaskSubmissionException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
