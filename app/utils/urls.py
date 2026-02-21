import urllib.parse

from app.logging import logger


def sanitize_url(link):
    parsed_link = urllib.parse.urlparse(link)
    if not parsed_link.scheme or not parsed_link.netloc:
        logger.error(f"Invalid link: {link}")
        return None

    sanitized_link = urllib.parse.urlunparse(parsed_link)
    return sanitized_link
