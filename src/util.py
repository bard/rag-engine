from typing import TypedDict, Literal
import requests
from data_uri_parser import DataURI  # type: ignore[import-untyped]
from urllib.parse import urlparse


class FetchResult(TypedDict):
    data: str
    type: Literal["text/plain", "text/html"]


def fetch_content(uri: str) -> FetchResult:
    """
    Fetch HTML content from various sources: local file, HTTP URL, or data URL.

    Args:
        source_url: file URL, HTTP URL, or data URL containing HTML content

    Returns:
        HTML content as string

    Raises:
        ValueError: If the source type is not supported or content cannot be retrieved
    """
    parsed = urlparse(str(uri))

    if not parsed.scheme or parsed.scheme == "file":
        with open(parsed.path, "r") as file:
            data = file.read()
            type = sniff_text_format(data)
            return {"data": data, "type": type}

    if parsed.scheme in ["http", "https"]:
        response = requests.get(str(uri))
        response.raise_for_status()
        data = response.text
        type = validate_content_type(
            response.headers.get("content-type", "").lower().split(";")[0]
        )
        return {"data": data, "type": type}

    if parsed.scheme == "data":
        parsed_uri = DataURI(uri)
        assert parsed_uri.mimetype is not None
        type = validate_content_type(parsed_uri.mimetype)
        data = (
            parsed_uri.data.decode("utf-8")
            if isinstance(parsed_uri.data, bytes)
            else parsed_uri.data
        )
        return {"data": data, "type": type}

    raise ValueError(f"Unsupported source type: {parsed.scheme}")


def sniff_text_format(content: str) -> Literal["text/plain", "text/html"]:
    """
    Detect whether a string contains HTML or plain text.

    Args:
        content: The string to analyze

    Returns:
        "text/html" if HTML tags are found, "text/plain" otherwise
    """

    html_indicators = ["<html", "<!doctype html", "<body"]

    content_lower = content.lower()
    for indicator in html_indicators:
        if indicator in content_lower:
            return "text/html"

    return "text/plain"


def validate_content_type(mime_type: str) -> Literal["text/html", "text/plain"]:
    """
    Validate and normalize content type to supported types.

    Args:
        mime_type: MIME type string to validate

    Returns:
        Normalized content type

    Raises:
        ValueError: If content type is not supported
    """
    if mime_type in ["text/html", "application/xhtml+xml", "application/html"]:
        return "text/html"
    elif mime_type == "text/plain":
        return "text/plain"
    else:
        raise ValueError(f"Unsupported content type: {mime_type}")
