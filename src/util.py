import requests
import base64
from urllib.parse import urlparse


def fetch_html_content(source_url: str) -> str:
    """
    Fetch HTML content from various sources: local file, HTTP URL, or data URL.

    Args:
        source_url: file URL, HTTP URL, or data URL containing HTML content

    Returns:
        HTML content as string

    Raises:
        ValueError: If the source type is not supported or content cannot be retrieved
    """
    parsed = urlparse(str(source_url))

    # Handle local file
    if not parsed.scheme or parsed.scheme == "file":
        with open(parsed.path, "r") as file:
            return file.read()

    # Handle HTTP(S) URLs
    if parsed.scheme in ["http", "https"]:
        response = requests.get(str(source_url))
        response.raise_for_status()
        return response.text

    # Handle data URLs
    if parsed.scheme == "data":
        # Split metadata and content
        header, encoded = source_url.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(encoded).decode("utf-8")
        return encoded

    raise ValueError(f"Unsupported source type: {parsed.scheme}")
