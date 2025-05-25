"""Simple HTTP utilities for Sifaka models.

This module provides a minimal utility function to reduce repetitive
HTTP session setup across model implementations.
"""

from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def setup_http_session(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Optional[List[int]] = None,
) -> requests.Session:
    """Set up a requests session with retry logic.

    Args:
        max_retries: Maximum number of retries.
        backoff_factor: Backoff factor for retries.
        status_forcelist: HTTP status codes to retry on.

    Returns:
        A configured requests.Session instance.
    """
    if status_forcelist is None:
        status_forcelist = [429, 500, 502, 503, 504]

    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session
