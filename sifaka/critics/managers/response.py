"""
Response parser for critics.

This module is deprecated and has been consolidated into sifaka.core.managers.response.
Import directly from sifaka.core.managers.response instead.

## Examples

```python
# Old usage (deprecated)
from sifaka.critics.managers.response import ResponseParser

# New usage (recommended)
from sifaka.core.managers.response import ResponseParser, create_response_parser
```
"""

from sifaka.core.managers.response import ResponseParser as CoreResponseParser


class ResponseParser(CoreResponseParser):
    """
    Parses responses from language models.

    This class is deprecated and has been consolidated into sifaka.core.managers.response.
    It inherits from the core ResponseParser for backward compatibility.

    Please use sifaka.core.managers.response.ResponseParser directly instead.
    """

    pass
