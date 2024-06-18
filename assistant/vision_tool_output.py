#
# vision_tool_output.py
#
# Output of vision tool.
#

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionToolOutput:
    is_error: bool
    response: str
    web_query: Optional[str] = None
