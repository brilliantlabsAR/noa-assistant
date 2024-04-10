#
# vision.py
#
# Vision tool base class.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from models import TokenUsage


@dataclass
class VisionOutput:
    response: str
    web_query: str
    reverse_image_search: bool

    def web_search_needed(self):
        return len(self.web_query) > 0

class Vision(ABC):
    @abstractmethod
    async def query_image(self, query: str, extra_context: str, image_bytes: bytes | None, token_usage_by_model: Dict[str, TokenUsage]) -> VisionOutput | None:
        pass