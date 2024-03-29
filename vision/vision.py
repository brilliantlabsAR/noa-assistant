#
# vision.py
#
# Vision tool base class.
#

from abc import ABC, abstractmethod
from typing import Dict

from models import TokenUsage


class Vision(ABC):
    @abstractmethod
    def query_image(self, system_message: str, query: str, image_bytes: bytes | None, token_usage_by_model: Dict[str, TokenUsage]) -> str:
        pass