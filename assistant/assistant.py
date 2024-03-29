#
# assistant.py
#
# Assistant base class and associated data structures.
#

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from pydantic import BaseModel

from models import Message, Capability, TokenUsage
from web_search import WebSearch
from vision import Vision

@dataclass
class AssistantResponse:
    token_usage_by_model: Dict[str, TokenUsage]
    capabilities_used: List[Capability]
    response: str
    debug_tools: str    # debugging information about tools used (no particular format guaranteed)

class Assistant(ABC):
    @abstractmethod
    def send_to_assistant(
        prompt: str,
        image_bytes: bytes | None, 
        message_history: List[Message] | None, 
        local_time: str | None,
        location_address: str | None,
        web_search: WebSearch | None = None,
        vision: Vision | None = None,
    ) -> AssistantResponse:
        pass