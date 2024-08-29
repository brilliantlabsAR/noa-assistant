#
# response.py
#
# Defines AssistantResponse, used by the assistant to return streaming responses.
#

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from models import Capability, TokenUsage


@dataclass
class AssistantResponse:
    token_usage_by_model: Dict[str, TokenUsage]
    capabilities_used: List[Capability]
    response: str
    timings: Dict[str, float]
    image: str
    topic_changed: bool
