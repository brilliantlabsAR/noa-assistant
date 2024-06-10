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
    stream_finished: bool   # for streaming versions, indicates final response chunk

    @staticmethod
    def _error_response(message: str) -> AssistantResponse:
        """
        Generates an error response intended as a tool output.

        Parameters
        ----------
        message : str
            Error message, intended to be consumed by LLM tool completion pass.

        Returns
        -------
        AssistantResponse
            AssistantResponse object.
        """
        return AssistantResponse(
            token_usage_by_model={},    # TODO
            capabilities_used=[],
            response=message,
            timings={},
            image="",
            stream_finished=True
        )