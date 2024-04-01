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
        model: str | None,
        web_search: WebSearch,
        vision: Vision
    ) -> AssistantResponse:
        """
        Sends a message from user to assistant.

        Parameters
        ----------
        prompt : str
            User message.
        image_bytes : bytes | None
            Image of what user is looking at.
        message_history : List[Mesage] | None
            Conversation history, excluding current user message we will run inference on.
        local_time : str | None
            User's local time in a human-readable format, which helps the LLM answer questions where
            the user indirectly references the date or time. E.g.,
            "Saturday, March 30, 2024, 1:21 PM".
        location_address : str | None
            User's current location, specified as a full or partial address. This provides context
            to the LLM and is especially useful for web searches. E.g.,
            "3985 Stevens Creek Blvd, Santa Clara, CA 95051".
        model : str | None
            Assistant model. Valid values will depend on the assistant implementation (e.g., OpenAI-
            based assistants will take "gpt-3.5-turbo", etc.) A default will be selected if None is
            passed.
        web_search : WebSearch
            Web search provider, invoked when a web search (including a reverse image search) is
            needed.
        vision : Vision
            Vision AI provider, invoked when understanding of what user is looking at is required.

        Returns
        -------
        AssistantResponse
            Assistant response (text and some required analytics).
        """
        pass