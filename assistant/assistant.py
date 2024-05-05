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
    async def send_to_assistant(
        prompt: str,
        noa_system_prompt: str | None,
        image_bytes: bytes | None, 
        message_history: List[Message] | None, 
        learned_context: Dict[str, str],
        local_time: str | None,
        location_address: str | None,
        model: str | None,
        web_search: WebSearch,
        vision: Vision,
        direct_vision_response : bool,
        speculative_vision: bool
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
        learned_context : Dict[str, str]
            Learned context about the user, as key-value pairs.
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
        direct_vision_response : bool
            If vision is the *only* tool invoked, whether to return its raw response directly
            without a second LLM pass to synthesize the results. This may produce responses that are
            more literal and verbose, and less consistent with the tone of the conversation (because
            the vision tool lacks full conversational context).
        speculative_vision : bool
            Whether to perform speculative vision queries (if supported by assistant). This will run
            the vision tool in parallel with the initial LLM request in *all* cases, using the user 
            prompt as the query, but only use the result if the LLM then determines the vision tool
            should have been used. This reduces latency by the duration of the initial LLM call by
            giving the vision tool (which is usually slow) a head start.

        Returns
        -------
        AssistantResponse
            Assistant response (text and some required analytics).
        """
        pass