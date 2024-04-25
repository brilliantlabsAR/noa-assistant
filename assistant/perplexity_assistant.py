#
# perplexity_assistant.py
#
# Assistant implementation based on Perplexity's LLM. Currently incapable of leveraging any tools,
# but includes built-in web search. Images not supported.
#

import requests
from typing import Any, Dict, List

import aiohttp
from pydantic import BaseModel

from .assistant import Assistant, AssistantResponse
from web_search import WebSearch
from vision import Vision
from models import Role, Message, Capability, accumulate_token_usage


class PerplexityMessage(BaseModel):
    role: str = None
    content: str = None

class MessageChoices(BaseModel):
    index: int = None
    finish_reason: str = None
    message: PerplexityMessage = None
    delta: dict = None

class Usage(BaseModel):
    prompt_tokens: int = None
    completion_tokens: int = None
    total_tokens: int = None

class PerplexityResponse(BaseModel):
    id: str = None
    model: str = None
    created: int = None
    usage: Usage = None
    object: str = None
    choices: List[MessageChoices] = None
    
    def summarise(self, count: int=5) -> str:
        if len(self.choices) >0:
            return self.choices[0].message.content
        else:
            return "No results found"
    
class PerplexityAssistant(Assistant):
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._session = None

    def __del__(self):
        self._session.detach()

    async def _lazy_init(self):
        if self._session is None:
            # This instantiation must happen inside of an async event loop
            self._session = aiohttp.ClientSession()

    async def send_to_assistant(
        self,
        prompt: str,
        image_bytes: bytes | None, 
        message_history: List[Message] | None, 
        learned_context: Dict[str, str],
        local_time: str | None,
        location_address: str | None,
        model: str | None,
        web_search: WebSearch,
        vision: Vision
    ) -> AssistantResponse:
        await self._lazy_init()

        # Default model
        model = model if model is not None else "pplx-7b-online"

        # Prepare response datastructure
        returned_response = AssistantResponse(token_usage_by_model={}, capabilities_used=[ Capability.ASSISTANT_KNOWLEDGE ], response="", debug_tools="")

        # Make copy of message history so we can modify it in-flight during tool use
        message_history = message_history.copy() if message_history else None

        # Add user message to message history or create a new one if necessary
        user_message = Message(role=Role.USER, content=prompt)
        system_message = Message(role=Role.SYSTEM, content=self._system_message(location=location_address))
        if not message_history:
            message_history = []
        if len(message_history) == 0:
            message_history = [ system_message ]
        else:
            # Insert system message before message history
            message_history.insert(0, system_message)
        message_history.append(user_message)

        # Call Perplexity
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": model,
            "messages": [ message.model_dump() for message in message_history ]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}"
        }
        json_text = await self._post(url=url, json=payload, headers=headers)
        if json_text is None:
            returned_response.response = "The assistant did not respond. Please try again."
            return returned_response
        print(json_text)

        # Return results
        perplexity_data = PerplexityResponse.model_validate_json(json_text)
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model=model,
            input_tokens=perplexity_data.usage.prompt_tokens,
            output_tokens=perplexity_data.usage.completion_tokens,
            total_tokens=perplexity_data.usage.total_tokens
        )
        returned_response.response = perplexity_data.choices[0].message.content if len(perplexity_data.choices) > 0 else ""
        return returned_response

    async def _post(self, url: str, json: Dict[str, Any], headers: Dict[str, str]) -> str | None:
        async with self._session.post(url=url, json=json, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to get resonse from Perplexity: {await response.text()}")
                return None
            return await response.text()

    @staticmethod
    def _system_message(location: str | None):
        if location is None or len(location) == 0:
            location = "<you do not know user's location and if asked, tell them so>"
        return f"reply in concise and short with high accurancy from web results if needed take location as {location}"

Assistant.register(PerplexityAssistant)