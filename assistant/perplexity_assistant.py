#
# perplexity_assistant.py
#
# Assistant implementation based on Perplexity's LLM. Currently incapable of leveraging any tools,
# but includes built-in web search. Images not supported.
#

import requests
from typing import List

from pydantic import BaseModel

from .assistant import Assistant, AssistantResponse
from web_search import WebSearch
from vision import Vision
from models import Role, Message, Capability, TokenUsage, accumulate_token_usage


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

    def send_to_assistant(
        self,
        prompt: str,
        image_bytes: bytes | None, 
        message_history: List[Message] | None, 
        local_time: str | None,
        location_address: str | None,
        web_search: WebSearch,
        vision: Vision
    ) -> AssistantResponse:
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
            "model": "pplx-7b-online",
            "messages": [ message.model_dump() for message in message_history ]
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}"
        }
        response = requests.post(url, json=payload, headers=headers)
        print(response)

        # Return results
        perplexity_data = PerplexityResponse.model_validate_json(response.text)
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model="pplx-7b-online",
            input_tokens=perplexity_data.usage.prompt_tokens,
            output_tokens=perplexity_data.usage.completion_tokens,
            total_tokens=perplexity_data.usage.total_tokens
        )
        returned_response.response = perplexity_data.choices[0].message.content if len(perplexity_data.choices) > 0 else ""
        return returned_response

    @staticmethod
    def _system_message(location: str | None):
        return f"reply in concise and short with high accurancy from web results if needed take location as {location}"

Assistant.register(PerplexityAssistant)