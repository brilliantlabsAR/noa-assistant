#
# perplexity.py
#
# Web search tool implementation based on Perplexity. Cannot perform searches with images.
#

from typing import Any, Dict, List

import aiohttp
from pydantic import BaseModel
import json
from web_search import WebSearch, WebSearchResult
from models import Role, Message, TokenUsage, accumulate_token_usage

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
    
    def summarise(self) -> str:
        if len(self.choices) > 0:
            return self.choices[0].message.content
        else:
            return "No results"

class PerplexityWebSearch(WebSearch):
    def __init__(self, api_key: str, model: str = "llama-3-sonar-small-32k-online"):
        super().__init__()
        self._api_key = api_key
        self._model = model
        self._session = None
        self._stream = True

    def __del__(self):
        if self._session:
            self._session.detach()

    async def _lazy_init(self):
        if self._session is None:
            # This instantiation must happen inside of an async event loop
            self._session = aiohttp.ClientSession()
    
    async def search_web(
        self,
        query: str,
        message_history: List[Message] | None,
        token_usage_by_model: Dict[str, TokenUsage],
        use_photo: bool = False,
        image_bytes: bytes | None = None,
        location: str | None = None
    ) -> WebSearchResult:
        await self._lazy_init()

        message_history = [] if message_history is None else message_history.copy()
        message_history = self._prune_history(message_history=message_history)

        messages = [
            Message(role=Role.SYSTEM, content=self._system_message(location=location))
        ] + message_history + [
            Message(role=Role.USER, content=query)
        ]
        print(messages)

        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": self._model,
            "messages": [ message.model_dump() for message in messages ],
            "stream": self._stream,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}"
        }
        json_text = await self._post(url=url, payload=payload, headers=headers)
        if json_text is None:
            return WebSearchResult(summary="No results", search_provider_metadata="")

        # Return results
        # print(json_text)
        try:
            perplexity_data = PerplexityResponse.model_validate_json(json_text)
        except Exception as e:
            print(json_text)
            print(f"Failed to parse Perplexity response: {e}")
            return WebSearchResult(summary="No results", search_provider_metadata="")
        accumulate_token_usage(
            token_usage_by_model=token_usage_by_model,
            model=self._model,
            input_tokens=perplexity_data.usage.prompt_tokens,
            output_tokens=perplexity_data.usage.completion_tokens,
            total_tokens=perplexity_data.usage.total_tokens
        )
        search_result = perplexity_data.choices[0].message.content if len(perplexity_data.choices) > 0 else "No results"
        return WebSearchResult(summary=search_result, search_provider_metadata="")
    
    async def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str | None:
        async with self._session.post(url=url, json=payload, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to get response from Perplexity: {await response.text()}")
                return None
            if self._stream:
                return_response = ""
                async for line in response.content.iter_any():
                    return_response = line.decode("utf-8").split("data: ")[1].strip()
                return return_response
            return await response.text()

    @staticmethod
    def _system_message(location: str | None):
        if location is None or len(location) == 0:
            location = "<you do not know user's location and if asked, tell them so>"
        return f"reply in concise and short with high accurancy from web results if needed take location as {location}"

    @staticmethod
    def _prune_history(
        message_history: List[Message],
        max_messages: int = 8
     ) -> List[Message]:
        """
        Prunes down the chat history to save tokens, improving inference speed and reducing cost.
        Generally, preserving all assistant responses is not needed, and only a limited number of
        user messages suffice to maintain a coherent conversation.

        Parameters
        ----------
        message_history : List[Message]
            Conversation history. This list will be mutated and returned.
        max_messages : int
            Maximum number of messages to preserve. Must be an even number because Perplexity
            requires alternating user and assistant messages.

        Returns
        -------
        List[Message]
            Pruned history. This is the same list passed as input.
        """
        if max_messages %2 != 0:
            print("ERROR: Discarding invalid message history for Perplexity. Require alternating user/assistant messages!")
            return []
        message_history.reverse()
        message_history = [ message for message in message_history if message.role != Role.SYSTEM ]
        message_history = message_history[0:max_messages]
        message_history.reverse()
        return message_history

WebSearch.register(PerplexityWebSearch)