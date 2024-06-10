#
# web_search_tool.py
#
# Web search tool implementation based on Perplexity. Cannot perform searches with images.
#

import asyncio
import timeit
from typing import Dict, List

import aiohttp
from pydantic import BaseModel

from .response import AssistantResponse
from models import Capability, Role, Message, TokenUsage

MODEL = "llama-3-sonar-small-32k-online"

class PerplexityMessage(BaseModel):
    role: str = None
    content: str = None

class MessageChoices(BaseModel):
    index: int = None
    finish_reason: str | None = None
    message: PerplexityMessage = None
    delta: PerplexityMessage = None

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

class WebSearchTool:
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._session = None

    def __del__(self):
        if self._session:
            self._session.detach()

    async def _lazy_init(self):
        if self._session is None:
            # This instantiation must happen inside of an async event loop
            self._session = aiohttp.ClientSession()
    
    async def search_web(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        timings: Dict[str, float],
        query: str,
        message_history: List[Message],
        location: str | None = None
    ): 
        t_start = timeit.default_timer()
        await self._lazy_init()

        message_history = self._prune_history(message_history=message_history)

        messages = [
            Message(role=Role.SYSTEM, content=self._system_message(location=location))
        ] + message_history + [
            Message(role=Role.USER, content=query)
        ]

        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": MODEL,
            "messages": [ message.model_dump() for message in messages ],
            "stream": True,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self._api_key}"
        }

        async with self._session.post(url=url, json=payload, headers=headers) as response:
            if response.status != 200:
                print(f"Failed to get response from Perplexity: {await response.text()}")
                yield AssistantResponse._error_response(message="Error: Web search failed. Inform the user that they should try again.")
                return
            
            # Stream out response
            accumulated_response = ""
            usage = None
            t_first = None
            async for line in response.content.iter_any():
                # Decode chunk from Perplexity
                json_chunk = line.decode("utf-8").split("data: ")[1].strip()
                try:
                    chunk = PerplexityResponse.model_validate_json(json_chunk)
                except Exception as e:
                    print(f"Error: Unable to decode chunk from Perplexity: {e}, chunk={json_chunk}")
                    yield AssistantResponse._error_response(message="Error: Web search failed. Inform the user they should try again.")
                    return
                
                if t_first is None:
                    t_first = timeit.default_timer()
                
                # Accumulate response
                content = self._get_delta_content(old_content=accumulated_response, new_content=chunk.choices[0].message.content)
                accumulated_response = chunk.choices[0].message.content
                usage = chunk.usage

                # Yield a partial response
                yield AssistantResponse(
                    token_usage_by_model={},
                    capabilities_used=[],
                    response=content,
                    timings={},
                    image="",
                    stream_finished=False
                )

            # Timings
            t_end = timeit.default_timer()
            if t_first is None:
                t_first = t_end
            timings["perplexity_first"] = t_first - t_start
            timings["perplexity_total"] = t_end - t_start

            # Accumulate token count
            if usage is not None:
                token_usage = TokenUsage(
                    input=usage.prompt_tokens,
                    output=usage.completion_tokens,
                    total=usage.total_tokens
                )
                if MODEL not in token_usage_by_model:
                    token_usage_by_model[MODEL] = token_usage
                else:
                    token_usage_by_model[MODEL].add(token_usage=token_usage)

            # Yield final, accumulated response with usage
            yield AssistantResponse(
                token_usage_by_model=token_usage_by_model,
                capabilities_used=[ Capability.WEB_SEARCH ],
                response=accumulated_response,
                timings=timings,
                image="",
                stream_finished=True
            )
    
    @staticmethod
    def _get_delta_content(old_content: str, new_content: str) -> str:
        # Perplexity's delta messages are broken but they do send cumulative content with each 
        # update, so we just use that to compute our own delta response.
        old_len = len(old_content)
        if len(new_content) < old_len:
            # This should never happen
            return ""
        return new_content[old_len:]
    
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
            Conversation history.
        max_messages : int
            Maximum number of messages to preserve. Must be an even number because Perplexity
            requires alternating user and assistant messages.

        Returns
        -------
        List[Message]
            Pruned history.
        """
        if max_messages %2 != 0:
            print("ERROR: Discarding invalid message history for Perplexity. Require alternating user/assistant messages!")
            return []
        message_history = message_history.copy()
        message_history.reverse()
        message_history = [ message for message in message_history if message.role != Role.SYSTEM ]
        message_history = message_history[0:max_messages]
        message_history.reverse()
        return message_history
