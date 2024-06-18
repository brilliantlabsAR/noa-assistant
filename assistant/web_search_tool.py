#
# web_search_tool.py
#
# Web search tool implementation based on Perplexity. Cannot perform searches with images.
#

import timeit
from typing import Dict, List

import aiohttp
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from models import Role, Message, TokenUsage, accumulate_token_usage

MODEL = "llama-3-sonar-small-32k-online"

class PerplexityMessage(BaseModel):
    role: str = None
    content: str = None

class MessageChoices(BaseModel):
    index: int = None
    finish_reason: str | None = None
    message: PerplexityMessage = None
    delta: PerplexityMessage = None

class PerplexityResponse(BaseModel):
    id: str = None
    model: str = None
    created: int = None
    usage: CompletionUsage = None
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
        flavor_prompt: str | None,
        message_history: List[Message],
        location: str | None = None
    ) -> str:
        t_start = timeit.default_timer()
        await self._lazy_init()

        message_history = self._prune_history(message_history=message_history)

        messages = [
            Message(role=Role.SYSTEM, content=self._system_message(flavor_prompt=flavor_prompt, location=location))
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
                return "Error: Web search failed. Inform the user that they should try again."
            
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
                    return "Error: Web search failed. Inform the user they should try again."
                
                if t_first is None:
                    t_first = timeit.default_timer()
                
                # Accumulate response
                accumulated_response = chunk.choices[0].message.content
                usage = chunk.usage

            # Timings
            t_end = timeit.default_timer()
            if t_first is None:
                t_first = t_end
            timings["perplexity_first"] = t_first - t_start
            timings["perplexity_total"] = t_end - t_start

            # Accumulate token count
            if usage is not None:
                accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=usage, model=MODEL)

            # Return final, accumulated response
            return accumulated_response
   
    @staticmethod
    def _system_message(flavor_prompt: str | None, location: str | None):
        if location is None or len(location) == 0:
            location = "<you do not know user's location and if asked, tell them so>"
        system_message = f"reply in concise and short with high accurancy from web results if needed take location as {location}"
        if flavor_prompt is not None:
            system_message = f"{system_message}\n{flavor_prompt}"
        return system_message

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
