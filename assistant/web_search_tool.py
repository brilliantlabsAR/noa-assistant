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
from .models import PerplexityResponse, MessageChoices, PerplexityMessage, SYSTEM_MESSAGE_WEB

MODEL = "llama-3.1-sonar-small-128k-online"


class WebSearchTool:
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._session = None

    def __del__(self):
        if self._session:
            self._session.detach()

    async def _lazy_init(self):
        if self._session is None or self._session.closed:
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
        # make sure user and assistant messages are alternating
        if len(message_history) > 0:
            query = "Knowing the previous chat was:  " + message_history[-1].content + "\n" + query
        if query.strip() == "":
            query = flavor_prompt
        messages = [
            Message(role=Role.SYSTEM, content=self._system_message(flavor_prompt=flavor_prompt, location=location))
        ] + message_history + [
            Message(role=Role.USER, content=query)
        ]
        print(f"Web search tool: {messages}")

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
                return "Web search failed. Please try again."
            
            # Stream out response
            accumulated_response = ""
            usage = None
            t_first = None
            error = False
            async for line in response.content.iter_any():
                # Decode chunk from Perplexity
                try:
                    json_chunk = line.decode("utf-8").split("data: ")[1].strip()

                    chunk = PerplexityResponse.model_validate_json(json_chunk)
                    if t_first is None:
                        t_first = timeit.default_timer()
                    
                    # Accumulate response
                    accumulated_response = chunk.choices[0].message.content
                    usage = chunk.usage
                except Exception as e:
                    print(f"Error: Unable to decode chunk from Perplexity: {e}, chunk={line.decode('utf-8')}")
                    error = True
                    # return "Error: Web search failed. Inform the user they should try again."
                
            if error:
                # request again without stream
                payload["stream"] = False
                async with  self._session.post(url=url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        self._session.close()
                        raise Exception(f"Failed to get response from Perplexity: {await response.text()}")
                    try:
                        response_json = await response.json()
                        accumulated_response = response_json["choices"][0]["message"]["content"]
                        usage = CompletionUsage(**response_json["usage"])
                    except Exception as e:
                        self._session.close()
                        raise e
                        return "Web seacrh failed. Please try again."

            # Timings
            t_end = timeit.default_timer()
            if t_first is None:
                t_first = t_end
            timings["perplexity_first"] = t_first - t_start
            timings["perplexity_total"] = t_end - t_start

            # Accumulate token count
            if usage is not None:
                accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=usage, model=MODEL)
            if self._session:
                await self._session.close()
                self._session = None
            # Return final, accumulated response
            return accumulated_response
   
    @staticmethod
    def _system_message(flavor_prompt: str | None, location: str | None):
        system_message = SYSTEM_MESSAGE_WEB
        if location is None or len(location) == 0:
            location = "<you do not know user's location and if asked, tell them so>"
        
        system_message = system_message.replace("[LOCATION]", location)
        system_message += f"\nreply in concise and short with high accurancy from web results for the query"
        if flavor_prompt is not None:
            system_message = f"{flavor_prompt}\n{system_message}"
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
        # make sure user and assistant messages are alternating
        message_history.reverse()
        print(f"Pruned message history: {message_history}")
        message_history = WebSearchTool.make_alternating(messages=message_history)
        return message_history
    
    @staticmethod
    def make_alternating(messages: List[Message]) -> List[Message]:
        """
        Ensure that the messages are alternating between user and assistant.
        """
        if not messages:
            return []

        alternating_messages = []
        expected_role = messages[0].role

        for message in messages:
            # Skip empty content messages
            if message.content.strip() == '':
                continue

            # Only add messages that match the expected role
            if message.role == expected_role:
                alternating_messages.append(message)
                # Toggle the expected role for the next message
                expected_role = Role.USER if expected_role == Role.ASSISTANT else Role.ASSISTANT

        # Ensure the last message is from the assistant
        if alternating_messages and alternating_messages[-1].role != Role.ASSISTANT:
            alternating_messages.pop()

        # If the first message is from the assistant, remove it
        if alternating_messages and alternating_messages[0].role == Role.ASSISTANT:
            alternating_messages.pop(0)

        return alternating_messages
