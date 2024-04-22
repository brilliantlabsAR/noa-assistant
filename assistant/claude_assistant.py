#
# claude_assistant.py
#
# Assistant implementation based on Anthropic's Claude series of models.
#

#
# TODO:
# -----
# - Factor out functions common to ClaudeAssistant and GPTAssistant.
#

from typing import Dict, List

import anthropic

from .assistant import Assistant, AssistantResponse
from web_search import WebSearch, WebSearchResult
from vision import Vision
from models import Role, Message, Capability, TokenUsage, accumulate_token_usage


####################################################################################################
# Prompts
####################################################################################################

#
# Top-level instructions
#

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly. When analyzing the user's view, speak as if you can actually see and never make references
to the photo or image you analyzed.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


####################################################################################################
# Tools
####################################################################################################


####################################################################################################
# Assistant Class
####################################################################################################

class ClaudeAssistant(Assistant):
    def __init__(self, client: anthropic.AsyncAnthropic):
        self._client = client
    
    # Refer to definition of Assistant for description of parameters
    async def send_to_assistant(
        self,
        prompt: str,
        image_bytes: bytes | None,
        message_history: List[Message] | None,
        location_address: str | None,
        local_time: str | None,
        model: str | None,
        web_search: WebSearch,
        vision: Vision
    ) -> AssistantResponse:
        model = model if model is not None else "claude-3-haiku-20240307"

        # Prepare response datastructure
        returned_response = AssistantResponse(token_usage_by_model={}, capabilities_used=[], response="", debug_tools="")

        # Make copy of message history so we can modify it in-flight during tool use
        message_history = message_history.copy() if message_history else []

        # Claude does not have a system role. Rather, a top-level system parameter must be supplied.
        # However, our API uses the OpenAI format. Therefore, we search for an existing system
        # message and, if it was supplied by the client, use that as the system message.
        system_text = SYSTEM_MESSAGE
        client_system_messages = [ message for message in message_history if message.role == Role.SYSTEM ]
        if len(client_system_messages) > 0:
            system_text = client_system_messages[0].content
        message_history = [ message for message in message_history if message.role != Role.SYSTEM ]

        # Add user's latest prompt
        user_message = Message(role=Role.USER, content=prompt)
        message_history.append(user_message)
        message_history = self._prune_history(message_history=message_history)

        # Learned context (TODO: implement me)
        learned_context = {}

        # Extra context to inject
        extra_context = self._create_context_system_message(local_time=local_time, location=location_address, learned_context=learned_context)

        # Initial Claude 
        first_response = await self._client.messages.create(
            model=model,
            system=system_text + "\n\n" + extra_context,
            messages=message_history,
            max_tokens=4096
        )

        # Aggregate token counts and potential initial response
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model=model,
            input_tokens=first_response.usage.input_tokens,
            output_tokens=first_response.usage.output_tokens,
            total_tokens=first_response.usage.input_tokens + first_response.usage.output_tokens
        )

        returned_response.response = first_response.content[0].text
        return returned_response

    @staticmethod
    def _prune_history(message_history: List[Message]) -> List[Message]:
        """
        Prunes down the chat history to save tokens, improving inference speed and reducing cost.
        Generally, preserving all assistant responses is not needed, and only a limited number of
        user messages suffice to maintain a coherent conversation.

        Parameters
        ----------
        message_history : List[Message]
            Conversation history. This list will be mutated and returned.

        Returns
        -------
        List[Message]
            Pruned history. This is the same list passed as input.
        """
        # Limit to most recent 5 user messages and 3 assistant responses
        assistant_messages_remaining = 3
        user_messages_remaining = 5
        message_history.reverse()
        i = 0
        while i < len(message_history):
            if message_history[i].role == Role.ASSISTANT:
                if assistant_messages_remaining == 0:
                    del message_history[i]
                else:
                    assistant_messages_remaining -= 1
                    i += 1
            elif message_history[i].role == Role.USER:
                if user_messages_remaining == 0:
                    del message_history[i]
                else:
                    user_messages_remaining -= 1
                    i += 1
            else:
                i += 1
        message_history.reverse()
        return message_history

    @staticmethod
    def _create_context_system_message(local_time: str | None, location: str | None, learned_context: Dict[str,str] | None) -> str:
        """
        Creates a string of additional context that can either be appended to the main system
        message or as a secondary system message before delivering the assistant response. This is
        how GPT is made aware of the user's location, local time, and any learned information that
        was extracted from prior conversation.

        Parameters
        ----------
        local_time : str | None
            Local time, if known.
        location : str | None
            Location, as a human readable address, if known.
        learned_context : Dict[str,str] | None
            Information learned from prior conversation as key-value pairs, if any.

        Returns
        -------
        str
            Message to combine with existing system message or to inject as a new, extra system
            message.
        """
        # Fixed context: things we know and need not extract from user conversation history
        context: Dict[str, str] = {}
        if local_time is not None and len(local_time) > 0:
            context["current_time"] = local_time
        else:
            context["current_time"] = "If asked, tell user you don't know current date or time because clock is broken"
        if location is not None and len(location) > 0:
            context["location"] = location
        else:
            context["location"] = "You do not know user's location and if asked, tell them so"

        # Merge in learned context
        if learned_context is not None:
            context.update(learned_context)

        # Convert to a list to be appended to a system message or treated as a new system message
        system_message_fragment = CONTEXT_SYSTEM_MESSAGE_PREFIX + "\n".join([ f"<{key}>{value}</{key}>" for key, value in context.items() if value is not None ])
        return system_message_fragment

Assistant.register(ClaudeAssistant)