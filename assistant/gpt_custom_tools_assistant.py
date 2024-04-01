#
# gpt_custom_tools_assistant.py
#
# Assistant implementation based on OpenAI's GPT models. This assistant is capable of leveraging
# separate web search and vision tools, which are handled with a custom approach that avoids using
# function calling and should therefore be portable to other comparable LLMs.
#
# NOTE: This is not currently ready for use and is purely experimental. See gpt_assistant.py for a
# functional assistant using OpenAI's function calling API.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import timeit
from typing import Any, Callable, Dict, List, Tuple

import openai
from openai.types.completion import CompletionUsage

from .assistant import Assistant, AssistantResponse
from web_search import WebSearch, WebSearchResult
from vision import Vision
from models import Role, Message, Capability, TokenUsage, accumulate_token_usage


TOOL_REQUIREMENTS_SYSTEM_MESSAGE = """
You are a smart AI assistant that lives inside smart AR glasses worn by your user. You have access 
to a photo from the smart glasses camera of what the user was seeing at the time they spoke.

Your job is to analyze their intent and query given conversation history. Your output will be consumed
by another AI assistant with identical capabilities. NEVER respond to the user or answer questions.
ONLY generate instructions in the following format:

REQUESTING_INFO: {Does the user want information? YES or NO}
PHOTO_NEEDED: {Does photo need to be analyzed from user's perspective to understand what they are referring to? For example, if user uses demonstrative pronouns or refers to something that is not clear from the conversation history. YES or NO}
PHOTO_QUERY: {User's query rephrased as command to an intelligent vision AI tool that sees what user sees and answers any question}
WEB_NEEDED: {Did user ask for info that could best be found with web search? YES or NO}
CURRENT_INFO: {Is the user asking for information that may have changed after 2023? YES or NO}
GENERAL_KNOWLEDGE: {Is the user asking for trivia or general knowledge that Wikipedia would have? YES or NO}
SEARCH_QUERY: {If WEB_NEEDED is YES, a web search query to get that info, else leave blank. Do NOT insert any placeholders.}
"""

@dataclass
class ToolRequirements:
    photo_needed: bool
    photo_query: str
    web_needed: bool
    current_info: bool
    general_knowledge: str
    search_query: str

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. Make your responses short (one or two sentences) and precise.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "Additional context about the user:"

VISION_PHOTO_DESCRIPTION_SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do you best with images.

Make your responses short (one or two sentences) and precise. Remember that you can see the photo,
so do not talk about that, just respond directly.
"""

VISION_GENERATE_REVERSE_IMAGE_SEARCH_QUERY_FROM_PHOTO_SYSTEM_MESSAGE = """
you are photo tool, with help of photo and user's query, make a short (1 SENTENCE) and concise google search query that can be searched on internet with google reverse image search to answer the user.
"""

def parse_yes_no(text: str | None) -> bool:
    if text is None:
        return False
    return "yes" in text.strip().lower()

class Tool(ABC):
    @abstractmethod
    def run(self, user_query: str, image_bytes: bytes | None, location: str | None, local_time: str | None, token_usage_by_model: Dict[str, TokenUsage], capabilities_used: List[Capability]) -> str:
        pass

class VisionTool(ABC):
    def __init__(self, tool_requirements: ToolRequirements, vision: Vision, web_search: WebSearch, reverse_image_search: bool, token_usage_by_model: Dict[str, TokenUsage]):
        self._tool_requirements = tool_requirements
        self._vision = vision
        self._web_search = web_search
        self._reverse_image_search = reverse_image_search

    def run(self, user_query: str, image_bytes: bytes | None, location: str | None, local_time: str | None, token_usage_by_model: Dict[str, TokenUsage], capabilities_used: List[Capability]) -> str:
        query = self._tool_requirements.photo_query if len(self._tool_requirements.photo_query) > 0 else user_query
        if self._web_search:
            # NOTE: The web search flag is rarely set for photos (in fact, I have yet to see it). This
            # means that SEARCH_QUERY is usually blank. Usually, CURRENT_INFO is set instead. For now,
            # we do not pass in any history.
            capabilities_used.append(Capability.REVERSE_IMAGE_SEARCH)
            extra_context = GPTCustomToolsAssistant._create_context_system_message(local_time=local_time, location=location, learned_context=None)
            system_prompt = VISION_GENERATE_REVERSE_IMAGE_SEARCH_QUERY_FROM_PHOTO_SYSTEM_MESSAGE + extra_context
            vision_response = self._vision.query_image(
                system_message=system_prompt,
                query=query,
                image_bytes=image_bytes,
                token_usage_by_model=token_usage_by_model
            )
            return self._web_search.search_web(query=vision_response.strip("\""), use_photo=True, image_bytes=image_bytes, location=location)
        else:
            # Vision only
            capabilities_used.append(Capability.VISION)
            query = self._tool_requirements.photo_query if len(self._tool_requirements.photo_query) > 0 else user_query
            return self._vision.query_image(system_message=VISION_PHOTO_DESCRIPTION_SYSTEM_MESSAGE, query=query, image_bytes=image_bytes)

class WebSearchTool(ABC):
    def __init__(self, tool_requirements: ToolRequirements, web_search: WebSearch):
        self._tool_requirements = tool_requirements
        self._web_search = web_search

    def run(self, user_query: str, image_bytes: bytes | None, location: str | None, local_time: str | None, token_usage_by_model: Dict[str, TokenUsage], capabilities_used: List[Capability]) -> str:
        capabilities_used.append(Capability.WEB_SEARCH)
        query = self._tool_requirements.search_query if len(self._tool_requirements.search_query) > 0 else user_query
        result = self._web_search.search_web(query=query, use_photo=False, image_bytes=image_bytes, location=location)
        return result.summary

class GPTCustomToolsAssistant(Assistant):
    def __init__(self, client: openai.OpenAI, model: str = "gpt-3.5-turbo-1106"):
        self._client = client
        self._model = model
        self._learned_context: Dict[str,str] = {}

    def send_to_assistant(
        self,
        prompt: str,
        image_bytes: bytes | None, 
        message_history: List[Message] | None, 
        location_address: str | None,
        local_time: str | None,
        web_search: WebSearch,
        vision: Vision
    ) -> AssistantResponse:
        start = timeit.default_timer()

        # Prepare response data structure
        final_response = AssistantResponse(tokens_usage_by_model={}, capabilities_used=[], response="", debug_tools="")

        # Make copy of message history so we can modify it in-flight
        message_history = message_history.copy() if message_history else []

        # Append user message
        user_message = Message(role=Role.USER, content=prompt)
        message_history.append(user_message)

        # Step 1: Tool analysis 
        tool_requirements, tool_token_usage, = self._get_tool_requirements(message_history=message_history)
        accumulate_token_usage(
            token_usage_by_model=final_response.token_usage_by_model,
            model=self._model,
            input_tokens=tool_token_usage.prompt_tokens,
            output_tokens=tool_token_usage.completion_tokens,
            total_tokens=tool_token_usage.total_tokens
        )

        # Step 2: Tool selection
        tool = self._select_tool(tool_requirements=tool_requirements, web_search=web_search, vision=vision)

        # Step 3: Get tool result
        tool_result_system_message = None
        if tool is not None:
            tool_result = tool.run(
                user_query=prompt,
                image_bytes=image_bytes,
                location=location_address,
                local_time=local_time,
                token_usage_by_model=final_response.token_usage_by_model,
                capabilities_used=final_response.capabilities_used
            )
            tool_result_system_message = Message(role=Role.SYSTEM, content=f"Answer using the following information: {tool_result}")
        
        # Step 4: Final assistant response
        system_message = Message(role=Role.SYSTEM, content=SYSTEM_MESSAGE)
        message_history = [ message for message in message_history if message.role != Role.SYSTEM ]
        message_history.insert(0, system_message)
        if tool_result_system_message is not None:
            # Incorporate as additional system message right before final one
            message_history.append(tool_result_system_message)
        extra_context_system_message = Message(role=Role.SYSTEM, content=self._create_context_system_message(local_time=local_time, location=location_address, learned_context=None))
        message_history.append(extra_context_system_message)
        assistant_response = self._client.chat.completions.create(
            model=self._model,
            messages=message_history
        )
        assistant_token_usage = assistant_response.usage

        # If no tools were used, only assistant capability recorded
        if len(final_response.capabilities_used) == 0:
            final_response.capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)

        # Generate final response object
        accumulate_token_usage(
            token_usage_by_model=final_response.token_usage_by_model,
            model=self._model,
            input_tokens=assistant_token_usage.prompt_tokens,
            output_tokens=assistant_token_usage.completion_tokens,
            total_tokens=assistant_token_usage.total_tokens
        )
        final_response.response = assistant_response.choices[0].message.content
        final_response.debug_tools = ("No tools" if tool is None else tool.__class__.__name__) + f" - {tool_requirements}"

        # Return final response
        return final_response
    
    def _select_tool(self, tool_requirements: ToolRequirements, web_search: WebSearch, vision: Vision) -> Tool | None:
        web_needed = (tool_requirements.web_needed and not tool_requirements.general_knowledge) or tool_requirements.current_info
        if tool_requirements.photo_needed:
            return VisionTool(
                tool_requirements=tool_requirements,
                vision=vision,
                web_search=web_search,
                reverse_image_search=web_needed
            )
        if web_needed:
            return WebSearchTool(tool_requirements=tool_requirements, web_search=web_search)
        return None

    def _get_tool_requirements(self, message_history: List[Message]) -> Tuple[ToolRequirements, CompletionUsage]:
        # Filter out system messages and take only last N user messages
        n = 5
        message_history = message_history.copy()
        message_history = [ message for message in message_history if message.role == Role.USER ]
        message_history = message_history[len(message_history) - n : len(message_history)]

        # System prompt
        system_message = Message(role=Role.SYSTEM, content=TOOL_REQUIREMENTS_SYSTEM_MESSAGE)
        message_history.insert(0, system_message)

        # Invoke LLM
        response = self._client.chat.completions.create(
            model=self._model,
            messages=message_history
        )
        token_usage = response.usage
        text = response.choices[0].message.content
        print(text)

        # Parse out tools
        field_names = [ "PHOTO_NEEDED", "PHOTO_QUERY", "WEB_NEEDED", "CURRENT_INFO", "GENERAL_KNOWLEDGE", "SEARCH_QUERY" ]
        value_by_name = {}
        for line in text.splitlines():
            for name in field_names:
                name_idx = line.find(name)
                if name_idx >= 0:
                    value = line[name_idx + len(name) : ].strip().lstrip(":").strip()
                    value_by_name[name] = value
                    break

        # Create tools structure
        photo_query = value_by_name.get("PHOTO_QUERY") if value_by_name.get("PHOTO_QUERY") else ""
        search_query = value_by_name.get("SEARCH_QUERY") if value_by_name.get("SEARCH_QUERY") else ""
        return ToolRequirements(
            photo_needed=parse_yes_no(value_by_name.get("PHOTO_NEEDED")),
            photo_query=photo_query,
            web_needed=parse_yes_no(value_by_name.get("WEB_NEEDED")),
            current_info=parse_yes_no(value_by_name.get("CURRENT_INFO")),
            general_knowledge=parse_yes_no(value_by_name.get("GENERAL_KNOWLEDGE")),
            search_query=search_query
        ), token_usage
    
    @staticmethod
    def _create_context_system_message(local_time: str | None, location: str | None, learned_context: Dict[str,str] | None) -> str:
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

Assistant.register(GPTCustomToolsAssistant)