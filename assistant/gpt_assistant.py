#
# gpt_assistant.py
#
# Assistant implementation based on OpenAI's GPT models. This assistant is capable of leveraging
# separate web search and vision tools.
#

#
# TODO:
# -----
# - Investigate SerpAPI failures, if they still happen.
# - Need everything to be made async.
# - Figure out how to get assistant to stop referring to "photo" and "image" when analyzing photos.
# - Improve people search.
#

import json
import time
import timeit
from typing import Any, Dict, List

import openai
from openai.types import CompletionUsage

from .assistant import Assistant, AssistantResponse
from web_search import WebSearch, WebSearchResult
from vision import Vision
from models import Role, Message, Capability, TokenUsage, accumulate_token_usage


SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

Make your responses short (one or two sentences) and precise. Respond without any preamble when giving
translations, just translate directly. When analyzing the user's view, speak as if you can actually
see and never make references to the photo or image you analyzed.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"

VISION_PHOTO_DESCRIPTION_SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but you NEVER mention the photo or image and instead respond as if you
are actually seeing. 

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do you best with images.

Make your responses short (one or two sentences) and precise. Respond without any preamble when giving
translations, just translate directly. When analyzing the user's view, speak as if you can actually
see and never make references to the photo or image you analyzed.
"""

VISION_GENERATE_SEARCH_DESCRIPTION_FROM_PHOTO_SYSTEM_MESSAGE = """
you are photo tool, with help of photo and user's query, make a short (1 SENTENCE) and concise google search query that can be searched on internet to answer the user.
"""

VISION_GENERATE_REVERSE_IMAGE_SEARCH_QUERY_FROM_PHOTO_SYSTEM_MESSAGE = """
you are photo tool, with help of photo and user's query, make a short (1 SENTENCE) and concise google search query that can be searched on internet with google reverse image search to answer the user.
"""

# These are context keys we try to detect in conversation history over time
LEARNED_CONTEXT_KEY_DESCRIPTIONS = {
    "UserName": "User's name",
    "DOB": "User's date of birth",
    "Food": "Foods and drinks user has expressed interest in"
}

LEARNED_CONTEXT_EXTRACTION_SYSTEM_MESSAGE = f"""
Given a transcript of what the user said, look for any of the following information being revealed:

""" + "\n".join([ key + ": "  + description for key, description in LEARNED_CONTEXT_KEY_DESCRIPTIONS.items() ]) + """

Make sure to list them in this format:

KEY=VALUE

If nothing was found, just say "END". ONLY PRODUCE ITEMS WHEN THE USER HAS ACTUALLY REVEALED THEM.
"""

DUMMY_SEARCH_TOOL_NAME = "general_knowledge_search"
SEARCH_TOOL_NAME = "search"
PHOTO_TOOL_NAME = "analyze_photo"
PHOTO_TOOL_QUERY_PARAM_NAME = "query"
PHOTO_TOOL_WEB_SEARCH_PARAM_NAME = "google_reverse_image_search"
PHOTO_TOOL_TRANSLATION_PARAM_NAME = "translate"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": DUMMY_SEARCH_TOOL_NAME,
            "description": """Trivial and general knowledge that would be expected to exist in Wikipedia or an encyclopedia""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "search query",
                    },
                },
                "required": [ "query" ]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": """Provides up to date information on news, retail products, current events, and esoteric knowledge""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "search query",
                    },
                },
                "required": [ "query" ]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": PHOTO_TOOL_NAME,
            "description": """Analyzes or describes the photo you have from the user's current perspective.
Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.""",
            "parameters": {
                "type": "object",
                "properties": {
                    PHOTO_TOOL_QUERY_PARAM_NAME: {
                        "type": "string",
                        "description": "User's query to answer, describing what they want answered, expressed as a command that NEVER refers to the photo or image itself"
                    },
                    PHOTO_TOOL_WEB_SEARCH_PARAM_NAME: {
                        "type": "boolean",
                        "description": "True ONLY if user wants to look up facts about contents of photo online (simply identifying what is in the photo does not count), otherwise always false"
                    },
                    PHOTO_TOOL_TRANSLATION_PARAM_NAME: {
                        "type": "boolean",
                        "description": "Translation of something in user's view required"
                    }
                },
                "required": [ PHOTO_TOOL_QUERY_PARAM_NAME, PHOTO_TOOL_WEB_SEARCH_PARAM_NAME, PHOTO_TOOL_TRANSLATION_PARAM_NAME ]
            },
        },
    },
]

class GPTAssistant(Assistant):
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
        web_search: WebSearch | None = None,
        vision: Vision | None = None
    ) -> AssistantResponse:
        start = timeit.default_timer()

        # Prepare response datastructure
        returned_response = AssistantResponse(token_usage_by_model={}, capabilities_used=[], response="", debug_tools="")

        print("Assistant input:")

        # Make copy of message history so we can modify it in-flight during tool use
        message_history = message_history.copy() if message_history else None

        # Add user message to message history or create a new one if necessary
        user_message = Message(role=Role.USER, content=prompt)
        system_message = Message(role=Role.SYSTEM, content=SYSTEM_MESSAGE)
        if not message_history:
            message_history = []
        if len(message_history) == 0:
            message_history = [ system_message ]
        else:
            # Insert system message before message history, unless client transmitted one they want
            # to use
            if len(message_history) > 0 and message_history[0].role != Role.SYSTEM:
                message_history.insert(0, system_message)
        message_history.append(user_message)
        message_history = self._prune_history(message_history=message_history)

        # Update learned context by analyzing last N messages.
        # TODO: this was for demo purposes and needs to be made more robust. Should be triggered 
        #       periodically or when user asks something for which context is needed.
        #self._learned_context.update(self._extract_learned_context(message_history=message_history, token_usage=returned_response.tokens_used))
        
        # Inject context into our copy by appending it to system message
        message_history[0].content += "\n\n" + self._create_context_system_message(local_time=local_time, location=location_address, learned_context=self._learned_context)
        # Initial GPT call, which may request tool use
        first_response = self._client.chat.completions.create(
            model=self._model,
            messages=message_history,
            tools=TOOLS,
            tool_choice="auto"
        )
        first_response_message = first_response.choices[0].message

        # Aggregate token counts and potential initial response
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model=self._model,
            input_tokens=first_response.usage.prompt_tokens,
            output_tokens=first_response.usage.completion_tokens,
            total_tokens=first_response.usage.total_tokens
        )

        print("Assistant output:")
        print(first_response_message.content)
        returned_response.response = first_response_message.content

        # Handle tool requests
        available_functions = {
            SEARCH_TOOL_NAME: web_search.search_web,                # returns WebSearchResult
            PHOTO_TOOL_NAME: self._handle_photo_tool,               # returns WebSearchResult | str
            DUMMY_SEARCH_TOOL_NAME: self._handle_general_knowledge,
        }
        tools_used = []
        tools_used.append({ "learned_context": self._learned_context }) # log context here for now
        if first_response_message.tool_calls:
            # Append initial response to history, which may include tool use
            message_history.append(first_response_message)
        
            for tool_call in first_response_message.tool_calls:
                # Determine which tool and what arguments to call with
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                if function_to_call is None:
                    # Error: GPT hallucinated a tool! We need to determine a way to handle this more 
                    # gracefully if it happens.
                    tools_used.append(self._create_hallucinated_tool_info_object(function_name=function_name))
                    # For now, remove last message (initial assistant response), then break out of tool
                    # loop. This will fall-through to re-try the initial message *without* tool use
                    # while returning the hallucinated tool above for debug logging.
                    message_history = message_history[:-1]
                    break

                # Ensure tools are getting the function args they require
                function_args = json.loads(tool_call.function.arguments)
                print(function_args,f"tool: {function_name}")
                function_args["location"] = location_address if location_address else "unknown"
                if "query" not in function_args:
                    function_args["query"] = prompt # all functions currently take this and if this wasn't generated by GPT, we must provide it
                if function_name in [ PHOTO_TOOL_NAME ]:
                    function_args["image_bytes"] = image_bytes
                    function_args["vision"] = vision
                    function_args["web_search"] = web_search
                    function_args["local_time"] = local_time
                    function_args["learned_context"] = self._learned_context
                    function_args["token_usage_by_model"] = returned_response.token_usage_by_model
                    function_args["capabilities_used"] = returned_response.capabilities_used

                # Call the tool
                tool_start_time = timeit.default_timer()
                function_response: WebSearchResult | str = function_to_call(**function_args)

                # Record capability used (except for case of photo tool, which reports on its own
                # because it represents multiple capabilities)
                if function_name == SEARCH_TOOL_NAME:
                    returned_response.capabilities_used.append(Capability.WEB_SEARCH)
                elif function_name == DUMMY_SEARCH_TOOL_NAME:
                    returned_response.capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)

                # Tool information
                total_tool_time = round(timeit.default_timer() - tool_start_time, 3)
                tools_used.append(
                    self._create_debug_tool_info_object(
                        function_name=function_name,
                        function_args=function_args,
                        tool_time=total_tool_time,
                        search_result=function_response.search_provider_metadata if isinstance(function_response, WebSearchResult) else None
                    )
                )

                # Format response appropriately
                assert isinstance(function_response, WebSearchResult) or isinstance(function_response, str)
                tool_output = function_response.summary if isinstance(function_response, WebSearchResult) else function_response
                
                # Append function response for GPT to continue
                message_history.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output,
                    }
                )

            # Get final response from model
            second_response = self._client.chat.completions.create(
                model=self._model,
                messages=message_history
            )

            # Aggregate tokens and response
            accumulate_token_usage(
                token_usage_by_model=returned_response.token_usage_by_model,
                model=self._model,
                input_tokens=second_response.usage.prompt_tokens,
                output_tokens=second_response.usage.completion_tokens,
                total_tokens=second_response.usage.total_tokens
            )
            returned_response.response = second_response.choices[0].message.content

        # Return final response   
        returned_response.debug_tools = json.dumps(tools_used)
        stop = timeit.default_timer()
        print(f"Time taken: {stop-start:.3f}")

        return returned_response

    @staticmethod
    def _handle_general_knowledge(
        query: str,
        image_bytes: bytes | None = None,
        local_time: str | None = None,
        location: str | None = None,
        learned_context: Dict[str,str] | None = None,
    ) -> str:
        # Return nothing so that LLM is forced to use its own output. This dummy tool exists as a
        # "catch all" for things that the LLM should know but would otherwise try to use the real
        # web search tool for.
        return ""

    @staticmethod
    def _handle_photo_tool(
        query: str,
        vision: Vision,
        web_search: WebSearch,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        google_reverse_image_search: bool = False,
        translate: bool = False,
        image_bytes: bytes | None = None,
        local_time: str | None = None,
        location: str | None = None,
        learned_context: Dict[str,str] | None = None
    ) -> str | WebSearchResult:
        extra_context = "\n\n" + GPTAssistant._create_context_system_message(local_time=local_time, location=location, learned_context=learned_context)

        # If no image bytes (glasses always send image but web playgrounds do not), return an error
        # message for the assistant to use
        if image_bytes is None or len(image_bytes) == 0:
            # Because this is a tool response, using "tell user" seems to ensure that the final 
            # assistant response is what we want
            return "Error: no photo supplied. Tell user: I think you're referring to something you can see. Can you provide a photo?"

        # Reverse image search? Use vision tool -> search query, then search.
        # Translation special case: never use reverse image search for it. 
        # NOTE: We do not pass history for now but maybe we should in some cases?
        if google_reverse_image_search and not translate:
            capabilities_used.append(Capability.REVERSE_IMAGE_SEARCH)
            system_prompt = VISION_GENERATE_REVERSE_IMAGE_SEARCH_QUERY_FROM_PHOTO_SYSTEM_MESSAGE + extra_context
            vision_response = vision.query_image(
                system_message=system_prompt,
                query=query,
                image_bytes=image_bytes,
                token_usage_by_model=token_usage_by_model
            )
            return web_search.search_web(query=vision_response.strip("\""), use_photo=True, image_bytes=image_bytes, location=location)
        
        # Just use vision tool
        capabilities_used.append(Capability.VISION)
        system_prompt = VISION_PHOTO_DESCRIPTION_SYSTEM_MESSAGE + extra_context
        response = vision.query_image(
            system_message=system_prompt,
            query=query,
            image_bytes=image_bytes,
            token_usage_by_model=token_usage_by_model
        )
        print(f"vision: {response}")
        return response

    @staticmethod
    def _prune_history(message_history: List[Message]) -> List[Message]:
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
    def _create_debug_tool_info_object(function_name: str, function_args: Dict[str, Any], tool_time: float, search_result: str | None = None):
        function_args = function_args.copy()
        
        # Sanitize bytes, which are often too long to print
        for arg_name, value in function_args.items():
            if isinstance(value, bytes):
                function_args[arg_name] = "<bytes>"
            if isinstance(value, list):
                function_args[arg_name] = ", ".join(function_args[arg_name])
        if "vision" in function_args:
            del function_args["vision"]
        if "web_search" in function_args:
            del function_args["web_search"]
        if "token_usage_by_model" in function_args:
            del function_args["token_usage_by_model"]
        if "prompt" in function_args:
            del function_args["prompt"]
        to_return = {
            "tool": function_name,
            "tool_args": function_args,
            "tool_time": tool_time
        }
        if search_result:
            to_return["search_result"] = search_result
        return to_return

    @staticmethod
    def _create_hallucinated_tool_info_object(function_name: str) -> Dict[str, str]:
        return { "tool": function_name, "hallucinated": "true" }
    
    def _extract_learned_context(self, message_history: List[Message], token_usage: TokenUsage) -> Dict[str,str]:
        # Grab last N user messages
        max_user_history = 2
        messages: List[Message] = []
        for i in range(len(message_history) - 1, -1, -1):
            if len(messages) >= max_user_history:
                break
            if message_history[i].role == Role.USER:
                messages.append(message_history[i])
        
        # Insert system message and reverse so that it is in the right order
        messages.append(Message(role=Role.SYSTEM, content=LEARNED_CONTEXT_EXTRACTION_SYSTEM_MESSAGE))
        messages.reverse()
        # print("Context extraction input:")
        # print(messages)

        # Process
        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages
        )

        # Do not forget to count tokens used!
        token_usage.input += response.usage.prompt_tokens
        token_usage.output += response.usage.completion_tokens
        token_usage.total += response.usage.total_tokens

        # # Debug: print raw output of context extraction
        # print("Learned context:")
        # print(response.choices[0].message.content)

        # Parse it into a dictionary
        learned_context: Dict[str,str] = {}
        lines = response.choices[0].message.content.splitlines()
        for line in lines:
            parts = line.split("=")
            if len(parts) == 2:
                key, value = parts
                if key in LEARNED_CONTEXT_KEY_DESCRIPTIONS:
                    learned_context[key] = value
        return learned_context

    @staticmethod
    def _create_context_system_message(local_time: str | None, location: str | None, learned_context: Dict[str,str] | None) -> str:
        # Fixed context: things we know and need not extract from user conversation history
        context: Dict[str, str] = {}
        if local_time is not None:
            context["current_time"] = local_time
        if location is not None:
            context["location"] = location

        # Merge in learned context
        if learned_context is not None:
            context.update(learned_context)

        # Convert to a list to be appended to a system message or treated as a new system message
        system_message_fragment = CONTEXT_SYSTEM_MESSAGE_PREFIX + "\n".join([ f"<{key}>{value}</{key}>" for key, value in context.items() if value is not None ])
        return system_message_fragment

Assistant.register(GPTAssistant)