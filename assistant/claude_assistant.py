#
# claude_assistant.py
#
# Assistant implementation based on Anthropic's Claude series of models.
#

#
# TODO:
# -----
# - Factor out functions common to ClaudeAssistant and GPTAssistant.
# - Occasionally Claude returns errors, debug these.
# - Claude sometimes messes up with follow-up questions and then refers to internal context. We may
#   need to try embedding extra context inside of the latest user message.
#

import asyncio
import json
import timeit
from typing import Any, Dict, List

import anthropic
from anthropic.types.beta.tools import ToolParam, ToolUseBlock

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

Make your responses precise and max 5 sentences. Respond without any preamble when giving translations,
just translate directly. When analyzing the user's view, speak as if you can actually see and never
make references to the photo or image you analyzed.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


####################################################################################################
# Tools
####################################################################################################

DUMMY_SEARCH_TOOL_NAME = "general_knowledge_search"
SEARCH_TOOL_NAME = "web_search"
PHOTO_TOOL_NAME = "analyze_photo"
QUERY_PARAM_NAME = "query"

TOOLS: List[ToolParam] = [
    {
        "name": DUMMY_SEARCH_TOOL_NAME,
        "description": "Non-recent trivia and general knowledge",
        "input_schema": {
            "type": "object",
            "properties": {
                QUERY_PARAM_NAME: {
                    "type": "string",
                    "description": "search query",
                }
            },
            "required": [ QUERY_PARAM_NAME ]
        }
    },
    {
        "name": SEARCH_TOOL_NAME,
        "description": "Up-to-date information on news, retail products, current events, local conditions, and esoteric knowledge",
        "input_schema": {
            "type": "object",
            "properties": {
                QUERY_PARAM_NAME: {
                    "type": "string",
                    "description": "search query",
                }
            },
            "required": [ QUERY_PARAM_NAME ]
        }
    },
    {
        "name": PHOTO_TOOL_NAME,
        "description": """Analyzes or describes the photo you have from the user's current perspective.
Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.""",
        "input_schema": {
            "type": "object",
            "properties": {
                QUERY_PARAM_NAME: {
                    "type": "string",
                    "description": "User's query to answer, describing what they want answered, expressed as a command that NEVER refers to the photo or image itself"
                }
            },
            "required": [ QUERY_PARAM_NAME ]
        }
    }
]

async def handle_tool(
    tool_call: ToolUseBlock,
    user_message: str,
    image_bytes: bytes | None,
    location: str | None,
    local_time: str | None,
    web_search: WebSearch,
    vision: Vision,
    learned_context: Dict[str, str] | None,
    token_usage_by_model: Dict[str, TokenUsage],
    capabilities_used: List[Capability],
    tools_used: List[Dict[str, Any]]
) -> str:
    tool_functions = {
        SEARCH_TOOL_NAME: web_search.search_web,                # returns WebSearchResult
        PHOTO_TOOL_NAME: handle_photo_tool,                     # returns WebSearchResult | str
        DUMMY_SEARCH_TOOL_NAME: handle_general_knowledge_tool,  # returns str
    }

    function_name = tool_call.name
    function_to_call = tool_functions.get(function_name)
    if function_to_call is None:
        # Error: Hallucinated a tool
        return "Error: you hallucinated a tool that doesn't exist. Tell user you had trouble interpreting the request and ask them to rephrase it."

    function_args = prepare_tool_arguments(
        tool_call=tool_call,
        user_message=user_message,
        image_bytes=image_bytes,
        location=location,
        local_time=local_time,
        web_search=web_search,
        vision=vision,
        learned_context=learned_context,
        token_usage_by_model=token_usage_by_model,
        capabilities_used=capabilities_used
    )

    tool_start_time = timeit.default_timer()
    function_response: WebSearchResult | str = await function_to_call(**function_args)
    total_tool_time = round(timeit.default_timer() - tool_start_time, 3)

    # Record capability used (except for case of photo tool, which reports on its own because it
    # can invoke multiple capabilities)
    if function_name == SEARCH_TOOL_NAME:
        capabilities_used.append(Capability.WEB_SEARCH)
    elif function_name == DUMMY_SEARCH_TOOL_NAME:
        capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)
    
    tools_used.append(
        create_debug_tool_info_object(
            function_name=function_name,
            function_args=function_args,
            tool_time=total_tool_time,
            search_result=function_response.search_provider_metadata if isinstance(function_response, WebSearchResult) else None
        )
    )

     # Format response appropriately
    assert isinstance(function_response, WebSearchResult) or isinstance(function_response, str)
    tool_output = function_response.summary if isinstance(function_response, WebSearchResult) else function_response
    return tool_output

def prepare_tool_arguments(
    tool_call: ToolUseBlock,
    user_message: str,
    image_bytes: bytes | None,
    location: str | None,
    local_time: str | None,
    web_search: WebSearch,
    vision: Vision,
    learned_context: Dict[str, str] | None,
    token_usage_by_model: Dict[str, TokenUsage],
    capabilities_used: List[Capability]
) -> Dict[str, Any]:
    # Get function description we passed to Claude. This function should be called after we have
    # validated that a valid tool call was generated.
    function_description = [ description for description in TOOLS if description["name"] == tool_call.name ][0]
    function_parameters = function_description["input_schema"]["properties"]

    # Parse arguments and ensure they are all str or bool for now. Drop any that aren't.
    args = tool_call.input.copy()
    for param_name in list(args.keys()):
        if param_name not in function_parameters:
            # Hallucinated parameter
            del args[param_name]
            continue
        if function_parameters[param_name]["type"] == "string" and type(args[param_name]) != str:
            del args[param_name]
            continue
        if function_parameters[param_name]["type"] == "boolean" and type(args[param_name]) != bool:
            del args[param_name]
            continue
        if function_parameters[param_name]["type"] not in [ "string", "boolean" ]:
            # Need to keep this up to date with the tools we define
            raise ValueError(f"Unsupported tool parameter type: {function_parameters[param_name]['type']}")

    # Fill in args required by all tools
    args["location"] = location if location else "unknown"
    args[QUERY_PARAM_NAME] = args[QUERY_PARAM_NAME] if QUERY_PARAM_NAME in args else user_message

    # Photo tool additional parameters we need to inject
    if tool_call.name == PHOTO_TOOL_NAME:
        args["image_bytes"] = image_bytes
        args["vision"] = vision
        args["web_search"] = web_search
        args["local_time"] = local_time
        args["learned_context"] = learned_context
        args["token_usage_by_model"] = token_usage_by_model
        args["capabilities_used"] = capabilities_used

    return args

async def handle_general_knowledge_tool(
    query: str,
    image_bytes: bytes | None = None,
    local_time: str | None = None,
    location: str | None = None,
    learned_context: Dict[str,str] | None = None,
) -> str:
    """
    Dummy general knowledge tool that tricks Claude into generating an answer directly instead of
    reaching for web search.
    """
    return ""

async def handle_photo_tool(
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
    extra_context = "\n\n" + ClaudeAssistant._create_context_system_message(local_time=local_time, location=location, learned_context=learned_context)

    # If no image bytes (glasses always send image but web playgrounds do not), return an error
    # message for the assistant to use
    if image_bytes is None or len(image_bytes) == 0:
        # Because this is a tool response, using "tell user" seems to ensure that the final
        # assistant response is what we want
        return "Error: no photo supplied. Tell user: I think you're referring to something you can see. Can you provide a photo?"

    # Vision tool
    capabilities_used.append(Capability.VISION)
    output = await vision.query_image(
        query=query,
        extra_context=extra_context,
        image_bytes=image_bytes,
        token_usage_by_model=token_usage_by_model
    )
    print(f"Vision: {output}")
    if output is None:
        return "Error: vision tool generated an improperly formatted result. Tell user that there was a temporary glitch and ask them to try again."

    # Do we need to perform a web search?
    if output.web_search_needed():
        capabilities_used.append(Capability.REVERSE_IMAGE_SEARCH if output.reverse_image_search else Capability.WEB_SEARCH)
        return await web_search.search_web(
            query=output.web_query.strip("\""),
            use_photo=output.reverse_image_search,
            image_bytes=image_bytes,
            location=location
        )
    
    # No, just return the vision tool response directly
    return output.response

def create_debug_tool_info_object(function_name: str, function_args: Dict[str, Any], tool_time: float, search_result: str | None = None) -> Dict[str, Any]:
    """
    Produces an object of arbitrary keys and values intended to serve as a debug description of tool
    use.
    """
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
        start = timeit.default_timer()
        model = model if model is not None else "claude-3-sonnet-20240229" #"claude-3-haiku-20240307"

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

        # Initial Claude response -- if no tools, this will be returned as final response
        first_response = await self._client.beta.tools.messages.create(
            model=model,
            system=system_text + "\n\n" + extra_context,
            messages=message_history,
            tools=TOOLS,
            max_tokens=4096
        )

        # Aggregate token counts
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model=model,
            input_tokens=first_response.usage.input_tokens,
            output_tokens=first_response.usage.output_tokens,
            total_tokens=first_response.usage.input_tokens + first_response.usage.output_tokens
        )

        # Handle tools
        tools_used = []
        tools_used.append({ "learned_context": learned_context })   # log context here for now
        if first_response.stop_reason != "tool_use":
            returned_response.response = first_response.content[0].text
        else:
            # Append tool message to history, as per Anthropic's example at https://github.com/anthropics/anthropic-sdk-python/blob/9fad441043ff7bfdf8786b64b1e4bbb27105b112/examples/tools.py
            message_history.append({ "role": first_response.role, "content": first_response.content })

            # Invoke all tool requests in parallel and wait for them to complete
            tool_calls: ToolUseBlock = [ content for content in first_response.content if content.type == "tool_use" ]
            #print(tool_calls)
            tool_handlers = []
            for tool_call in tool_calls:
                tool_handlers.append(
                    handle_tool(
                        tool_call=tool_call,
                        user_message=prompt,
                        image_bytes=image_bytes,
                        location=location_address,
                        local_time=local_time,
                        web_search=web_search,
                        vision=vision,
                        learned_context=learned_context,
                        token_usage_by_model=returned_response.token_usage_by_model,
                        capabilities_used=returned_response.capabilities_used,
                        tools_used=tools_used
                    )
                )
            tool_outputs = await asyncio.gather(*tool_handlers)
            #print(tool_outputs)

            # Submit tool responses
            tool_response_message = {
                "role": "user",
                "content": []
            }
            for i in range(len(tool_outputs)):
                tool_response_message["content"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_calls[i].id,
                        "content": [ { "type": "text", "text": tool_outputs[i] } ]
                    }
                )
            message_history.append(tool_response_message)
            
            # Get final response from model
            second_response = await self._client.beta.tools.messages.create(
                model=model,
                system=system_text + "\n\n" + extra_context,
                messages=message_history,
                tools=TOOLS,
                max_tokens=4096
            )

            # Aggregate tokens and response
            accumulate_token_usage(
                token_usage_by_model=returned_response.token_usage_by_model,
                model=model,
                input_tokens=second_response.usage.input_tokens,
                output_tokens=second_response.usage.output_tokens,
                total_tokens=second_response.usage.input_tokens + second_response.usage.output_tokens
            )
            returned_response.response = second_response.content[0].text

        # If no tools were used, only assistant capability recorded
        if len(returned_response.capabilities_used) == 0:
            returned_response.capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)
        
        # Return final response
        returned_response.debug_tools = json.dumps(tools_used)
        stop = timeit.default_timer()
        print(f"Time taken: {stop-start:.3f}")

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
        how Noa is made aware of the user's location, local time, and any learned information that
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