#
# assistant.py
#
# Assistant class and response object.
#

import asyncio
import base64
from dataclasses import dataclass
from enum import Enum
import json
import timeit
from typing import Any, Dict, List, Optional

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from .claude_vision import vision_query_claude
from .gpt_vision import vision_query_gpt
from .response import AssistantResponse
from .web_search_tool import WebSearchTool
from generate_image.replicate import ReplicateGenerateImage
from models import Capability, Message, Role, TokenUsage, accumulate_token_usage
from util import detect_media_type


####################################################################################################
# GPT Configuration
####################################################################################################

MODEL = "gpt-4o"

# MODEL = "gpt-4o-mini"
####################################################################################################
# Prompts
####################################################################################################

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo.
only reply in English. Don’t include markdown or emojis.
Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


####################################################################################################
# Tools
####################################################################################################

SEARCH_TOOL_NAME = "web_search"
VISION_TOOL_NAME = "analyze_photo"
QUERY_PARAM_NAME = "query"

IMAGE_GENERATION_TOOL_NAME = "generate_image"
IMAGE_GENERATION_PARAM_NAME = "description"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": """Up-to-date information on news, retail products, current events, local conditions, and esoteric knowledge""",
            "parameters": {
                "type": "object",
                "properties": {
                    QUERY_PARAM_NAME: {
                        "type": "string",
                        "description": "search query",
                    },
                },
                "required": [ QUERY_PARAM_NAME ]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": VISION_TOOL_NAME,
            "description": """Analyzes or describes the photo you have from the user's current perspective.
Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.""",
            "parameters": {
                "type": "object",
                "properties": {
                    QUERY_PARAM_NAME: {
                        "type": "string",
                        "description": "User's query to answer expressed as a command that NEVER refers to the photo or image itself"
                    },
                },
                "required": [ QUERY_PARAM_NAME ]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": IMAGE_GENERATION_TOOL_NAME,
            "description": """Generates an image based on a description or prompt.""",
            "parameters": {
                "type": "object",
                "properties": {
                    IMAGE_GENERATION_PARAM_NAME: {
                        "type": "string",
                        "description": "description of the image to generate"
                    },
                },
                "required": [ IMAGE_GENERATION_PARAM_NAME ]
            },
        },
    }
] 

tool_function_names = [ tool["function"]["name"] for tool in TOOLS ]

@dataclass
class ToolOutput:
    text: str
    safe_for_final_response: bool # whether this can be output directly to user as a response (no second LLM call required)
    image_base64: Optional[str] = None

####################################################################################################
# Assistant
####################################################################################################

class AssistantVisionTool(str, Enum):
    GPT4O = "gpt-4o"
    HAIKU = "haiku"

class Assistant:
    def __init__(
        self,
        openai_client: openai.AsyncOpenAI,
        anthropic_client: anthropic.AsyncAnthropic,
        perplexity_api_key: str,
        vision_tool: AssistantVisionTool
    ):
        self._client = openai_client
        self._anthropic_client = anthropic_client
        self._vision_tool = vision_tool
        self._web_tool = WebSearchTool(api_key=perplexity_api_key)
    
    async def send_to_assistant(
        self,
        prompt: str,
        flavor_prompt: str | None,
        image_bytes: bytes | None,
        message_history: List[Message] | None,
        location_address: str | None,
        local_time: str | None
    ):
        """
        Sends a message from user to assistant.

        Parameters
        ----------
        prompt : str
            User message.
        flavor_prompt : str | None
            Optional flavor prompt to append to system messages to give the assistant personality or
            allow for user customization.
        image_bytes : bytes | None
            Image of what user is looking at.
        message_history : List[Mesage] | None
            Conversation history, excluding current user message we will run inference on.
        location_address : str | None
            User's current location, specified as a full or partial address. This provides context
            to the LLM and is especially useful for web searches. E.g.,
            "3985 Stevens Creek Blvd, Santa Clara, CA 95051".
        local_time : str | None
            User's local time in a human-readable format, which helps the LLM answer questions where
            the user indirectly references the date or time. E.g.,
            "Saturday, March 30, 2024, 1:21 PM".

        Returns
        ------
        AssistantResponse
            Assistant response (text and some required analytics).
        """
        t_start = timeit.default_timer()

        token_usage_by_model: Dict[str, TokenUsage] = {}
        speculative_token_usage_by_model: Dict[str, TokenUsage] = {}
        capabilities_used: List[Capability] = []
        speculative_capabilities_used: List[Capability] = []
        timings: Dict[str, float] = {}

        message_history = message_history if message_history else []
        truncated_message_history = self._truncate_message_history(message_history=message_history)
        messages = self._insert_system_message(messages=truncated_message_history, system_message=SYSTEM_MESSAGE)
        messages = self._insert_user_message(messages=messages, user_prompt=prompt)
        messages = self._inject_extra_context(
            messages=messages,
            flavor_prompt=flavor_prompt,
            location_address=location_address,
            local_time=local_time
        )

        task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]] = {}
        self._create_speculative_tool_calls(
            task_by_tool_name=task_by_tool_name,
            token_usage_by_model=speculative_token_usage_by_model,
            capabilities_used=speculative_capabilities_used,
            timings=timings,
            query=prompt,
            flavor_prompt=flavor_prompt,
            message_history=message_history,
            location_address=location_address,
            local_time=local_time
        )

        initial_response = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=initial_response.usage, model=MODEL)
        initial_response_message = initial_response.choices[0].message
        t_initial = timeit.default_timer()
        timings["initial"] = t_initial - t_start

        # Output
        output_task: Optional[asyncio.Task[ToolOutput]] = None
        using_speculative_output_task = False

        # Determine whether we have tool calls to handle
        tool_calls: List[ChatCompletionMessageToolCall] = [] if initial_response_message.tool_calls is None else initial_response_message.tool_calls
        image_generation_description = self._get_image_generation_description(tool_calls=tool_calls, user_prompt=prompt)
        if len(tool_calls) == 0:
            # No tools: return initial assistant response directly
            capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)
            output_task = self._return_output(output=ToolOutput(text=initial_response_message.content, safe_for_final_response=True))
        elif image_generation_description is not None:
            # Special case: image generation
            tool_output = await self._handle_image_generation_tool(
                token_usage_by_model=token_usage_by_model,
                capabilities_used=capabilities_used,
                timings=timings,
                description=image_generation_description,
                flavor_prompt=flavor_prompt,
                message_history=message_history,
                image_bytes=image_bytes,
                location_address=location_address,
                local_time=local_time
            )
            output_task = self._return_output(output=tool_output)
        else:
            # Tool calls requested. Determine where to output final response from: one of the
            # speculative tools or a final LLM response
            print(f"Tools requested: {[ tool_call.function.name for tool_call in tool_calls ]}")
            output_task = self._get_direct_output_task(tool_calls=tool_calls, speculative_task_by_tool_name=task_by_tool_name)
            if output_task is not None:
                # Speculative task will supply output
                using_speculative_output_task = True
            else:
                # Cannot use a speculative tool, need to perform tool calls and create an output
                # task
                messages.append(initial_response_message)
                tool_tasks = await self._handle_tools(
                    token_usage_by_model=token_usage_by_model,
                    capabilities_used=capabilities_used,
                    timings=timings,
                    task_by_tool_name=task_by_tool_name,
                    tool_calls=tool_calls,
                    flavor_prompt=flavor_prompt,
                    message_history=message_history,
                    image_bytes=image_bytes,
                    location_address=location_address,
                    local_time=local_time
                )

                if len(tool_tasks) != 1:
                    # Multiple tools were called, need to wait for them to complete and then invoke
                    # LLM a second time for a coherent output
                    output_task = await self._complete_tool_response(
                        token_usage_by_model=token_usage_by_model,
                        timings=timings,
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_outputs=await asyncio.gather(*tool_tasks)
                    )
                else:
                    # Single tool: attempt to output it directly, if its output needs no further
                    # processing
                    tool_output = await tool_tasks[0]
                    if tool_output.safe_for_final_response:
                        print(f"Directly outputting tool response: {tool_calls[0].function.name}")
                        output_task = self._return_output(output=tool_output)
                    else:
                        # Tool returned an output that needs to be processed
                        output_task = await self._complete_tool_response(
                            token_usage_by_model=token_usage_by_model,
                            timings=timings,
                            messages=messages,
                            tool_calls=tool_calls,
                            tool_outputs=[ tool_output ]
                        )
            
        # Final response
        output = await asyncio.wait_for(output_task, timeout=60)    # will throw on timeout, killing request
        if using_speculative_output_task:
            # Make sure to output correct capabilities and token usage from speculative task
            capabilities_used = speculative_capabilities_used
            accumulate_token_usage(token_usage_by_model=token_usage_by_model, other=speculative_token_usage_by_model, model=MODEL)
        assert output.safe_for_final_response                       # final output must be safe for direct response
        self._cancel_tasks(task_by_tool_name)                       # ensure any remaining speculative tasks are killed
        timings["total"] = timeit.default_timer() - t_start
        return AssistantResponse(
            token_usage_by_model=token_usage_by_model,
            capabilities_used=capabilities_used,
            response=output.text,
            timings=timings,
            image="" if output.image_base64 is None else output.image_base64
        )
        
    def _create_speculative_tool_calls(
        self,
        task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]],
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        query: str,
        flavor_prompt: str,
        message_history: List[Message],
        location_address: str | None,
        local_time: str | None
    ):
        # Always kick off a web search
        tool_call = ChatCompletionMessageToolCall(
            id="speculative_web_search_tool",
            function=Function(
                arguments=json.dumps({ "query": query }),
                name=SEARCH_TOOL_NAME
            ),
            type="function"
        )
        task_by_tool_name[SEARCH_TOOL_NAME] = self._create_tool_call(
            token_usage_by_model=token_usage_by_model,
            capabilities_used=capabilities_used,
            timings=timings,
            tool_call=tool_call,
            flavor_prompt=flavor_prompt,
            message_history=message_history,
            image_bytes=None,
            location_address=location_address,
            local_time=local_time
        )

    @staticmethod
    def _get_direct_output_task(tool_calls: List[ChatCompletionMessageToolCall], speculative_task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]]) -> Optional[asyncio.Task[ToolOutput]]:
        """
        Identifies the speculative tool task, if any, that can be used to answer the user's query
        without performing further tool calls or invoking the LLM again. 

        This only works for speculative tasks, which are given the full user query and cannot be
        used on actual tool tasks, which are given new queries. This method assumes that multiple
        tool requests for the same tool can be handled by the single speculative tool task of the
        corresponding tool (e.g., if multiple web search tool calls are requested, the single
        speculative web tool task can be used to answer the user).

        Parameters
        ----------
        tool_calls : List[ChatCompletionMessageToolCall]
            Tool call requests.
        speculative_task_by_tool_name : Dict[str, asyncio.Task[ToolOutput]]
            Speculative tasks already underway.

        Returns
        -------
        Optional[asyncio.Task[ToolOutput]]
            The task to use for output or None if speculative tasks cannot be used.
        """
        tool_names = set([ tool_call.function.name for tool_call in tool_calls ])
        if len(tool_names) == 1 and tool_calls[0].function.name in speculative_task_by_tool_name:
            print(f"Using speculative tool: {tool_calls[0].function.name}")
            return speculative_task_by_tool_name[tool_calls[0].function.name]
        return None

    async def _complete_tool_response(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        timings: Dict[str, float],
        messages: List[Message],
        tool_calls: List[ChatCompletionMessageToolCall],
        tool_outputs: List[ToolOutput]
    ) -> asyncio.Task[ToolOutput]:
        assert len(tool_calls) == len(tool_outputs)

        # Tool responses -> messages
        for i in range(len(tool_outputs)):
            messages.append(
                {
                    "tool_call_id": tool_calls[i].id,
                    "role": "tool",
                    "name": tool_calls[i].function.name,
                    "content": tool_outputs[i].text
                }
            )

        # Create task to make second call to LLM (with tool responses)
        return asyncio.create_task(self._handle_tool_completion(token_usage_by_model=token_usage_by_model, timings=timings, messages=messages))
    
    async def _handle_tool_completion(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        timings: Dict[str, float],
        messages: List[Message]
    ) -> ToolOutput:
        # Second (and final) LLM call
        t1 = timeit.default_timer()
        response = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=response.usage, model=MODEL)
        timings["final"] = timeit.default_timer() - t1
        return ToolOutput(text=response.choices[0].message.content, safe_for_final_response=True)
        
    async def _handle_tools(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]],
        tool_calls: List[ChatCompletionMessageToolCall],
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> List[asyncio.Task[ToolOutput]]:
        """
        The created tasks will mutate oken_usage_by_model, capabilities_used, and timings.
        """
        # Tool names in the order that function calling requested them
        requested_tool_names = [ tool_call.function.name for tool_call in tool_calls ]

        # Cancel the speculative tool calls we don't need any more
        #TODO: reuse some of these if it makes sense to!
        for tool_name in list(task_by_tool_name.keys()):
            if tool_name not in requested_tool_names:
                self._cancel_task(task_by_tool_name=task_by_tool_name, tool_name=tool_name)
        
        # Create additional tasks for any other tools requested
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name not in task_by_tool_name:
                task_by_tool_name[tool_name] = self._create_tool_call(
                    token_usage_by_model=token_usage_by_model,
                    capabilities_used=capabilities_used,
                    timings=timings,
                    tool_call=tool_call,
                    flavor_prompt=flavor_prompt,
                    message_history=message_history,
                    image_bytes=image_bytes,
                    location_address=location_address,
                    local_time=local_time
                )

        # Return all the tasks in the order that function calling requested them
        return [ task_by_tool_name[tool_name] for tool_name in requested_tool_names ]

    def _create_tool_call(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        tool_call: ChatCompletionMessageToolCall,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> asyncio.Task[ToolOutput]:
        tool_functions_by_name = {
            SEARCH_TOOL_NAME: self._handle_web_search_tool,
            VISION_TOOL_NAME: self._handle_vision_tool
        }
        # Validate tool
        if tool_call.function.name not in tool_functions_by_name:
            print(f"Error: Hallucinated tool: {tool_call.function.name}")
            return asyncio.create_task(self._return_tool_error_message(message="Error: you hallucinated a tool that doesn't exist. Tell user you had trouble interpreting the request and ask them to rephrase it."))
        args: Dict[str, Any] | None = self._validate_tool_args(tool_call=tool_call)
        if args is None:
            return asyncio.create_task(self._return_tool_error_message(message="Error: you failed to use a required parameter. Tell user you had trouble interpreting the request and ask them to rephrase it."))

        # Fill in common parameters to all tools
        args["token_usage_by_model"] = token_usage_by_model
        args["capabilities_used"] = capabilities_used
        args["timings"] = timings
        args["flavor_prompt"] = flavor_prompt
        args["message_history"] = message_history
        args["image_bytes"] = image_bytes
        args["location_address"] = location_address
        args["local_time"] = local_time

        # Create tool task
        tool_name = tool_call.function.name
        tool_function = tool_functions_by_name[tool_name]
        print(f"Created tool task: name={tool_name}, args={tool_call.function.arguments}")
        return asyncio.create_task(tool_function(**args))
    
    @staticmethod
    async def _return_tool_error_message(message: str) -> ToolOutput:
        # These tool error messages are not intended for user consumption and must be processed by
        # a second LLM call
        return ToolOutput(text=message, safe_for_final_response=False) 
    
    @staticmethod
    async def _return_output(output: ToolOutput) -> ToolOutput:
        return output
    
    async def _handle_vision_tool(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        query: str,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> ToolOutput:
        t_start = timeit.default_timer()

        image_base64: Optional[str] = None
        media_type: Optional[str] = None
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = detect_media_type(image_bytes=image_bytes)

        if self._vision_tool == AssistantVisionTool.GPT4O:
            output = await vision_query_gpt(
                client=self._client,
                token_usage_by_model=token_usage_by_model,
                query=query,
                image_base64=image_base64,
                media_type=media_type
            )
        else:
            output = await vision_query_claude(
                client=self._anthropic_client,
                token_usage_by_model=token_usage_by_model,
                query=query,
                image_base64=image_base64,
                media_type=media_type
            )

        t_end = timeit.default_timer()
        timings["vision_tool"] = t_end - t_start
        capabilities_used.append(Capability.VISION)

        if output.web_query is None or output.web_query == "":
            # We never allow vision tool to be final response because it does not have sufficient
            # context to answer on its own
            return ToolOutput(text=output.response, safe_for_final_response=False)    
        
        # Perform web search and produce a synthesized response telling assistant where each piece of
        # information came from. Web search will lack important vision information. We need to return
        # both and have the assistant figure out which info to use.
        web_result = await self._handle_web_search_tool(
            token_usage_by_model=token_usage_by_model,
            capabilities_used=capabilities_used,
            timings=timings,
            query=output.web_query,
            flavor_prompt=flavor_prompt,
            message_history=message_history,
            image_bytes=None,
            location_address=location_address,
            local_time=local_time
        )
        return ToolOutput(text=f"HERE IS WHAT YOU SEE: {output.response}\nEXTRA INFO FROM WEB: {web_result}", safe_for_final_response=False)
    
    async def _handle_web_search_tool(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        query: str,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> ToolOutput:
        t_start = timeit.default_timer()
        output = await self._web_tool.search_web(
            token_usage_by_model=token_usage_by_model,
            timings=timings,
            query=query,
            flavor_prompt=flavor_prompt,
            message_history=message_history,
            location=location_address
        )
        t_end = timeit.default_timer()
        timings["web_search_tool"] = t_end - t_start
        capabilities_used.append(Capability.WEB_SEARCH)
        return ToolOutput(text=output, safe_for_final_response=True)
    
    async def _handle_image_generation_tool(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        capabilities_used: List[Capability],
        timings: Dict[str, float],
        description: str,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> ToolOutput:
        if image_bytes is None or len(image_bytes) == 0:
            return ToolOutput(text="I need a photo in order to generate an image.", safe_for_final_response=True)
        t_start = timeit.default_timer()
        image_generator = ReplicateGenerateImage()
        image_base64 = await image_generator.generate_image(query=description, use_image=True, image_bytes=image_bytes)
        capabilities_used.append(Capability.IMAGE_GENERATION)
        timings["image_generation"] = timeit.default_timer() - t_start
        return ToolOutput(text="Here is the image you requested.", safe_for_final_response=True, image_base64=image_base64)
    
    @staticmethod
    def _get_image_generation_description(tool_calls: List[ChatCompletionMessageToolCall], user_prompt: str) -> Optional[str]:
        for tool_call in tool_calls:
            if tool_call.function.name == IMAGE_GENERATION_TOOL_NAME:
                try:
                    args = json.loads(tool_call.function.arguments)
                    return args[IMAGE_GENERATION_PARAM_NAME]
                except:
                    # Function calling failed to produce the required parameter, just use user
                    # prompt as a fallback
                    return user_prompt
        return None

    @staticmethod
    def _validate_tool_args(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
        # Parse arguments and ensure they are all str or bool for now. Drop any that aren't.
        function_description = [ description for description in TOOLS if description["function"]["name"] == tool_call.function.name ][0]
        function_parameters = function_description["function"]["parameters"]["properties"]
        args: Dict[str, Any] = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except:
            pass
        for param_name in list(args.keys()):
            if param_name not in function_parameters:
                # GPT hallucinated a parameter
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
        
        # Ensure all required params are present
        for param_name in function_parameters:
            if param_name not in args:
                return None
            
        return args

    @staticmethod
    def _cancel_tasks(task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]]):
        for tool_name, task in list(task_by_tool_name.items()):
            task.cancel()
            del task_by_tool_name[tool_name]
   
    @staticmethod
    def _cancel_task(task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]], tool_name: str):
        task = task_by_tool_name.get(tool_name)
        if task is not None:
            task.cancel()
            del task_by_tool_name[tool_name]

    @staticmethod
    def _insert_system_message(messages: List[Message], system_message: str): 
        messages = messages.copy()
        system_message = Message(role=Role.SYSTEM, content=system_message)
        if len(messages) == 0:
            messages = [ system_message ]
        else:
            # Insert system message before message history, unless client transmitted one they want
            # to use
            if len(messages) > 0 and messages[0].role != Role.SYSTEM:
                messages.insert(0, system_message)
        return messages
    
    @staticmethod
    def _insert_user_message(messages: List[Message], user_prompt: str):
        messages = messages.copy()
        user_message = Message(role=Role.USER, content=user_prompt)
        messages.append(user_message)
        return messages
    
    @staticmethod
    def _truncate_message_history(message_history: List[Message]):
        """
        Prunes down the chat history to save tokens, improving inference speed and reducing cost.
        Generally, preserving all assistant responses is not needed, and only a limited number of
        user messages suffice to maintain a coherent conversation.

        Parameters
        ----------
        message_history : List[Message]
            Conversation history. This list will be not be mutated.
        
        Returns
        -------
        List[Message]
            A new, truncated list of messages.
        """
        # Limit to most recent 5 user messages and 3 assistant responses
        message_history = message_history.copy()
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
    def _inject_extra_context(
        messages: List[Message],
        flavor_prompt: str | None,
        location_address: str | None,
        local_time: str | None
    ):
        # Fixed context: things we know and need not extract from user conversation history
        context: Dict[str, str] = {}
        if local_time is not None and len(local_time) > 0:
            context["current_time"] = local_time
        else:
            context["current_time"] = "If asked, tell user you don't know current date or time because clock is broken"
        if location_address is not None and len(location_address) > 0:
            context["location"] = location_address
        else:
            context["location"] = "You do not know user's location and if asked, tell them so"

        # Unclear whether multiple system messages are confusing to the assistant or not but cursory
        # testing shows this seems to work.
        extra_context = CONTEXT_SYSTEM_MESSAGE_PREFIX + "\n".join([ f"<{key}>{value}</{key}>" for key, value in context.items() if value is not None ])
        if flavor_prompt is not None:
            extra_context = f"{flavor_prompt}\n{extra_context}"
        extra_context_message = Message(role=Role.SYSTEM, content=extra_context)

        messages = messages.copy()
        messages.append(extra_context_message)
        return messages