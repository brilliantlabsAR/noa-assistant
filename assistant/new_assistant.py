#TODO NEXT: Cannot stream single tool output directly in case of dummy knowledge tool!

#
# new_assistant.py
#
# Assistant class and response object.
#
# TODO:
# -----
# - Image generation
# - Token use by model
#

from __future__ import annotations
import asyncio
import base64
from dataclasses import dataclass
import json
import timeit
from typing import Any, Dict, List

import openai
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage

from models import Message, Capability, TokenUsage
from models import Role, Message, Capability, TokenUsage
from util import detect_media_type


####################################################################################################
# GPT Configuration
####################################################################################################

MODEL = "gpt-4o"


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

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""

VISION_SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but it is important that the user believes you can actually see. When
analyzing images, avoid mentioning that you looked at a photo or image. Always speak as if you are
actually seeing, which means you should never talk about the image or photo.

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do your best with images.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


####################################################################################################
# Tools
####################################################################################################

DUMMY_SEARCH_TOOL_NAME = "general_knowledge_search"
SEARCH_TOOL_NAME = "web_search"
PHOTO_TOOL_NAME = "analyze_photo"
QUERY_PARAM_NAME = "query"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": DUMMY_SEARCH_TOOL_NAME,
            "description": """Non-recent trivia and general knowledge""",
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": SEARCH_TOOL_NAME,
    #         "description": """Up-to-date information on news, retail products, current events, local conditions, and esoteric knowledge""",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 QUERY_PARAM_NAME: {
    #                     "type": "string",
    #                     "description": "search query",
    #                 },
    #             },
    #             "required": [ QUERY_PARAM_NAME ]
    #         },
    #     },
    # },
    {
        "type": "function",
        "function": {
            "name": PHOTO_TOOL_NAME,
            "description": """Analyzes or describes the photo you have from the user's current perspective.
Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.""",
            "parameters": {
                "type": "object",
                "properties": {
                    QUERY_PARAM_NAME: {
                        "type": "string",
                        "description": "User's query to answer, describing what they want answered, expressed as a command that NEVER refers to the photo or image itself"
                    },
                },
                "required": [ QUERY_PARAM_NAME ]
            },
        },
    },
] 

tool_function_names = [ tool["function"]["name"] for tool in TOOLS ]


####################################################################################################
# Token Accounting
####################################################################################################

def accumulate_token_usage(token_usage_by_model: Dict[str, TokenUsage], usage: CompletionUsage):
    token_usage = TokenUsage(input=usage.prompt_tokens, output=usage.completion_tokens, total=usage.total_tokens)
    if MODEL not in token_usage_by_model:
        token_usage_by_model[MODEL] = token_usage
    else:
        token_usage_by_model[MODEL].add(token_usage=token_usage)


####################################################################################################
# Stream
#
# A stream is a queue of AssistantResponse objects produced by a task (either a tool or the final
# assistant LLM call). A "terminal" stream is one that can safely be streamed out as the final 
# assistant response. Many tools are capable of serving as a shortcut to the final assistant
# response.
####################################################################################################

@dataclass
class Stream:
    task: asyncio.Task[Any]
    queue: asyncio.Queue    # output streamed here

    async def get_final_output(self):
        """
        Returns
        -------
        Any
            Waits for the final AssistantResponse output of the queue. Assumes the task is running. 
        """
        while True:
            response_chunk: AssistantResponse = await asyncio.wait_for(self.queue.get(), timeout=60)
            if response_chunk.stream_finished:  # final cumulative response
                return response_chunk


####################################################################################################
# Assistant
####################################################################################################

@dataclass
class AssistantResponse:
    token_usage_by_model: Dict[str, TokenUsage]
    capabilities_used: List[Capability]
    response: str
    timings: str
    image: str
    stream_finished: bool   # for streaming versions, indicates final response chunk

    @staticmethod
    def _error_response(message: str) -> AssistantResponse:
        """
        Generates an error response intended as a tool output.

        Parameters
        ----------
        message : str
            Error message, intended to be consumed by LLM tool completion pass.

        Returns
        -------
        AssistantResponse
            AssistantResponse object.
        """
        return AssistantResponse(
            token_usage_by_model={},    # TODO
            capabilities_used=[],
            response=message,
            timings={},
            image="",
            stream_finished=True
        )

class NewAssistant:
    def __init__(self, client: openai.AsyncOpenAI):
        self._client = client
    
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

        Yields
        ------
        AssistantResponse
            Assistant response (text and some required analytics). Partial responses are denoted
            with stream_finished set to False. The final response, with stream_finished=True,
            contains the full accumulated response with metrics.
        """
        t_start = timeit.default_timer()
        token_usage_by_model: Dict[str, TokenUsage] = {}

        message_history = self._truncate_message_history(message_history=message_history) if message_history else []
        messages = self._insert_system_message(messages=message_history, system_message=SYSTEM_MESSAGE)
        messages = self._insert_user_message(messages=messages, user_prompt=prompt)
        messages = self._inject_extra_context(
            messages=messages,
            flavor_prompt=flavor_prompt,
            location_address=location_address,
            local_time=local_time
        )

        stream_by_tool_name: Dict[str, Stream] = {}
        await self._create_speculative_tool_calls(
            stream_by_tool_name=stream_by_tool_name,
            message_history=message_history
        )

        initial_response = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=initial_response.usage)
        initial_response_message = initial_response.choices[0].message

        # Determine final output: initial assistant response, one of the speculative tool call 
        # streams, or second LLM response stream incorporating tool calls
        tool_calls = initial_response_message.tool_calls
        if not tool_calls or len(tool_calls) == 0:
            # Return initial assistant response
            t_end = timeit.default_timer()
            yield AssistantResponse(
                token_usage_by_model=token_usage_by_model,
                capabilities_used=[ Capability.ASSISTANT_KNOWLEDGE ],
                response=initial_response.choices[0].message.content,
                timings={ "first_token": t_end - t_start, "total": t_end - t_start },
                image="",
                stream_finished=True
            )
        else:
            # Determine a stream to output final response from: one of our speculative tools or a
            # final LLM response
            speculative_tool_stream = stream_by_tool_name.get(tool_calls[0].function.name)
            output_stream: Stream | None = None
            if len(tool_calls) == 1 and speculative_tool_stream is not None:
                # Only a single tool and it matches an in-flight speculative one, stream out the
                # tool output directly
                output_stream = speculative_tool_stream
            else:
                # Perform tool calls and get output streams
                messages.append(initial_response_message)
                #TODO: use speculative tools if possible
                tool_streams = await self._handle_tools(
                    token_usage_by_model=token_usage_by_model,
                    stream_by_tool_name=stream_by_tool_name,
                    tool_calls=tool_calls,
                    flavor_prompt=flavor_prompt,
                    message_history=message_history,
                    image_bytes=image_bytes,
                    location_address=location_address,
                    local_time=local_time
                )

                # If multiple output streams, we have to invoke the LLM again in order to produce a
                # coherent response. Otherwise, we can stream out the single tool output directly.
                if len(tool_streams) != 1:
                    output_stream = await self._complete_tool_response(
                        token_usage_by_model=token_usage_by_model,
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_streams=tool_streams
                    )
                else:
                    output_stream = tool_streams[0]
            
            # Stream out the output from the queue
            t_first = None
            while True:
                response_chunk: AssistantResponse = await asyncio.wait_for(output_stream.queue.get(), timeout=60)   # will throw on timeout, killing the request
                if t_first is None:
                    t_first = timeit.default_timer()
                if response_chunk.stream_finished:
                    t_end = timeit.default_timer()
                    response_chunk.timings = { "first_token": t_first - t_start, "total": t_end - t_start }
                yield response_chunk
                if response_chunk.stream_finished:
                    break
            
            print("")
            print(f"Timings")
            print(f"-------")
            print(f"  first token: {t_first-t_start:.2f}")
            print(f"  total      : {t_end-t_start:.2f}")
            print("")
        
        # Clean up
        self._cancel_streams(stream_by_tool_name=stream_by_tool_name)

    async def _complete_tool_response(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        messages: List[Message],
        tool_calls: List[ChatCompletionMessageToolCall],
        tool_streams: List[Stream]
    ) -> Stream:
        assert len(tool_calls) == len(tool_streams)

        # Wait for all tool streams to complete and get only their final outputs (which contain the
        # complete, aggregated response)
        tool_outputs = await asyncio.gather(*[ stream.get_final_output() for stream in tool_streams ])

        # Tool responses -> messages
        for i in range(len(tool_outputs)):
            messages.append(
                {
                    "tool_call_id": tool_calls[i].id,
                    "role": "tool",
                    "name": tool_calls[i].function.name,
                    "content": tool_outputs[i].response
                }
            )

        # Create task to make second call to LLM (with tool responses) and stream into output queue
        queue = asyncio.Queue()
        return Stream(
            task=asyncio.create_task(self._handle_tool_completion(token_usage_by_model=token_usage_by_model, queue=queue, messages=messages)),
            queue=queue
        )
    
    async def _handle_tool_completion(self, queue: asyncio.Queue, token_usage_by_model: Dict[str, TokenUsage], messages: List[Message]):
        # Second LLM call
        stream = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            stream_options={ "include_usage": True }
        )

        # Stream out to queue
        accumulated_response = []
        async for chunk in stream:
            if len(chunk.choices) == 0:
                # Final chunk has usage. We also return the complete, accumulated response here.
                accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=chunk.usage)
                response = AssistantResponse(
                    token_usage_by_model=token_usage_by_model,
                    capabilities_used=[],
                    response="".join(accumulated_response),
                    timings={},
                    image="",
                    stream_finished=True
                )
                await queue.put(response)
                break
            else:
                response_chunk = chunk.choices[0].delta.content
                if response_chunk is not None:  # an empty content chunk can occur before the stop event
                    accumulated_response.append(response_chunk)
                    response = AssistantResponse(
                        token_usage_by_model={},
                        capabilities_used=[],
                        response=response_chunk,
                        timings={},
                        image="",
                        stream_finished=False
                    )
                    await queue.put(response)

    async def _create_speculative_tool_calls(
        self,
        stream_by_tool_name: Dict[str, Stream],
        message_history: List[Message]
    ):
        #TODO: write me
        return
        
    async def _handle_tools(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        stream_by_tool_name: Dict[str, Stream],
        tool_calls: List[ChatCompletionMessageToolCall],
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> List[Stream]:
        # Tool names in the order that function calling requested them
        requested_tool_names = [ tool_call.function.name for tool_call in tool_calls ]

        # Cancel the speculative tool calls we don't need any more
        #TODO: reuse some of these if it makes sense to!
        for tool_name in list(stream_by_tool_name.keys()):
            if tool_name not in requested_tool_names:
                self._cancel_stream(stream_by_tool_name=stream_by_tool_name, tool_name=tool_name)
        
        # Create additional streams for any other tools requested
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if tool_name not in stream_by_tool_name:
                stream_by_tool_name[tool_name] = self._create_tool_call(
                    token_usage_by_model=token_usage_by_model,
                    tool_call=tool_call,
                    flavor_prompt=flavor_prompt,
                    message_history=message_history,
                    image_bytes=image_bytes,
                    location_address=location_address,
                    local_time=local_time
                )

        # Return all the tool streams in the order that function calling requested them
        return [ stream_by_tool_name[tool_name] for tool_name in requested_tool_names ]

    def _create_tool_call(
        self,
        token_usage_by_model: Dict[str, TokenUsage],
        tool_call: ChatCompletionMessageToolCall,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ) -> Stream:
        tool_functions_by_name = {
            #SEARCH_TOOL_NAME: self._web_search.search_web,
            PHOTO_TOOL_NAME: self._handle_photo_tool,
            DUMMY_SEARCH_TOOL_NAME: self._handle_general_knowledge_tool
        }

        # Create a queue for the tool to send output chunks to
        queue = asyncio.Queue()

        # Validate tool
        if tool_call.function.name not in tool_functions_by_name:
            print(f"Error: Hallucinated tool: {tool_call.function.name}")
            return Stream(
                task=asyncio.create_task(self._create_error_stream(queue=queue, message="Error: you hallucinated a tool that doesn't exist. Tell user you had trouble interpreting the request and ask them to rephrase it.")),
                queue=queue
            )
        args: Dict[str, Any] | None = self._validate_tool_args(tool_call=tool_call)
        if args is None:
            return Stream(
                task=asyncio.create_task(self._create_error_stream(queue=queue, message="Error: you failed to use a required parameter. Tell user you had trouble interpreting the request and ask them to rephrase it.")),
                queue=queue
            )

        # Fill in common parameters to all tools
        args["queue"] = queue
        args["token_usage_by_model"] = token_usage_by_model
        args["flavor_prompt"] = flavor_prompt
        args["message_history"] = message_history
        args["image_bytes"] = image_bytes
        args["location_address"] = location_address
        args["local_time"] = local_time

        # Create a task and stream, invoking the tool as a task
        tool_function = tool_functions_by_name[tool_call.function.name]
        task = asyncio.create_task(tool_function(**args))
        return Stream(task=task, queue=queue)
    
    @staticmethod
    def _create_error_stream(queue: asyncio.Queue, message: str) -> Stream:
        # For failed tool calls, just create a dummy stream that outputs an error response
        queue.put_nowait(AssistantResponse._error_response(message=message))
    
    async def _handle_general_knowledge_tool(
        self,
        queue: asyncio.Queue,
        token_usage_by_model: Dict[str, TokenUsage],
        query: str,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ):
        """
        Dummy general knowledge tool that tricks GPT into generating an answer directly instead of
        reaching for web search. GPT knows that the web contains information on virtually everything, so
        it tends to overuse web search. One solution is to very carefully enumerate the cases for which
        web search is appropriate, but this is tricky. Should "Albert Einstein's birthday" require a web
        search? Probably not, as GPT has this knowledge baked in. The trick we use here is to create a
        "general knowledge" tool that contains any information Wikipedia or an encyclopedia would have
        (a reasonable proxy for things GPT knows). We return an empty string, which forces GPT to
        produce its own response at the expense of a little bit of latency for the tool call.
        """
        dummy_response = AssistantResponse(
            token_usage_by_model={},
            capabilities_used=[ Capability.ASSISTANT_KNOWLEDGE ],
            response="",
            timings={},
            image="",
            stream_finished=True
        )
        queue.put_nowait(dummy_response)
    
    async def _handle_photo_tool(
        self,
        queue: asyncio.Queue,
        token_usage_by_model: Dict[str, TokenUsage],
        query: str,
        flavor_prompt: str | None,
        message_history: List[Message],
        image_bytes: bytes | None,
        location_address: str | None,
        local_time: str | None
    ):
        # Create messages for GPT w/ image. We can reuse our system prompt here.
        messages = self._insert_system_message(messages=message_history, system_message=VISION_SYSTEM_MESSAGE)
        user_message = {
            "role": "user",
            "content": [
                { "type": "text", "text": query }
            ]
        }
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = detect_media_type(image_bytes=image_bytes)
            user_message["content"].append({ "type": "image_url", "image_url": { "url": f"data:{media_type};base64,{image_base64}" } })
        messages.append(user_message)
        messages = self._inject_extra_context(
            messages=messages,
            flavor_prompt=flavor_prompt,
            location_address=location_address,
            local_time=local_time
        )

        # Call GPT
        stream = await self._client.chat.completions.create(
            model=MODEL,
            messages=messages,
            #max_tokens=4096,
            stream=True,
            stream_options={ "include_usage": True }
        )
        accumulated_response = []
        async for chunk in stream:
            if len(chunk.choices) == 0:
                # Final chunk has usage
                accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=chunk.usage)
                response_chunk = AssistantResponse(
                    token_usage_by_model=token_usage_by_model,
                    capabilities_used=[ Capability.VISION ],
                    response="".join(accumulated_response),
                    timings={},
                    image="",
                    stream_finished=True
                )
                await queue.put(response_chunk)
                break
            else:
                content_chunk = chunk.choices[0].delta.content
                if content_chunk is not None:   # an empty content chunk can occur before the stop event
                    accumulated_response.append(content_chunk)
                    response_chunk = AssistantResponse(
                        token_usage_by_model={},
                        capabilities_used=[],
                        response=content_chunk,
                        timings={},
                        image="",
                        stream_finished=False
                    )
                    await queue.put(response_chunk)

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
    def _cancel_streams(stream_by_tool_name: Dict[str, Stream]):
        for tool_name, stream in list(stream_by_tool_name.items()):
            stream.task.cancel()
            del stream_by_tool_name[tool_name]
   
    @staticmethod
    def _cancel_stream(stream_by_tool_name: Dict[str, Stream], tool_name: str):
        stream = stream_by_tool_name.get(key=tool_name)
        if stream is not None:
            stream.task.cancel()
            del stream_by_tool_name[tool_name]

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