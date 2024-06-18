#
# gpt_assistant.py
#
# Assistant implementation based on OpenAI's GPT models. This assistant is capable of leveraging
# separate web search and vision tools.
#
# Support also exists for using Groq because it mirrors OpenAI's API.
#

#
# TODO:
# -----
# - Speculative vision tool should create a proper tools_used entry.
# - Move to streaming completions and detect timeouts when a threshold duration elapses since the
#   the last token was emitted.
# - Figure out how to get assistant to stop referring to "photo" and "image" when analyzing photos.
# - Improve people search.
#

import asyncio
import base64
import json
import timeit
from typing import Any, Dict, List

import openai
from openai.types.chat import ChatCompletionMessageToolCall
import groq
# from groq.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from .assistant import Assistant, AssistantResponse
from .context import create_context_system_message
from web_search import WebSearch, WebSearchResult
from vision import Vision, GPT4Vision
from vision.utils import detect_media_type
from models import Role, Message, Capability, TokenUsage, accumulate_token_usage
from generate_image.replicate import ReplicateGenerateImage

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

It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""


####################################################################################################
# Tools
####################################################################################################

DUMMY_SEARCH_TOOL_NAME = "general_knowledge_search"
IMAGE_GENERATION_TOOL_NAME = "generate_image"
SEARCH_TOOL_NAME = "web_search"
PHOTO_TOOL_NAME = "analyze_photo"
IMAGE_GENERATION_PARAM_NAME = "description"
QUERY_PARAM_NAME = "query"

TOOLS = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": DUMMY_SEARCH_TOOL_NAME,
    #         "description": """Non-recent trivia and general knowledge""",
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
            "name": SEARCH_TOOL_NAME,
            "description": """Up-to-date information on news, retail products, current events, local conditions, and esoteric knowledge""",
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         QUERY_PARAM_NAME: {
            #             "type": "string",
            #             "description": "search query",
            #         },
            #     },
            #     "required": [ QUERY_PARAM_NAME ]
            # },
        },
    },
    {
        "type": "function",
        "function": {
            "name": PHOTO_TOOL_NAME,
            "description": """Analyzes or describes the photo you have from the user's current perspective.
Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.""",
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         QUERY_PARAM_NAME: {
            #             "type": "string",
            #             "description": "User's query to answer, describing what they want answered, expressed as a command that NEVER refers to the photo or image itself"
            #         },
            #     },
            #     "required": [ QUERY_PARAM_NAME ]
            # },
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

async def handle_tool(
    tools: List[Any],
    tool_call: ChatCompletionMessageToolCall,
    user_message: str,
    message_history: List[Message] | None,
    image_bytes: bytes | None,
    location: str | None,
    local_time: str | None,
    web_search: WebSearch,
    vision: Vision,
    learned_context: Dict[str, str] | None,
    token_usage_by_model: Dict[str, TokenUsage],
    capabilities_used: List[Capability],
    tools_used: List[Dict[str, Any]],
    timings: Dict[str, str],
    extra_context: str|None
) -> str:
    tool_functions = {
        SEARCH_TOOL_NAME: web_search.search_web,                # returns WebSearchResult
        PHOTO_TOOL_NAME: handle_photo_tool,                     # returns WebSearchResult | str
        DUMMY_SEARCH_TOOL_NAME: handle_general_knowledge_tool,  # returns str
        IMAGE_GENERATION_TOOL_NAME: handle_image_generation_tool # returns str
    }

    function_name = tool_call.function.name
    function_to_call = tool_functions.get(function_name)
    if function_to_call is None:
        # Error: GPT hallucinated a tool
        return "Error: you hallucinated a tool that doesn't exist. Tell user you had trouble interpreting the request and ask them to rephrase it."

    function_args = prepare_tool_arguments(
        tools=tools,
        tool_call=tool_call,
        user_message=user_message,
        message_history=message_history,
        image_bytes=image_bytes,
        location=location,
        local_time=local_time,
        web_search=web_search,
        vision=vision,
        learned_context=learned_context,
        token_usage_by_model=token_usage_by_model,
        capabilities_used=capabilities_used,
        extra_context=extra_context
    )

    tool_start_time = timeit.default_timer()
    if function_name == IMAGE_GENERATION_TOOL_NAME and image_bytes is None:
        return "NO_IMAGE_PROVIDED_ERROR"
    function_response: WebSearchResult | str = await function_to_call(**function_args)
    total_tool_time = round(timeit.default_timer() - tool_start_time, 3)
    timings[f"tool_{function_name}"] = f"{total_tool_time:.3f}"

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
    tools: List[Any],
    tool_call: ChatCompletionMessageToolCall,
    user_message: str,
    message_history: List[Message] | None,
    image_bytes: bytes | None,
    location: str | None,
    local_time: str | None,
    web_search: WebSearch,
    vision: Vision,
    learned_context: Dict[str, str] | None,
    token_usage_by_model: Dict[str, TokenUsage],
    capabilities_used: List[Capability],
    extra_context: str|None
) -> Dict[str, Any]:
    # Get function description we passed to GPT. This function should be called after we have
    # validated that a valid tool call was generated.
    # function_description = [ description for description in tools if description["function"]["name"] == tool_call.function.name ][0]
    # function_parameters = function_description["function"]["parameters"]["properties"]

    # Parse arguments and ensure they are all str or bool for now. Drop any that aren't.
    args: Dict[str, Any] = {}
    try:
        args = json.loads(tool_call.function.arguments)
    except:
        pass
    # for param_name in list(args.keys()):
        # if param_name not in function_parameters:
        #     # GPT hallucinated a parameter
        #     del args[param_name]
        #     continue
        # if function_parameters[param_name]["type"] == "string" and type(args[param_name]) != str:
        #     del args[param_name]
        #     continue
        # if function_parameters[param_name]["type"] == "boolean" and type(args[param_name]) != bool:
        #     del args[param_name]
        #     continue
        # if function_parameters[param_name]["type"] not in [ "string", "boolean" ]:
        #     # Need to keep this up to date with the tools we define
        #     raise ValueError(f"Unsupported tool parameter type: {function_parameters[param_name]['type']}")

    # Fill in args required by all tools
    args["location"] = location if location else "unknown"
    args[QUERY_PARAM_NAME] = args[QUERY_PARAM_NAME] if QUERY_PARAM_NAME in args else user_message
    args["message_history"] = message_history
    args["token_usage_by_model"] = token_usage_by_model
    args["extra_context"] = extra_context
    # Photo tool additional parameters we need to inject
    if tool_call.function.name == PHOTO_TOOL_NAME:
        args["image_bytes"] = image_bytes
        args["vision"] = vision
        args["web_search"] = web_search
        args["local_time"] = local_time
        args["learned_context"] = learned_context
        args["capabilities_used"] = capabilities_used
        args["message_history"] = message_history
        args["query"] = user_message
    if tool_call.function.name == IMAGE_GENERATION_TOOL_NAME:
        args["image_bytes"] = image_bytes

    return args

async def handle_general_knowledge_tool(
    query: str,
    message_history: List[Message] | None,
    token_usage_by_model: Dict[str, TokenUsage],
    image_bytes: bytes | None = None,
    local_time: str | None = None,
    location: str | None = None,
    learned_context: Dict[str,str] | None = None,
    extra_context: str | None = None
) -> str:
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
    return ""

async def handle_photo_tool(
    query: str,
    message_history: List[Message] | None,
    vision: Vision,
    web_search: WebSearch,
    token_usage_by_model: Dict[str, TokenUsage],
    capabilities_used: List[Capability],
    google_reverse_image_search: bool = False,  # default in case GPT doesn't generate it
    translate: bool = False,                    # default in case GPT doesn't generate it
    image_bytes: bytes | None = None,
    local_time: str | None = None,
    location: str | None = None,
    learned_context: Dict[str,str] | None = None,
    extra_context: str | None = None
) -> str | WebSearchResult:
    # extra_context = "\n\n" + create_context_system_message(local_time=local_time, location=location, learned_context=learned_context)

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
        message_history=message_history,
        extra_context=extra_context,
        image_bytes=image_bytes,
        token_usage_by_model=token_usage_by_model
    )
    print(f"Vision: {output}")
    if output is None:
        return "Error: vision tool generated an improperly formatted result. Tell user that there was a temporary glitch and ask them to try again."
    
    # If no web search required, output vision response directly
    if not output.web_search_needed():
        return output.response

    # Perform web search and produce a synthesized response telling assistant where each piece of
    # information came from. Web search will lack important vision information. We need to return
    # both and have the assistant figure out which info to use.
    capabilities_used.append(Capability.REVERSE_IMAGE_SEARCH if output.reverse_image_search else Capability.WEB_SEARCH)
    extra_context += "\nOutput from photo:" + output.response
    web_result = await web_search.search_web(
        query=output.web_query.strip("\""),
        message_history=message_history,
        use_photo=output.reverse_image_search,
        image_bytes=image_bytes,
        location=location,
        token_usage_by_model=token_usage_by_model,
        extra_context=extra_context
    )
    
    return web_result

async def handle_image_generation_tool(
    query: str,
    message_history: List[Message] | None,
    description: str,
    token_usage_by_model: Dict[str, TokenUsage],
    image_bytes: bytes | None = None,
    local_time: str | None = None,
    location: str | None = None,
    learned_context: Dict[str,str] | None = None,
    extra_context: str | None = None
) -> str:
    """
    Generates an image based on a description or prompt.
    """
    # Generate image
    image_generator = ReplicateGenerateImage()
    image = await image_generator.generate_image(query=description, use_image=True, image_bytes=image_bytes)
    return image

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
        if isinstance(value, list) and arg_name != "message_history":
            function_args[arg_name] = ", ".join(function_args[arg_name])
    if "message_history" in function_args:
        del function_args["message_history"]
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

class GPTAssistant(Assistant):
    def __init__(self, client: openai.AsyncOpenAI | groq.AsyncGroq):
        """
        Instantiate the assistant using an OpenAI GPT or Groq model. The Groq API is a clone of
        OpenAI's, allowing a Groq client to be passed.
        """
        self._client = client

    # Refer to definition of Assistant for description of parameters
    async def send_to_assistant(
        self,
        prompt: str,
        noa_system_prompt: str | None,
        image_bytes: bytes | None,
        message_history: List[Message] | None,
        learned_context: Dict[str, str],
        location_address: str | None,
        local_time: str | None,
        model: str | None,
        web_search: WebSearch,
        vision: Vision,
        speculative_vision: bool
    ) -> AssistantResponse:
        # Default model (differs for OpenAI and Groq)
        if model is None:
            if type(self._client) == openai.AsyncOpenAI:
                model = "gpt-4o"
            elif type(self._client) == groq.AsyncGroq:
                model = "llama3-70b-8192"
            else:
                raise TypeError("client must be AsyncOpenAI or AsyncGroq")
        
        # Get copy of tool description
        tools = TOOLS.copy()

        # GPT-4o is a special case: if vision tool is also GPT-4o, then we remove it as a tool and
        # always submit images with queries.
        gpt4o_end_to_end = False
        # End-to-end mode DISABLED for now to improve latency: in end-to-end mode, every image is
        # processed, which is slower for queries that don't require image analysis. So we actually
        # want to use the vision tool to device when to do that. Assumption is that most questions
        # are not vision-related.
        if  False and model == "gpt-4o" and isinstance(vision, GPT4Vision) and vision.model == "gpt-4o":
            speculative_vision = False  # doesn't make sense anymore
            tools = [ tool for tool in tools if tool["function"]["name"] != PHOTO_TOOL_NAME ]
            print("End-to-end GPT-4o assistant activated")
            gpt4o_end_to_end = True

        # Keep track of time taken
        timings: Dict[str, str] = {}

        # Prepare response datastructure
        returned_response = AssistantResponse(token_usage_by_model={}, capabilities_used=[], response="", debug_tools="", timings="")

        # Make copy of message history so we can modify it in-flight during tool use
        message_history = message_history.copy() if message_history else None
        full_message_history = message_history.copy() if message_history else None

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

        # Patch up user message to include image if we are in end-to-end gpt-4o mode
        if gpt4o_end_to_end and image_bytes is not None:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = detect_media_type(image_bytes=image_bytes)
            user_message = {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    { "type": "image_url", "image_url": { "url": f"data:{media_type};base64,{image_base64}" } }
                ]
            }
            message_history[-1] = user_message

        # Inject context into our copy by appending it to system message. Unclear whether multiple
        # system messages are confusing to the assistant or not but cursory testing shows this
        # seems to work.
        extra_context = create_context_system_message(local_time=local_time, location=location_address, learned_context=learned_context)
        if noa_system_prompt is not None:
            extra_context = f"{noa_system_prompt}\n{extra_context}"
        extra_context_message = Message(role=Role.SYSTEM, content=extra_context)
        message_history.append(extra_context_message)

        # Start timing of initial LLM call and entire process
        t0 = timeit.default_timer()
        tstart = t0

        # Speculative vision call
        speculative_vision_task = asyncio.create_task(
            handle_photo_tool(
                query=prompt,
                message_history=full_message_history,
                vision=vision,
                web_search=web_search,
                token_usage_by_model=returned_response.token_usage_by_model,
                capabilities_used=returned_response.capabilities_used,
                google_reverse_image_search=False,
                translate=False,
                image_bytes=image_bytes,
                local_time=local_time,
                location=location_address,
                learned_context=learned_context,
                extra_context=extra_context
            )
        ) if speculative_vision else None

        speculative_search_task = asyncio.create_task(
            web_search.search_web(
                query=prompt,
                message_history=full_message_history,
                token_usage_by_model=returned_response.token_usage_by_model,
                image_bytes=image_bytes,
                location=location_address,
                extra_context=extra_context
            )
        ) 

        # Initial GPT call, which may request tool use
        initial_llm_task = asyncio.create_task(
            self._client.chat.completions.create(
                model=model,
                messages=message_history,
                tools=tools,
                tool_choice="auto"
            )
        )

        # Kick off both tasks but ensure LLM completes
        initial_tasks = [ initial_llm_task ]
        if speculative_vision_task is not None:
            initial_tasks.append(speculative_vision_task)
        if speculative_search_task is not None:
            initial_tasks.append(speculative_search_task)
        completed_tasks, pending_tasks = await asyncio.wait(initial_tasks, return_when=asyncio.FIRST_COMPLETED)
        first_response = await initial_llm_task
        first_response_message = first_response.choices[0].message
        t1 = timeit.default_timer()
        timings["llm_initial"] = f"{t1-t0:.3f}"

        # Aggregate token counts and potential initial response
        accumulate_token_usage(
            token_usage_by_model=returned_response.token_usage_by_model,
            model=model,
            input_tokens=first_response.usage.prompt_tokens,
            output_tokens=first_response.usage.completion_tokens,
            total_tokens=first_response.usage.total_tokens
        )

        # If there are no tool requests, the initial response will be returned
        returned_response.response = first_response_message.content

        # Handle tool requests
        tools_used = []
        tools_used.append({ "learned_context": learned_context })   # log context here for now
        if first_response_message.tool_calls:
            # If image generation tool then kill speculative tasks
            if first_response_message.tool_calls[0].function.name == IMAGE_GENERATION_TOOL_NAME:
                self._cancel_tasks([ speculative_vision_task, speculative_search_task ])
            # Append initial response to history, which may include tool use
            message_history.append(first_response_message)

            # Invoke all the tools in parallel and wait for them all to complete. Vision is special:
            # we already have a speculative query in progress.
            t0 = timeit.default_timer()
            tool_handlers = []
            for tool_call in first_response_message.tool_calls:
                if tool_call.function.name == PHOTO_TOOL_NAME and speculative_vision_task is not None:
                    tool_handlers.append(speculative_vision_task)
                    tools_used.append(
                        create_debug_tool_info_object(
                            function_name=PHOTO_TOOL_NAME,
                            function_args={},
                            tool_time=-1,
                            search_result=None
                        )
                    )
                elif tool_call.function.name == SEARCH_TOOL_NAME and speculative_search_task is not None:
                    tool_handlers.append(speculative_search_task)
                    returned_response.capabilities_used.append(Capability.WEB_SEARCH)
                    tools_used.append(
                        create_debug_tool_info_object(
                            function_name=SEARCH_TOOL_NAME,
                            function_args={},
                            tool_time=-1,
                            search_result=None
                        )
                    )
                else:
                    tool_handlers.append(
                        handle_tool(
                            tools=tools,
                            tool_call=tool_call,
                            user_message=prompt,
                            message_history=full_message_history,   # full history because tools may have their own requirements on history length
                            image_bytes=image_bytes,
                            location=location_address,
                            local_time=local_time,
                            web_search=web_search,
                            vision=vision,
                            learned_context=learned_context,
                            token_usage_by_model=returned_response.token_usage_by_model,
                            capabilities_used=returned_response.capabilities_used,
                            tools_used=tools_used,
                            timings=timings,
                            extra_context=extra_context
                        )
                    )
            tool_outputs = await asyncio.gather(*tool_handlers)
            t1 = timeit.default_timer()
            timings["tool_calls"] = f"{t1-t0:.3f}"

            # Ensure everything is str
            for i in range(len(tool_outputs)):
                if isinstance(tool_outputs[i], WebSearchResult):
                    tool_outputs[i] = tool_outputs[i].summary

            # Append all the responses for GPT to continue
            for i in range(len(tool_outputs)):
                # If image generation tool then return response
                if first_response_message.tool_calls[i].function.name == IMAGE_GENERATION_TOOL_NAME:
                    if tool_outputs[i] == "NO_IMAGE_PROVIDED_ERROR":
                        tool_outputs[i] = "I think you're referring to something you can see. Can you provide a photo?"
                        returned_response.image = ""
                    else:
                        returned_response.response = "Here is the image you requested"
                        returned_response.capabilities_used.append(Capability.IMAGE_GENERATION)
                        returned_response.debug_tools = json.dumps(tools_used)
                        returned_response.image = tool_outputs[i]
                        t1 = timeit.default_timer()
                        timings["total_time"] = f"{t1-tstart:.3f}"
                        returned_response.timings = json.dumps(timings)
                        return returned_response
                if first_response_message.tool_calls[i].function.name == PHOTO_TOOL_NAME and len(first_response_message.tool_calls) == 1:  
                    returned_response.response = tool_outputs[i]

                if first_response_message.tool_calls[i].function.name == SEARCH_TOOL_NAME and ( len(first_response_message.tool_calls) == 1 or all(first_response_message.tool_calls[i].function.name == SEARCH_TOOL_NAME for i in range(len(first_response_message.tool_calls))) ):  
                    returned_response.response = tool_outputs[i]
                if  len(first_response_message.tool_calls) > 1:  
                    message_history.append(
                        {
                            "tool_call_id": first_response_message.tool_calls[i].id,
                            "role": "tool",
                            "name": first_response_message.tool_calls[i].function.name,
                            "content": tool_outputs[i],
                        }
                    )
            if len(first_response_message.tool_calls) == 1 or all(first_response_message.tool_calls[i].function.name == SEARCH_TOOL_NAME for i in range(len(first_response_message.tool_calls))):  
                self._cancel_tasks([ speculative_vision_task, speculative_search_task ])
                #  don't need do new llm call
                
                returned_response.debug_tools = json.dumps(tools_used)
                returned_response.image = ""

                t1 = timeit.default_timer()
                timings["total_time"] = f"{t1-tstart:.3f}"
                returned_response.timings = json.dumps(timings)

                return returned_response
            # Get final response from model
            t0 = timeit.default_timer()
            second_response = await self._client.chat.completions.create(
                model=model,
                messages=message_history
            )
            t1 = timeit.default_timer()
            timings["llm_final"] = f"{t1-t0:.3f}"

            # Aggregate tokens and response
            accumulate_token_usage(
                token_usage_by_model=returned_response.token_usage_by_model,
                model=model,
                input_tokens=second_response.usage.prompt_tokens,
                output_tokens=second_response.usage.completion_tokens,
                total_tokens=second_response.usage.total_tokens
            )
            returned_response.response = second_response.choices[0].message.content
        else:
            # No tools, cancel speculative tasks
            self._cancel_tasks([ speculative_vision_task, speculative_search_task ])

        # If no tools were used, only assistant capability recorded
        if len(returned_response.capabilities_used) == 0:
            returned_response.capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)

        # Total time
        t1 = timeit.default_timer()
        timings["total_time"] = f"{t1-tstart:.3f}"

        # Return final response
        returned_response.debug_tools = json.dumps(tools_used)
        returned_response.timings = json.dumps(timings)
        returned_response.image = ""
        return returned_response
    
    @staticmethod
    def _cancel_tasks(tasks: list):
        for task in tasks:
            if task is not None:
                task.cancel()

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

Assistant.register(GPTAssistant)