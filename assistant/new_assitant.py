#
# assistant.py
#
# Assistant class and response object.
#

import asyncio
import json
import timeit
from typing import  Dict, List, Optional
import anthropic
import openai
from openai.types.chat import ChatCompletionMessageToolCall

from .response import AssistantResponse
from .web_search_tool import WebSearchTool
from models import Capability, Message, Role, TokenUsage, accumulate_token_usage
from .models import ToolOutput, AssistantVisionTool, NoaResponse, TEXT_MODEL_GPT, SYSTEM_MESSAGE, SEARCH_TOOL_NAME, TOPIC_CHNAGED_PARAM_NAME, TOOLS
from .assistant_base import AssistantBase


class Assistant(AssistantBase):
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
        topic_changed = False
        token_usage_by_model: Dict[str, TokenUsage] = {}
        speculative_token_usage_by_model: Dict[str, TokenUsage] = {}
        capabilities_used: List[Capability] = []
        speculative_capabilities_used: List[Capability] = []
        timings: Dict[str, float] = {}
        output_task: Optional[asyncio.Task[ToolOutput]] = None
        using_speculative_output_task = False

        message_history = message_history if message_history else []
        # take only the last 10 messages
        if len(message_history) > 10:
            message_history = message_history[-10:]
        system_message_final = SYSTEM_MESSAGE + "\n" + self.extra_context(flavor_prompt=flavor_prompt, location_address=location_address, local_time=local_time)
        messages = [Message(role=Role.SYSTEM, content=system_message_final)] + message_history + [ Message(role=Role.USER, content=prompt)]
        task_by_tool_name: Dict[str, asyncio.Task[ToolOutput]] = {}

        # #  start a web search tool in advance
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

        # Initial response from LLM. can be a tool call or a direct response
        initial_response = await self._client.beta.chat.completions.parse(
            model=TEXT_MODEL_GPT,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            response_format=NoaResponse
        )

        # collect token usage
        accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=initial_response.usage, model=TEXT_MODEL_GPT)
        initial_response_message = initial_response.choices[0].message
        timings["initial"] = timeit.default_timer() - t_start



        # Determine whether we have tool calls to handle
        tool_calls: List[ChatCompletionMessageToolCall] = [] if initial_response_message.tool_calls is None else initial_response_message.tool_calls

        # check if we have an image generation tool call
        image_generation_description, topic_changed = self._get_image_generation_description(tool_calls=tool_calls, user_prompt=prompt)

        if len(tool_calls) == 0:
            # No tools: return initial assistant response directly
            capabilities_used.append(Capability.ASSISTANT_KNOWLEDGE)
            output_task = self.make_async(ToolOutput(text=initial_response_message.parsed.response,
                                                    safe_for_final_response=True,
                                                     topic_changed=initial_response_message.parsed.topic_changed))
        elif image_generation_description is not None:
            # Special case: image generation
            tool_output =  self._handle_image_generation_tool(
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
            output_task = tool_output
        else:
            # Tool calls requested. Determine where to output final response from: one of the
            # speculative tools or a final LLM response
            print(f"Tools requested: {[ tool_call.function.name for tool_call in tool_calls ]}")
            
            # check if we have a speculative web search tool call
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                if tool_name == SEARCH_TOOL_NAME:
                    topic_changed = json.loads(tool_call.function.arguments).get(TOPIC_CHNAGED_PARAM_NAME, False)
                    output_task = task_by_tool_name.get(SEARCH_TOOL_NAME, None)
                    using_speculative_output_task = True
                    break

            if not using_speculative_output_task:
                # Cannot use a speculative tool, need to perform tool calls and create an output
                # cancel speculative tasks
                self._cancel_tasks(task_by_tool_name)
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
                    local_time=local_time,
                    prompt=prompt
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
                        output_task = self.make_async(tool_output)
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
            accumulate_token_usage(token_usage_by_model=token_usage_by_model, other=speculative_token_usage_by_model, model=TEXT_MODEL_GPT)
        assert output.safe_for_final_response                       # final output must be safe for direct response
        self._cancel_tasks(task_by_tool_name)                       # ensure any remaining speculative tasks are killed
        timings["total"] = timeit.default_timer() - t_start
        # round off all timings to 2 decimal places
        for key in timings:
            timings[key] = round(timings[key], 2)

        return AssistantResponse(
            token_usage_by_model=token_usage_by_model,
            capabilities_used=capabilities_used,
            response=output.text,
            timings=timings,
            image="" if output.image_base64 is None else output.image_base64,
            topic_changed=True if topic_changed or output.topic_changed else False,
        )
        