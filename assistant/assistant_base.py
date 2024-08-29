from typing import Any, Dict, List, Optional
import asyncio
import base64
import json
import timeit
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
import openai
from .claude_vision import vision_query_claude
from .gpt_vision import vision_query_gpt
from generate_image.replicate import ReplicateGenerateImage
from models import Capability, Message, TokenUsage, accumulate_token_usage
from util import detect_media_type
from .models import (
    AssistantVisionTool,
    ToolOutput,
    NoaResponse,
    SEARCH_TOOL_NAME,
    VISION_TOOL_NAME,
    IMAGE_GENERATION_TOOL_NAME,
    IMAGE_GENERATION_PARAM_NAME,
    TOPIC_CHNAGED_PARAM_NAME,
    CONTEXT_SYSTEM_MESSAGE_PREFIX,
    TEXT_MODEL_GPT
)
class AssistantBase:
    _client: openai.AsyncOpenAI = None
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
                arguments=json.dumps({ "query": query, "topic_changed": False }),
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
        response = await self._client.beta.chat.completions.parse(
            model=TEXT_MODEL_GPT,
            messages=messages,
            # tools=TOOLS, #TODO: do we second round of tools eg image generation after vision tool?
            # tool_choice="auto",
            response_format=NoaResponse
        )
        accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=response.usage, model=TEXT_MODEL_GPT)
        timings["final"] = timeit.default_timer() - t1
        # print(f"Final response: {response}")
        return ToolOutput(text=response.choices[0].message.parsed.response, safe_for_final_response=True, topic_changed=response.choices[0].message.parsed.topic_changed)
        
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
        local_time: str | None,
        prompt: str | None
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
                    local_time=local_time,
                    prompt=prompt
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
        local_time: str | None,
        prompt: str | None = None
    ) -> asyncio.Task[ToolOutput]:
        tool_functions_by_name = {
            SEARCH_TOOL_NAME: self._handle_web_search_tool,
            VISION_TOOL_NAME: self._handle_vision_tool
        }
        # Validate tool
        if tool_call.function.name not in tool_functions_by_name:
            print(f"Error: Hallucinated tool: {tool_call.function.name}")
            return asyncio.create_task(self._return_tool_error_message(message="Error: you hallucinated a tool that doesn't exist. Tell user you had trouble interpreting the request and ask them to rephrase it."))
        args: Dict[str, Any] | None = json.loads(tool_call.function.arguments)
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
        # if vision combined user prompt in query
        if prompt is not None and tool_call.function.name == VISION_TOOL_NAME:
            args["query"] = "user prompt: " + prompt + "\n AI prompt: " + args["query"]
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
    async def make_async(output: Any) -> Any:
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
        local_time: str | None,
        topic_changed: bool | None = None
    ) -> ToolOutput:
        t_start = timeit.default_timer()

        image_base64: Optional[str] = None
        media_type: Optional[str] = None
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = detect_media_type(image_bytes=image_bytes)
        
        extra_context = CONTEXT_SYSTEM_MESSAGE_PREFIX + "\n".join([ f"<{key}>{value}</{key}>" for key, value in { "location": location_address, "current_time": local_time }.items() if value is not None ])
        extra_context = f"{flavor_prompt}\n{extra_context}" if flavor_prompt is not None else extra_context
        if self._vision_tool == AssistantVisionTool.GPT4O:
            output = await vision_query_gpt(
                client=self._client,
                token_usage_by_model=token_usage_by_model,
                query=query,
                image_base64=image_base64,
                media_type=media_type,
                message_history= [] if topic_changed  else message_history,
                extra_context=extra_context
            )
        else:
            output = await vision_query_claude(
                client=self._anthropic_client,
                token_usage_by_model=token_usage_by_model,
                query=query,
                image_base64=image_base64,
                media_type=media_type,
                message_history= [] if topic_changed  else message_history,
                extra_context=extra_context
            )

        t_end = timeit.default_timer()
        timings["vision_tool"] = t_end - t_start
        capabilities_used.append(Capability.VISION)

        if output.web_query is None or output.web_query == "":
            return ToolOutput(text=output.response, safe_for_final_response=True, topic_changed=topic_changed) 
        
        # Perform web search and produce a synthesized response telling assistant where each piece of
        # information came from. Web search will lack important vision information. We need to return
        # both and have the assistant figure out which info to use.
        web_result = await self._handle_web_search_tool(
            token_usage_by_model=token_usage_by_model,
            capabilities_used=capabilities_used,
            timings=timings,
            query=output.web_query,
            flavor_prompt=flavor_prompt,
            message_history=[],
            image_bytes=None,
            location_address=location_address,
            local_time=local_time,
        )
        return ToolOutput(text=f"{web_result.text}", safe_for_final_response=True, topic_changed=topic_changed)
        # return ToolOutput(text=f"HERE IS WHAT YOU SEE: {output.response}\nEXTRA INFO FROM WEB: {web_result}", safe_for_final_response=False)
    
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
        local_time: str | None,
        topic_changed: bool | None = None
    ) -> ToolOutput:
        t_start = timeit.default_timer()
        output = await self._web_tool.search_web(
            token_usage_by_model=token_usage_by_model,
            timings=timings,
            query=query,
            flavor_prompt=flavor_prompt,
            message_history=[] if topic_changed else message_history,
            location=location_address
        )
        t_end = timeit.default_timer()
        timings["web_search_tool"] = t_end - t_start
        capabilities_used.append(Capability.WEB_SEARCH)
        return ToolOutput(text=output, safe_for_final_response=True, topic_changed=topic_changed)
    
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
                    return args[IMAGE_GENERATION_PARAM_NAME], args[TOPIC_CHNAGED_PARAM_NAME]
                except:
                    # Function calling failed to produce the required parameter, just use user
                    # prompt as a fallback
                    return user_prompt, False
        return None, False

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
    def extra_context(
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
        return extra_context