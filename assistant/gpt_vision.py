#
# gpt_vision.py
#
# GPT-4-based vision tool.
#

from typing import Dict, Optional

import openai
from pydantic import BaseModel

from models import Message, TokenUsage, accumulate_token_usage
from .vision_tool_output import VisionToolOutput
from util import is_blurry_image
import base64
from .models import VISION_MODEL_GPT, SYSTEM_MESSAGE_VISION_GPT, VisionResponse


async def vision_query_gpt(
    client: openai.AsyncOpenAI,
    token_usage_by_model: Dict[str, TokenUsage],
    query: str,
    image_base64: str | None,
    media_type: str | None,
    message_history: list[Message]=[],
    extra_context: str | None = None
) -> VisionToolOutput:
    # Create messages for GPT w/ image. No message history or extra context for this tool, as we
    # will rely on second LLM call. Passing in message history and extra context necessary to
    # allow direct tool output seems to cause this to take longer, hence we don't permit it.

    # Check if image is blurry
    if image_base64 is not None and is_blurry_image(base64.b64decode(image_base64)):
        return VisionToolOutput(
            is_error=False,
            response="I’m having a hard time seeing the image clearly, try shifting position or lighting and try again.",
            web_query=None
        )
    user_message = {
        "role": "user",
        "content": [
            { "type": "text", "text": query }
        ]
    }
    clean_message_history = [m for m in message_history if m.role == "user" or m.role == "assistant"]
    
    _system_message = SYSTEM_MESSAGE_VISION_GPT

    if extra_context is not None:
        _system_message = _system_message + "\n" + extra_context
    if image_base64 is not None and media_type is not None:
        user_message["content"].append({ "type": "image_url", "image_url": { "url": f"data:{media_type};base64,{image_base64}" } })
    messages = [
        { "role": "system", "content": _system_message },
    ] + clean_message_history + [user_message]

    # Call GPT
    response = await client.chat.completions.create(
        model=VISION_MODEL_GPT,
        messages=messages,
    )
    accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=response.usage, model=VISION_MODEL_GPT)

    # Parse structured output. Response expected to be JSON but may be wrapped with 
    # ```json ... ```
    try:
        content = response.choices[0].message.content
        json_start = content.find("{")
        json_end = content.rfind("}")
    except Exception as e:
        print(f"Error: Unable to parse vision response: {e}")
        return VisionToolOutput(
            is_error=False,
            response="I’m having a hard time seeing the image clearly, try shifting position or lighting and try again.",
            web_query=None
        )

    json_string = content[json_start : json_end + 1] if json_start > -1 and json_end > -1 else content
    try:
        vision_response = VisionResponse.model_validate_json(json_data=json_string)
        # vision_response.reverse_image_search = vision_response.reverse_image_search is not None and vision_response.reverse_image_search == True
        # if len(vision_response.web_query) == 0 and vision_response.reverse_image_search:
            # If no web query output but reverse image search asked for, just use user query
            # directly. This is sub-optimal and it would be better to figure out a way to ensure
            # web_query is generated when reverse_image_search is true.
            # vision_response.web_query = query
    except Exception as e:
        print(f"Error: Unable to parse vision response: {e}")
        return VisionToolOutput(
            is_error=True,
            response="I’m having a hard time seeing the image clearly, try shifting position or lighting and try again.",
            web_query=None
        )

    # Return vision output
    return VisionToolOutput(
        is_error=False,
        response=vision_response.response,
        web_query = vision_response.web_query if vision_response.web_query is not None and len(vision_response.web_query) > 0 else None
    )