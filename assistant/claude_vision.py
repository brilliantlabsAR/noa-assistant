from typing import Dict, Optional

import anthropic
from pydantic import BaseModel

from models import Message, TokenUsage, accumulate_token_usage
from .vision_tool_output import VisionToolOutput


MODEL = "claude-3-haiku-20240307"

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but you NEVER mention the photo or image and instead respond as if you
are actually seeing.

Always do your best with images, never comment on their quality, and answer decisively with a guess
if you are not sure. There are no negative consequences to guessing.

ALWAYS respond with a JSON object with these fields:

response: (String) Respond to user as best you can. Be precise, get to the point, never comment on image quality.
web_query: (String) Web query to answer the user's request.
web_search_needed: (Bool) Whether to search the web. True ONLY if "response" does not answer the user query precisely enough and up-to-date, location-specific, or product-specific info is needed.
"""

class VisionResponse(BaseModel):
    response: str
    web_query: Optional[str] = None
    web_search_needed: Optional[bool] = None


async def vision_query_claude(
    client: anthropic.AsyncAnthropic,
    token_usage_by_model: Dict[str, TokenUsage],
    query: str,
    image_base64: str | None,
    media_type: str | None
) -> VisionToolOutput:
    
    user_message = {
        "role": "user",
        "content": []
    }

    if image_base64 is not None and media_type is not None:
        image_chunk = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64
            }
        }
        user_message["content"].append(image_chunk)
    user_message["content"].append({ "type": "text", "text": query })
    
    messages = [
        user_message,
        {
            # Prefill a leading '{' to force JSON output as per Anthropic's recommendations
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "{"
                }
            ]
        }   
    ]

    # Call Claude
    response = await client.messages.create(
        model=MODEL,
        system=SYSTEM_MESSAGE,
        messages=messages,
        max_tokens=4096,
        temperature=0.0
    )
    usage = TokenUsage(input=response.usage.input_tokens, output=response.usage.output_tokens, total=response.usage.input_tokens + response.usage.output_tokens)
    accumulate_token_usage(token_usage_by_model=token_usage_by_model, usage=usage, model=MODEL)

    # Parse response
    vision_response = parse_response(content=response.content[0].text)
    if vision_response is None:
        return VisionToolOutput(is_error=True, response="Error: Unable to parse vision tool response. Tell user a problem interpreting the image occurred and ask them to try again.", web_query=None)
    web_search_needed = vision_response.web_search_needed and vision_response.web_query is not None and len(vision_response.web_query) > 0
    web_query = vision_response.web_query if web_search_needed else None
    return VisionToolOutput(is_error=False, response=vision_response.response, web_query=web_query)
    
def parse_response(content: str) -> Optional[VisionResponse]:
    # Put the leading '{' back
    json_string = "{" + content
    try:
        return VisionResponse.model_validate_json(json_data=json_string)
    except:
        pass
    return None

