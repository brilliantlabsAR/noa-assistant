#
# claude_vision.py
#
# Claude-based vision tool.
#

from typing import Dict, Optional, List

import anthropic
from pydantic import BaseModel

from models import Message, TokenUsage, accumulate_token_usage
from .vision_tool_output import VisionToolOutput
from util import is_blurry_image
import base64

MODEL = "claude-3-haiku-20240307"
# MODEL = "claude-3-5-sonnet-20240620"
SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but you NEVER mention the photo or image and instead respond as if you
are actually seeing.

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do your best with images.

ALWAYS respond with a valid JSON object with these fields:

response: (String) Respond to user as best you can. Be precise, get to the point, and speak as though you actually see the image. If it needs a web search it will be a description of the image.
web_query: (String) Empty if your "response" answers everything user asked. If web search based on visual description would be more helpful, create a query (e.g. up-to-date, location-based, or product info).

examples:
1. If the user asks "What do you see?" and the image is a cat in a room, you would respond:
{
  "response": "You are looking at a cat in a room.",
  "web_query": ""
}

2. If the user asks "What is that?" and the image is a red shoe with white laces, you would respond:
{
    "response": "A red shoe with white laces.",
    "web_query": "red shoe with white laces"
}
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
    media_type: str | None,
    message_history: list[Message]=[]
) -> VisionToolOutput:
    
    user_message = {
        "role": "user",
        "content": []
    }
    # Check if image is blurry
    if image_base64 is not None and is_blurry_image(base64.b64decode(image_base64)):
        print("Image is too blurry to interpret.")
        return VisionToolOutput(
            is_error=False,
            response="The image is too blurry to interpret. Please try again.",
            web_query=None
        )
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
    clean_message_history = [mag for mag in message_history if mag.role == "assistant" or mag.role == "user"]
    clean_message_history = make_alternating(messages=clean_message_history)
    messages = clean_message_history + [
        user_message  
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
    print(f"Vision response: {vision_response}")
    if vision_response is None:
        return VisionToolOutput(is_error=True, response="Error: Unable to parse vision tool response. Tell user a problem interpreting the image occurred and ask them to try again.", web_query=None)
    web_search_needed = vision_response.web_search_needed and vision_response.web_query is not None and len(vision_response.web_query) > 0
    web_query = vision_response.web_query if web_search_needed else None
    return VisionToolOutput(is_error=False, response=vision_response.response, web_query=web_query)
    
def parse_response(content: str) -> Optional[VisionResponse]:
    try:
        return VisionResponse.model_validate_json(json_data=content)
    except:
        pass
    return None

def make_alternating(messages: List[Message]) -> List[Message]:
    """
    Ensure that the messages are alternating between user and assistant.
    """
    # Start with the first message's role
    if len(messages) == 0:
        return []
    expected_role = messages[0].role
    last_message = messages[-1]
    alternating_messages = []
    expected_role = "user" if expected_role == "assistant" else "assistant"
    
    for i, message in enumerate(messages):
        if message.content.strip()=='':
            continue
        if message.role != expected_role:
            continue
        
        alternating_messages.append(message)
        expected_role = "assistant" if expected_role == "user" else "user"
    
    # Ensure the last message is from the assistant
    if alternating_messages and alternating_messages[-1].role != "assistant":
        if last_message.role == "assistant":
            alternating_messages.append(last_message)
        else:
            alternating_messages.pop()
    # if first message is from assistant, remove it
    if alternating_messages and alternating_messages[0].role == "assistant":
        alternating_messages.pop(0)
    return alternating_messages


