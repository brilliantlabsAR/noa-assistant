from enum import Enum
from typing import Optional, List
from pydantic import BaseModel
from dataclasses import dataclass
import openai
from openai.types.completion_usage import CompletionUsage

###################################################################################################
# GPT Configuration
####################################################################################################

# TEXT_MODEL_GPT = "gpt-4o-mini-2024-07-18"

TEXT_MODEL_GPT = "gpt-4o-2024-08-06"

VISION_MODEL_GPT = "gpt-4o-2024-08-06"
####################################################################################################
# Prompts
####################################################################################################

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.

You can use the following tools to help you answer user queries:
- web_search: Provides up-to-date information on news, retail products, current events, local
  conditions, and esoteric knowledge. Performs a web search based on the user's query.
- analyze_photo: Analyzes or describes the photo you have from the user's current perspective. Use
    this tool if the user refers to something not identifiable from conversation context, such as with
    a demonstrative pronoun.
- generate_image : Generates an image based on a description or prompt.

If you think the topic of conversation has changed, return the following `topic_changed` as part of your result:
{
  'response': 'Your response here',
  'topic_changed': true
}
If you think the topic of conversation has not changed, return the following `topic_changed` as part of your result:
{
  'response': 'Your response here',
  'topic_changed': true
}
if conversation topic has changed, set `topic_changed` to true in the tool response too.
"""

SYSTEM_MESSAGE_VISION_GPT = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but you NEVER mention the photo or image and instead respond as if you
are actually seeing.

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do your best with images.

It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo in your wording.
only reply in English. Don’t include markdown , emojis or new lines.

ALWAYS respond with a valid JSON object with these fields:
response: (String) Respond to user as best you can. Be precise, get to the point, and speak as though you actually see the image. If it needs a web search it will be a description of the image.
web_query: (String) Empty if your "response" answers everything user asked. If web search based on visual description would be more helpful, create a query (e.g. up-to-date, location-based, or product info).

examples:
1. If the user asks "What do you see?" and the image is a cat in a room, you would respond:
{
  "response": "You are looking at a cat in a room.",
  "web_query": ""
}

2. If the user asks "where i can buy this?" and the image is a red shoe with white laces, you would respond:
{
    "response": "A red shoe with white laces.",
    "web_query": "red shoe with white laces"
}

"""

CONTEXT_SYSTEM_MESSAGE_PREFIX = "## Additional context about the user:"


SYSTEM_MESSAGE_WEB = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions.
only reply in English. Don’t include markdown , emojis or new lines.
"""

####################################################################################################
# Tools
####################################################################################################

SEARCH_TOOL_NAME = "web_search"
VISION_TOOL_NAME = "analyze_photo"
QUERY_PARAM_NAME = "query"
TOPIC_CHNAGED_PARAM_NAME = "topic_changed"

IMAGE_GENERATION_TOOL_NAME = "generate_image"
IMAGE_GENERATION_PARAM_NAME = "description"

####################################################################################################

@dataclass
class ToolOutput:
    text: str
    safe_for_final_response: bool # whether this can be output directly to user as a response (no second LLM call required)
    image_base64: Optional[str] = None
    topic_changed: Optional[bool] = None

class NoaResponse(BaseModel):
    response: str
    topic_changed : bool

class WebSearch(BaseModel):
    """
    Up-to-date information on news, retail products, current events, local conditions, and esoteric knowledge.
    If you think the topic of conversation has changed, return the following `topic_changed` as true:
    """
    query: str
    topic_changed: bool

class AnalyzePhoto(BaseModel):
    """
    Analyzes or describes the photo you have from the user's current perspective.
    Use this tool if user refers to something not identifiable from conversation context, such as with a demonstrative pronoun.
    If you think the topic of conversation has changed, return the following `topic_changed` as true:
    """
    query: str
    topic_changed: bool

class GenerateImage(BaseModel):
    """
    Generates an image based on a description or prompt.
    If you think the topic of conversation has changed, return the following `topic_changed` as true:
    """
    description: str
    topic_changed: bool


class VisionResponse(BaseModel):
    response: str
    web_query: Optional[str]

class AssistantVisionTool(str, Enum):
    GPT4O = "gpt-4o"
    HAIKU = "haiku"

TOOLS = [ 
            openai.pydantic_function_tool(WebSearch, name=SEARCH_TOOL_NAME),
            openai.pydantic_function_tool(AnalyzePhoto, name=VISION_TOOL_NAME),
            openai.pydantic_function_tool(GenerateImage, name=IMAGE_GENERATION_TOOL_NAME)
        ]



# For perplexity
class PerplexityMessage(BaseModel):
    role: str = None
    content: str = None

class MessageChoices(BaseModel):
    index: int = None
    finish_reason: str | None = None
    message: PerplexityMessage = None
    delta: PerplexityMessage = None

class PerplexityResponse(BaseModel):
    id: str = None
    model: str = None
    created: int = None
    usage: CompletionUsage = None
    object: str = None
    choices: List[MessageChoices] = None
    
    def summarise(self) -> str:
        if len(self.choices) > 0:
            return self.choices[0].message.content
        else:
            return "No results"
