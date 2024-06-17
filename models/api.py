#
# api.py
#
# Server API models.
#

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel

from .token_usage import TokenUsage


class Role(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

class Message(BaseModel):
    role: Role
    content: str

class Capability(str, Enum):
    ASSISTANT_KNOWLEDGE = "assistant_knowledge"
    WEB_SEARCH = "web_search"
    VISION = "vision"
    REVERSE_IMAGE_SEARCH = "reverse_image_search"
    IMAGE_GENERATION = "image_generation"

class GenerateImageService(str, Enum):
    REPLICATE   = "replicate"

class MultimodalRequest(BaseModel):
    messages: Optional[List[Message]]
    prompt: Optional[str] = ""
    noa_system_prompt: Optional[str] = None
    local_time: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[str] = None
    longitude: Optional[str] = None
    generate_image_service: Optional[GenerateImageService] = GenerateImageService.REPLICATE
    openai_key: Optional[str] = None
    perplexity_key: Optional[str] = None

class MultimodalResponse(BaseModel):
    user_prompt: str
    response: str
    image: str
    token_usage_by_model: Dict[str, TokenUsage]
    capabilities_used: List[Capability]
    timings: Dict[str, float]
    stream_finished: bool = True