
#
# claude_vision.py
#
# Vision tool implementation based on Anthropic.
#

import base64
from typing import Dict

import anthropic

from .vision import Vision
from .utils import detect_media_type
from models import TokenUsage, accumulate_token_usage


class ClaudeVision(Vision):
    def __init__(self, client:anthropic.Anthropic, model:str="claude-3-haiku-20240307"):
        self._client = client
        self._model = model
    
    def query_image(self, system_message: str, query: str, image_bytes: bytes | None, token_usage_by_model: Dict[str, TokenUsage]) -> str:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes is not None else ""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": detect_media_type(image_bytes=image_bytes),
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": query
                    }
                ]
            }   
        ]

        response = self._client.messages.create(
            model=self._model,
            system=system_message,
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )

        accumulate_token_usage(
            token_usage_by_model=token_usage_by_model,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )

        return response.content[0].text

Vision.register(ClaudeVision)