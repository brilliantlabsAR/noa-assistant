#
# gpt4vision.py
#
# Vision tool implementation based on GPT-4.
#

import base64
from typing import Dict

import openai

from .vision import Vision
from .utils import detect_media_type
from models import TokenUsage, accumulate_token_usage


class GPT4Vision(Vision):
    def __init__(self, client: openai.OpenAI, model: str = "gpt-4-vision-preview"):
        self._client = client
        self._model = model
    
    def query_image(self, system_message: str, query: str, image_bytes: bytes | None, token_usage_by_model: Dict[str, TokenUsage]) -> str:
        messages = [
            { "role": "system", "content": system_message },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": query }
                ]
            }
        ]
        
        if image_bytes:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            media_type = detect_media_type(image_bytes=image_bytes)
            messages[1]["content"].append({ "type": "image_url", "image_url": { "url": f"data:{media_type};base64,{image_base64}" } }),
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096
        )

        accumulate_token_usage(
            token_usage_by_model=token_usage_by_model,
            model=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
        
        return response.choices[0].message.content
    
Vision.register(GPT4Vision)