#
# gpt4vision.py
#
# Vision tool implementation based on GPT-4.
#

import base64
import timeit
from typing import Dict, Optional

import openai
from pydantic import BaseModel

from .vision import Vision, VisionOutput
from .utils import detect_media_type
from models import TokenUsage, accumulate_token_usage


SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke but you NEVER mention the photo or image and instead respond as if you
are actually seeing.

The camera is unfortunately VERY low quality but the user is counting on you to interpret the
blurry, pixelated images. NEVER comment on image quality. Do your best with images.

ALWAYS respond with a JSON object with these fields:

response: (String) Respond to user as best you can. Be precise, get to the point, and speak as though you actually see the image.
web_query: (String) Empty if your "response" answers everything user asked. If web search based on visual description would be more helpful, create a query (e.g. up-to-date, location-based, or product info).
reverse_image_search: (Bool) True if your web query from description is insufficient and including the *exact* thing user is looking at as visual target is needed.
"""

class ModelOutput(BaseModel):
    response: str
    web_query: Optional[str] = None
    reverse_image_search: Optional[bool] = None


class GPT4Vision(Vision):
    def __init__(self, client: openai.AsyncOpenAI, model: str = "gpt-4o"):
        self._client = client
        self._model = model
    
    @property
    def model(self):
        return self._model
    
    async def query_image(self, query: str, extra_context: str, image_bytes: bytes | None, token_usage_by_model: Dict[str, TokenUsage]) -> VisionOutput | None:
        messages = [
            { "role": "system", "content": SYSTEM_MESSAGE + extra_context },
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
        
        # Stream out the response then accumulate
        tstart = timeit.default_timer()
        t1 = None
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=4096,
            stream=True,
            stream_options={ "include_usage": True }
        )
        accumulated_response = []
        async for chunk in stream:
            if len(chunk.choices) == 0:
                # Final chunk has usage
                accumulate_token_usage(
                    token_usage_by_model=token_usage_by_model,
                    model=self._model,
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens
                )
                break
            else:
                response_chunk = chunk.choices[0].delta.content
                if response_chunk is not None:  # an empty content chunk can occur before the stop event
                    if t1 is None:
                        t1 = timeit.default_timer()
                    accumulated_response.append(response_chunk)
        response = "".join(accumulated_response)
        tend = timeit.default_timer()
        time_to_first = t1 - tstart
        total_time = tend - tstart
        pct_improvement = 100.0 * (1.0 - time_to_first / total_time)
        print(f"TIMING -- first: {time_to_first:.1f}, total: {total_time:.1f}, improvement: {pct_improvement:.1f}%")
        
        # Convert to VisionResponse and return
        output = self._parse_response(response)
        if output is None:
            return None
        web_query = output.web_query if output.web_query is not None else ""
        reverse_image_search = output.reverse_image_search is not None and output.reverse_image_search == True
        if len(web_query) == 0 and reverse_image_search:
            # If no web query output but reverse image search asked for, just use user query
            # directly. This is sub-optimal and it would be better to figure out a way to ensure
            # web_query is generated when reverse_image_search is true.
            web_query = query
        return VisionOutput(response=output.response, web_query=web_query, reverse_image_search=reverse_image_search)
    
    @staticmethod
    def _parse_response(content: str) -> ModelOutput | None:
        # Response expected to be JSON but may be wrapped with ```json ... ```
        json_start = content.find("{")
        json_end = content.rfind("}")
        json_string = content[json_start : json_end + 1]
        try:
            return ModelOutput.model_validate_json(json_data=json_string)
        except:
            pass
        return None
    
Vision.register(GPT4Vision)