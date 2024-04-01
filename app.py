#
# app.py
#
# Noa assistant server application. Provides /mm endpoint.
#

from datetime import datetime
from enum import Enum
from io import BytesIO
import os
import traceback
from typing import Dict, List, Optional, Annotated

import openai
import anthropic
from pydantic import BaseModel, ValidationError, Field
from pydub import AudioSegment
from fastapi import FastAPI, status, Form, UploadFile, Depends, Request
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder

from models import Message, Capability, SearchAPI, VisionModel, GenerateImageService, TokenUsage, MultimodalRequest, MultimodalResponse
from web_search import WebSearch, DataForSEOWebSearch, SerpWebSearch
from vision import Vision, GPT4Vision, ClaudeVision
from generate_image import GenerateImage, ReplicateGenerateImage
from assistant import Assistant, AssistantResponse, GPTAssistant, GPTCustomToolsAssistant, PerplexityAssistant


####################################################################################################
# Configuration
####################################################################################################

EXPERIMENT_AI_PORT = os.environ.get('EXPERIMENT_AI_PORT',8000)
SEARCH_API = os.environ.get('SEARCH_API','serp')
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", None)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)


####################################################################################################
# Server API 
####################################################################################################

app = FastAPI()

class Checker:
    def __init__(self, model: BaseModel):
        self.model = model

    def __call__(self, data: str = Form(...)):
        try:
            return self.model.model_validate_json(data)
        except ValidationError as e:
            raise HTTPException(
                detail=jsonable_encoder(e.errors()),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

def transcribe(client: openai.OpenAI, audio_bytes: bytes) -> str:
    # Create a file-like object for Whisper API to consume
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    buffer = BytesIO()
    buffer.name = "voice.mp4"
    audio.export(buffer, format="mp4")

    # Whisper
    transcript = client.audio.translations.create(
        model="whisper-1", 
        file=buffer,
    )
    return transcript.text

def get_web_search_provider(app, mm: MultimodalRequest) -> WebSearch:
    # Use provider specified in request options
    if mm.search_api == SearchAPI.SERP:
        return SerpWebSearch(save_to_file=options.save, engine=mm.search_engine.value, max_search_results=mm.max_search_results)
    elif mm.search_api == SearchAPI.DATAFORSEO:
        return DataForSEOWebSearch(save_to_file=options.save, max_search_results=mm.max_search_results)

    # Default provider
    return app.state.web_search

def get_vision_provider(app, mm: MultimodalRequest) -> Vision:
    # Use provider specified 
    if mm.vision == VisionModel.GPT4Vision:
        return GPT4Vision(client=app.state.openai_client)
    elif mm.vision in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        return ClaudeVision(client=app.state.anthropic_client, model=mm.vision)
    
    # Default provider
    return app.state.vision

@app.get('/health')
async def health():
    return {"status":200,"message":"running ok"}

@app.post("/mm")
async def mm(request: Request, mm: Annotated[str, Form()], audio : UploadFile = None, image: UploadFile = None):
    try:
        mm: MultimodalRequest = Checker(MultimodalRequest)(data=mm)
        print(mm)

        # Transcribe voice prompt if it exists
        voice_prompt = ""
        if audio:
            audio_bytes = await audio.read()
            voice_prompt = transcribe(client=request.app.state.openai_client, audio_bytes=audio_bytes)

        # Construct final prompt
        user_prompt = mm.prompt + " " + voice_prompt

        # Image data
        image_bytes = (await image.read()) if image else None

        # Location data
        address = mm.address

        # User's local time
        local_time = mm.local_time

        # Image generation (bypasses assistant altogether)
        if mm.generate_image != 0:
            if mm.generate_image_service == GenerateImageService.REPLICATE:
                generate_image = ReplicateGenerateImage()
                image_url = generate_image.generate_image(
                    query=user_prompt,
                    use_image=True,
                    image_bytes=image_bytes
                )
                return MultimodalResponse(
                    user_prompt=user_prompt,
                    response="",
                    image=image_url,
                    token_usage_by_model={},
                    capabilities_used=[Capability.IMAGE_GENERATION],
                    total_tokens=0,
                    input_tokens=0,
                    output_tokens=0,
                    debug_tools=""
                )

        # Get assistant tool providers
        web_search: WebSearch = get_web_search_provider(app=request.app, mm=mm)
        vision: Vision = get_vision_provider(app=request.app, mm=mm)
        
        # Call the assistant and deliver the response
        try:
            assistant: Assistant = app.state.assistant
            assistant_response: AssistantResponse = assistant.send_to_assistant(
                prompt=user_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                local_time=local_time,
                location_address=address,
                model=mm.assistant_model,
                web_search=web_search,
                vision=vision
            )

            return MultimodalResponse(
                user_prompt=user_prompt,
                response=assistant_response.response,
                image="",
                token_usage_by_model=assistant_response.token_usage_by_model,
                capabilities_used=assistant_response.capabilities_used,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                debug_tools=assistant_response.debug_tools
            )
        except Exception as e:
            print(f"{traceback.format_exc()}")
            raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")

    except Exception as e:
        print(f"{traceback.format_exc()}")
        raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")


####################################################################################################
# Program Entry Point
####################################################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store", help="Perform search query and exit")
    parser.add_argument("--location", action="store", default="San Francisco", help="Set location address used for all queries (e.g., \"San Francisco\")")
    parser.add_argument("--save", action="store", help="Save DataForSEO response object to file")
    parser.add_argument("--search-api", action="store", default=SEARCH_API, help="Search API to use (serp or dataforseo)")
    parser.add_argument("--assistant", action="store", default="gpt", help="Assistant to use (gpt or perplexity)")
    parser.add_argument("--server", action="store_true", help="Start server")
    parser.add_argument("--image", action="store", help="Image filepath for image query")
    parser.add_argument("--vision", action="store", help="Vision model to use (gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229)", default="claude-3-haiku-20240307")
    options = parser.parse_args()

    # AI clients
    app.state.openai_client = openai.OpenAI()
    app.state.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Instantiate a web search provider
    app.state.web_search = None
    if options.search_api == "serp":
        app.state.web_search = SerpWebSearch(save_to_file=options.save, engine="google")
    elif options.search_api == "dataforseo":
        app.state.web_search = DataForSEOWebSearch(save_to_file=options.save)
    else:
        raise ValueError("--search-api must be either 'serp' or 'dataforseo")

    # Instantiate a vision provider
    app.state.vision = None
    if options.vision == "gpt-4-vision-preview":
        app.state.vision = GPT4Vision(client=app.state.openai_client)
    elif VisionModel(options.vision) in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        app.state.vision = ClaudeVision(client=app.state.anthropic_client, model=options.vision)
    else:
        raise ValueError("--vision must be one of: gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229")

    # Instantiate an assistant
    if options.assistant == "gpt":
        app.state.assistant = GPTAssistant(client=app.state.openai_client)
    elif options.assistant == "gpt-custom-tools":
        app.state.assistant = GPTCustomToolsAssistant(client=app.state.openai_client)
    elif options.assistant == "perplexity":
        app.state.assistant = PerplexityAssistant(api_key=PERPLEXITY_API_KEY)
    else:
        raise ValueError("--assistant must be one of: gpt, gpt-custom-tools, perplexity")

    # Load image if one was specified (for performing a test query)
    image_bytes = None
    if options.image:
        with open(file=options.image, mode="rb") as fp:
            image_bytes = fp.read()

    # Test query
    if options.query:
        response = app.state.assistant.send_to_assistant(
            prompt=options.query,
            image_bytes=image_bytes,
            message_history=[],
            local_time=datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),  # e.g., Friday, March 8, 2024, 11:54 AM
            location_address=options.location,
            model=None,
            web_search=app.state.web_search,
            vision=app.state.vision,

        )
        print(response)

    # Run server
    if options.server:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(EXPERIMENT_AI_PORT))