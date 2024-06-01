#
# app.py
#
# Noa assistant server application. Provides /mm endpoint.
#

import asyncio
from datetime import datetime
from io import BytesIO
import os
import traceback
from typing import Annotated, Dict, List, Tuple

import openai
import anthropic
import groq
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment
from fastapi import FastAPI, status, Form, UploadFile, Request
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder

from models import Capability, TokenUsage, SearchAPI, VisionModel, GenerateImageService, MultimodalRequest, MultimodalResponse, ExtractLearnedContextRequest, ExtractLearnedContextResponse
from web_search import WebSearch, DataForSEOWebSearch, SerpWebSearch, PerplexityWebSearch
from vision import Vision, GPT4Vision, ClaudeVision
from vision.utils import process_image
from generate_image import ReplicateGenerateImage
from assistant import Assistant, AssistantResponse, GPTAssistant, ClaudeAssistant, extract_learned_context


####################################################################################################
# Configuration
####################################################################################################

EXPERIMENT_AI_PORT = os.environ.get('EXPERIMENT_AI_PORT',8000)
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

async def transcribe(client: openai.AsyncOpenAI, audio_bytes: bytes) -> str:
    # Create a file-like object for Whisper API to consume
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    buffer = BytesIO()
    buffer.name = "voice.mp4"
    audio.export(buffer, format="mp4")

    # Whisper
    transcript = await client.audio.translations.create(
        model="whisper-1", 
        file=buffer,
    )
    return transcript.text

def validate_assistant_model(model: str | None, models: List[str]) -> str:
    """
    Ensures a valid model is selected.

    Parameters
    ----------
    model : str | None
        Model name to use.
    models : List[str]
        List of valid models. The first model is the default model.

    Returns
    -------
    str
        If the model name is in the list, returns it as-is, otherwise returns the first model in the
        list by default.
    """
    if model is None or model not in models:
        return models[0]
    return model

def get_assistant(app, mm: MultimodalRequest) -> Tuple[Assistant, str | None]:
    assistant_model = mm.assistant_model

    # Default assistant if none selected
    if mm.assistant is None or (mm.assistant not in [ "gpt", "claude", "groq" ]):
        return app.state.assistant, None    # None for assistant_model will force assistant to use its own internal default choice
    
    # Return assistant and a valid model for it
    if mm.assistant == "gpt":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "gpt-4o", "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-1106-preview" ])
        return GPTAssistant(client=app.state.openai_client), assistant_model
    elif mm.assistant == "claude":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229" ])
        return ClaudeAssistant(client=app.state.anthropic_client), assistant_model
    elif mm.assistant == "groq":
        assistant_model = validate_assistant_model(model=mm.assistant_model, models=[ "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it" ])
        return GPTAssistant(client=app.state.groq_client), assistant_model # Groq uses GPTAssistant
    
    # Should never fall through to here
    return None, ""

def get_web_search_provider(app, mm: MultimodalRequest) -> WebSearch:
    # Use provider specified in request options
    if mm.search_api == SearchAPI.SERP:
        return SerpWebSearch(save_to_file=options.save, engine=mm.search_engine.value, max_search_results=mm.max_search_results)
    elif mm.search_api == SearchAPI.DATAFORSEO:
        return DataForSEOWebSearch(save_to_file=options.save, max_search_results=mm.max_search_results)
    elif mm.search_api == SearchAPI.PERPLEXITY:
        return PerplexityWebSearch(api_key=PERPLEXITY_API_KEY)

    # Default provider
    return app.state.web_search

def get_vision_provider(app, mm: MultimodalRequest) -> Vision | None:
    # Use provider specified 
    if mm.vision in [VisionModel.GPT4O, VisionModel.GPT4Vision ]:
        return GPT4Vision(client=app.state.openai_client, model=mm.vision)
    elif mm.vision in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        return ClaudeVision(client=app.state.anthropic_client, model=mm.vision)
    
    # Default provider
    return app.state.vision

@app.get('/health')
async def api_health():
    return {"status":200,"message":"running ok"}

@app.post("/mm")
async def api_mm(request: Request, mm: Annotated[str, Form()], audio : UploadFile = None, image: UploadFile = None):
    try:
        mm: MultimodalRequest = Checker(MultimodalRequest)(data=mm)
        print(mm)

        # Transcribe voice prompt if it exists
        voice_prompt = ""
        if audio:
            audio_bytes = await audio.read()
            voice_prompt = await transcribe(client=request.app.state.openai_client, audio_bytes=audio_bytes)

        # Construct final prompt
        if mm.prompt is None or len(mm.prompt) == 0 or mm.prompt.isspace() or mm.prompt == "":
            user_prompt = voice_prompt
        else:
            user_prompt = mm.prompt + " " + voice_prompt

        # Image data
        image_bytes = (await image.read()) if image else None
        # preprocess image
        if image_bytes:
            image_bytes = process_image(image_bytes)
        # Location data
        address = mm.address

        # User's local time
        local_time = mm.local_time

        # Image generation (bypasses assistant altogether)
        if mm.generate_image != 0:
            if mm.generate_image_service == GenerateImageService.REPLICATE:
                generate_image = ReplicateGenerateImage()
                image_url = await generate_image.generate_image(
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
                    timings="",
                    debug_tools=""
                )

        # Get assistant tool providers
        web_search: WebSearch = get_web_search_provider(app=request.app, mm=mm)
        vision: Vision = get_vision_provider(app=request.app, mm=mm)
        
        # Call the assistant and deliver the response
        try:
            assistant, assistant_model = get_assistant(app=app, mm=mm)
            assistant_response: AssistantResponse = await assistant.send_to_assistant(
                prompt=user_prompt,
                noa_system_prompt=mm.noa_system_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                learned_context={},
                local_time=local_time,
                location_address=address,
                model=assistant_model,
                web_search=web_search,
                vision=vision,
                speculative_vision=mm.speculative_vision
            )

            return MultimodalResponse(
                user_prompt=user_prompt,
                response=assistant_response.response,
                image=assistant_response.image,
                token_usage_by_model=assistant_response.token_usage_by_model,
                capabilities_used=assistant_response.capabilities_used,
                total_tokens=0,
                input_tokens=0,
                output_tokens=0,
                timings=assistant_response.timings,
                debug_tools=assistant_response.debug_tools
            )
        except Exception as e:
            print(f"{traceback.format_exc()}")
            raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")

    except Exception as e:
        print(f"{traceback.format_exc()}")
        raise HTTPException(400, detail=f"{str(e)}: {traceback.format_exc()}")

@app.post("/extract_learned_context")
async def api_extract_learned_context(request: Request, params: Annotated[str, Form()]):
    try:
        params: ExtractLearnedContextRequest = Checker(ExtractLearnedContextRequest)(data=params)
        print(params)

        token_usage_by_model: Dict[str, TokenUsage] = {}

        # Perform extraction
        try:
            learned_context = await extract_learned_context(
                client=request.app.state.openai_client,
                message_history=params.messages,
                model="gpt-3.5-turbo-1106",
                existing_learned_context=params.existing_learned_context,
                token_usage_by_model=token_usage_by_model
            )

            return ExtractLearnedContextResponse(
                learned_context=learned_context,
                token_usage_by_model=token_usage_by_model
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
    parser.add_argument("--search-api", action="store", default="perplexity", help="Search API to use (perplexity, serp, dataforseo)")
    parser.add_argument("--assistant", action="store", default="gpt", help="Assistant to use (gpt, claude, groq)")
    parser.add_argument("--server", action="store_true", help="Start server")
    parser.add_argument("--image", action="store", help="Image filepath for image query")
    parser.add_argument("--vision", action="store", help="Vision model to use (gpt-4o, gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229)", default="gpt-4o")
    options = parser.parse_args()

    # AI clients
    app.state.openai_client = openai.AsyncOpenAI()
    app.state.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    app.state.groq_client = groq.AsyncGroq()

    # Instantiate a default web search provider
    app.state.web_search = None
    if options.search_api == "serp":
        app.state.web_search = SerpWebSearch(save_to_file=options.save, engine="google")
    elif options.search_api == "dataforseo":
        app.state.web_search = DataForSEOWebSearch(save_to_file=options.save)
    elif options.search_api == "perplexity":
        app.state.web_search = PerplexityWebSearch(api_key=PERPLEXITY_API_KEY)
    else:
        raise ValueError("--search-api must be one of: serp, dataforseo, perplexity")

    # Instantiate a default vision provider
    app.state.vision = None
    if options.vision in [ "gpt-4o", "gpt-4-vision-preview" ]:
        app.state.vision = GPT4Vision(client=app.state.openai_client, model=options.vision)
    elif VisionModel(options.vision) in [VisionModel.CLAUDE_HAIKU, VisionModel.CLAUDE_SONNET, VisionModel.CLAUDE_OPUS]:
        app.state.vision = ClaudeVision(client=app.state.anthropic_client, model=options.vision)
    else:
        raise ValueError("--vision must be one of: gpt-4o, gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229")

    # Instantiate a default assistant
    if options.assistant == "gpt":
        app.state.assistant = GPTAssistant(client=app.state.openai_client)
    elif options.assistant == "claude":
        app.state.assistant = ClaudeAssistant(client=app.state.anthropic_client)
    elif options.assistant == "groq":
        app.state.assistant = GPTAssistant(client=app.state.groq_client)
    else:
        raise ValueError("--assistant must be one of: gpt, claude, groq")

    # Load image if one was specified (for performing a test query)
    image_bytes = None
    if options.image:
        with open(file=options.image, mode="rb") as fp:
            image_bytes = fp.read()

    # Test query
    if options.query:
        async def run_query() -> str:
            return await app.state.assistant.send_to_assistant(
                prompt=options.query,
                image_bytes=image_bytes,
                message_history=[],
                learned_context={},
                local_time=datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),  # e.g., Friday, March 8, 2024, 11:54 AM
                location_address=options.location,
                model=None,
                web_search=app.state.web_search,
                vision=app.state.vision,

            )
        response = asyncio.run(run_query())
        print(response)

    # Run server
    if options.server:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(EXPERIMENT_AI_PORT))