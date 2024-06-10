#
# app.py
#
# Noa assistant server application. Provides /mm endpoint.
#

from io import BytesIO
import os
import traceback
from typing import Annotated, List
import glob

import openai
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment
from fastapi import FastAPI, status, Form, UploadFile, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder

from models import Message, MultimodalRequest, MultimodalResponse
from vision.utils import process_image
from assistant import NewAssistant


####################################################################################################
# Configuration
####################################################################################################

EXPERIMENT_AI_PORT = os.environ.get('EXPERIMENT_AI_PORT',8000)
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", None)


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

@app.get('/health')
async def api_health():
    return {"status":200,"message":"running ok"}

MAX_FILES = 100
AUDIO_DIR = "audio"

def get_next_filename():
    existing_files = sorted(glob.glob(f"{AUDIO_DIR}/audio*.wav"))
    # if audio directory does not exist, create it
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    if len(existing_files) < MAX_FILES:
        return f"{AUDIO_DIR}/audio{len(existing_files) + 1}.wav"
    else:
        # All files exist, so find the oldest one to overwrite
        oldest_file = min(existing_files, key=os.path.getmtime)
        return oldest_file

async def assistant_response_generator(
    assistant: NewAssistant,
    prompt: str,
    flavor_prompt: str,
    message_history: List[Message] | None,
    image_bytes: bytes | None,
    location_address: str | None,
    local_time: str | None
):
    i = 0
    async for assistant_response_chunk in assistant.send_to_assistant(
        prompt=prompt,
        flavor_prompt=flavor_prompt,
        image_bytes=image_bytes,
        message_history=message_history,
        location_address=location_address,
        local_time=local_time
    ):
        include_user_prompt = i == 0 or assistant_response_chunk.stream_finished    # first and final chunk
        response_chunk = MultimodalResponse(
            user_prompt=prompt if include_user_prompt else "",
            response=assistant_response_chunk.response,
            image=assistant_response_chunk.image,
            token_usage_by_model=assistant_response_chunk.token_usage_by_model,
            capabilities_used=assistant_response_chunk.capabilities_used,
            timings=assistant_response_chunk.timings,
            debug_tools="",
            stream_finished=assistant_response_chunk.stream_finished
        )
        i += 1
        yield f"event: json\ndata:{response_chunk.model_dump_json()}\n\n"
    yield "event: end\ndata:{}\n\n"

@app.post("/mm")
async def api_mm(request: Request, mm: Annotated[str, Form()], audio : UploadFile = None, image: UploadFile = None):
    try:
        mm: MultimodalRequest = Checker(MultimodalRequest)(data=mm)
        print(mm)

        # Transcribe voice prompt if it exists
        voice_prompt = ""
        if audio:
            audio_bytes = await audio.read()
            if mm.testing_mode:
                #  save audio file
                # set timestamp
                # filepath = "audio.wav" + str(datetime.now().timestamp())
                filepath = get_next_filename()
                with open(filepath, "wb") as f:
                    f.write(audio_bytes)
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
        
        # Call the assistant and deliver the response
        try:
            response_chunks = []
            async for response_chunk in request.app.state.assistant.send_to_assistant(
                prompt=user_prompt,
                flavor_prompt=mm.noa_system_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                location_address=address,
                local_time=local_time
            ):
                response_chunks.append(response_chunk)
            assistant_response = response_chunks[-1]    # last chunk has the complete response
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

@app.post("/mm_stream")
async def api_mm(request: Request, mm: Annotated[str, Form()], audio : UploadFile = None, image: UploadFile = None):
    try:
        mm: MultimodalRequest = Checker(MultimodalRequest)(data=mm)
        print(mm)

        # Transcribe voice prompt if it exists
        voice_prompt = ""
        if audio:
            audio_bytes = await audio.read()
            if mm.testing_mode:
                #  save audio file
                # set timestamp
                # filepath = "audio.wav" + str(datetime.now().timestamp())
                filepath = get_next_filename()
                with open(filepath, "wb") as f:
                    f.write(audio_bytes)
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

       # Call the assistant and deliver the response
        try:
            generator = assistant_response_generator(
                assistant=request.app.state.assistant,
                prompt=user_prompt,
                flavor_prompt=mm.noa_system_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                location_address=address,
                local_time=local_time
            )
            return StreamingResponse(generator, media_type="text/event-stream")
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
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", action="store", default="San Francisco", help="Set location address used for all queries (e.g., \"San Francisco\")")
    options = parser.parse_args()
    app.state.openai_client = openai.AsyncOpenAI()
    app.state.assistant = NewAssistant(client=app.state.openai_client)
    uvicorn.run(app, host="0.0.0.0", port=int(EXPERIMENT_AI_PORT))