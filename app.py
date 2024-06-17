#
# app.py
#
# Noa assistant server application. Provides /mm endpoint.
#

import glob
from io import BytesIO
import os
import traceback
from typing import Annotated, Optional

from fastapi import FastAPI, status, Form, UploadFile, Request
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
import openai
from pydantic import BaseModel, ValidationError
from pydub import AudioSegment

from models import MultimodalRequest, MultimodalResponse
from util import process_image
from assistant import Assistant


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

@app.post("/mm")
async def api_mm(request: Request, mm: Annotated[str, Form()], audio : UploadFile = None, image: UploadFile = None):
    try:
        mm: MultimodalRequest = Checker(MultimodalRequest)(data=mm)
        print(mm)

        # Which OpenAI client to use: Brilliant's or one with user-supplied API key?
        user_openai_key: Optional[str] = mm.openai_key if mm.openai_key is not None and len(mm.openai_key) > 0 else None
        openai_client = openai.AsyncOpenAI(api_key=user_openai_key) if user_openai_key is not None else request.app.state.openai_client

        # Instantiate a new assistant if either Perplexity API key or OpenAI key is supplied
        user_perplexity_key: Optional[str] = mm.perplexity_key if mm.perplexity_key is not None and len(mm.perplexity_key) > 0 else None
        perplexity_key = PERPLEXITY_API_KEY if user_perplexity_key is None else user_perplexity_key
        assistant = request.app.state.assistant if user_openai_key is None and user_perplexity_key is None else Assistant(client=openai_client, perplexity_api_key=perplexity_key)

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
            voice_prompt = await transcribe(client=openai_client, audio_bytes=audio_bytes)

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
            assistant_response = await assistant.send_to_assistant(
                prompt=user_prompt,
                flavor_prompt=mm.noa_system_prompt,
                image_bytes=image_bytes,
                message_history=mm.messages,
                location_address=address,
                local_time=local_time
            )
            return MultimodalResponse(
                user_prompt=user_prompt,
                response=assistant_response.response,
                image=assistant_response.image,
                token_usage_by_model=assistant_response.token_usage_by_model,
                capabilities_used=assistant_response.capabilities_used,
                timings=assistant_response.timings
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
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--location", action="store", default="San Francisco", help="Set location address used for all queries (e.g., \"San Francisco\")")
    options = parser.parse_args()
    app.state.openai_client = openai.AsyncOpenAI()
    app.state.assistant = Assistant(client=app.state.openai_client, perplexity_api_key=PERPLEXITY_API_KEY)
    uvicorn.run(app, host="0.0.0.0", port=int(EXPERIMENT_AI_PORT))