import asyncio
from frame_sdk import Frame
from frame_sdk.display import Alignment, PaletteColors
from frame_sdk.camera import Quality, AutofocusType
import time
import math
import os
import openai
import base64
import json
from dotenv import load_dotenv
load_dotenv()
json_file = "frame_vision_test.json"
MODEL = "gpt-4o"

SYSTEM_MESSAGE = """
You are Noa, a smart personal AI assistant inside the user's AR smart glasses that answers all user
queries and questions. You have access to a photo from the smart glasses camera of what the user was
seeing at the time they spoke.

It is important that the user believes you can actually see. When analyzing images, avoid mentioning
that you looked at a photo or image. Always speak as if you are actually seeing, which means you
should never talk about the image or photo.

Make your responses precise. Respond without any preamble when giving translations, just translate
directly.
"""

async def call_vision_api(image_path, client: openai.AsyncOpenAI, query: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    user_message = {
        "role": "user",
        "content": [
            { "type": "text", "text": query },
            { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{image_base64}" } }
        ]
    }
    messages = [
        { "role": "system", "content": SYSTEM_MESSAGE },
        user_message
    ]

    # Call GPT
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

async def main():
    async with Frame() as f:
        response = []
        if os.path.exists(json_file):
            with open(json_file, "r") as fd:
                response = json.load(fd)
        openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Connected: {f.bluetooth.is_connected()}")
        while True:
            try:
                # let's get the current battery level
                print(f"Frame battery: {await f.get_battery_level()}%")
                print("Tap the Frame to continue...")
                await f.display.show_text("Tap the Frame to take a photo", align=Alignment.MIDDLE_CENTER)
                await f.motion.wait_for_tap()
                if not os.path.exists("captured"):
                    os.makedirs("captured")
                ts = math.floor(time.time())
                path = f"captured/frame-test-photo-{ts}.jpg"
                await f.camera.save_photo(path)
                print("=================Photo saved===============\n")
                await f.display.show_text("Photo saved", align=Alignment.MIDDLE_CENTER)
                start = time.time()
                ai_response = await call_vision_api(path, openai_client, "What do you see?")
                elapsed = time.time()-start
                print(f"AI response -- {elapsed:.3f} seconds:\n{ai_response}\n\n")
                response.append({"image_path":path,"response":ai_response, "latency": f"{elapsed:.3f}"})
                with open(json_file, "w") as fd:
                    json.dump(response, fd)
                await f.display.scroll_text(ai_response, 4, 0.14)
                await asyncio.sleep(3)
            except KeyboardInterrupt as e:
                break
            except Exception as e:
                print(e)
                break
    print("disconnected")



asyncio.run(main())