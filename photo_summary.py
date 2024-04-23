#
# photo_summary.py
#
# Test of summarizing a set of daily snapshots.
#

import base64
import os

import openai

MODEL = "gpt-4-turbo"

# Supply these photos and edit the timestamps. Photos should fit within 640x640 or 512x512 pixels.
timestamped_files = [ 
    ("9:00am", "0.jpg"), 
    ("10:00am", "1.jpg"),
    ("11:00am", "2.jpg"),
    ("12:00pm", "3.jpg"),
    ("1:00pm", "4.jpg"),
    ("2:00pm", "5.jpg"),
    ("4:30pm", "6.jpg"),
    ("6:30pm", "7.jpg")
]

SUMMARIZATION_PROMPT = """
Here are a series of images and the times they were taken at. To the best of your ability, analyze the sequence and summarize my day.
Don't be wishy washy, provide concise and confident responses. Give your response in the following format:

SUMMARY: Up to 5 sentences summarizing my day as a keepsake for my memories.
SUGGESTION: A helpful suggestion for me given what you've learned about me. One or two sentences. Assume the role of a helpful assistant.
"""

def load_file_base64(filepath: str) -> str:
    with open(file=filepath, mode="rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")

if __name__ == "__main__":
    client = openai.OpenAI()

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": SUMMARIZATION_PROMPT
            }
        ]
    }

    for timestamp, file in timestamped_files:
        image_base64 = load_file_base64(filepath=file)
        image_description = {
            "type": "text",
            "text": f"At {timestamp}:"
        }
        image_attachment = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
        message["content"].append(image_description)
        message["content"].append(image_attachment)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[ message ],
        max_tokens=4096
    )

    print(response.choices[0].message.content)

    
    

    