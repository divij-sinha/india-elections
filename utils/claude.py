import base64
import json
import os
from io import BytesIO

from httpx import post
from PIL import Image


def talk_claude(messages: list):
    model = "claude-3-5-haiku-20241022"
    key = os.environ["CLAUDE_API_KEY"]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.0,
        }
    )

    resp = post("https://api.anthropic.com/v1/messages", data=body, headers=headers)
    return resp


def get_table_claude(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_str,
                    },
                },
                {
                    "type": "text",
                    "text": "What is the table in this image? DO NOT REPLY WITH ANYTHING ELSE. REPLY ONLY WITH THE TABLE IN THIS IMAGE. IT SHOULD BE IN CSV FORMAT. NO MORE AND NO LESS.",
                },
            ],
        }
    ]
    resp = talk_claude(messages=messages)
    return resp
