import json
import os

from httpx import post


def talk_claude(messages: list):
    model = "claude-3-5-haiku-20241022"
    key = os.environ["CLAUDE_KEY"]
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
