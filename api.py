from fastapi import FastAPI, Request, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Any, Generator
from poe_api_wrapper import PoeApi
import helpers as hh
import json
import time
import asyncio
import random

app = FastAPI()
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")
async def chat_completions(request: Request):
    data = await request.json()
    streaming = data.get("stream") if "stream" in data else False
    model = data.get("model") if "model" in data else "gpt-3.5-turbo"
    messages = data.get("messages")

    # Validate messages format
    if not hh.validate_messages_format(messages):
        raise HTTPException(status_code=400, detail="Invalid messages format.")
    response = None
    while not response: # this may be stupid
        response = await poe_handler(model, messages) 
    return streaming_response(streaming, response, model)

async def poe_handler(model, messages):
    tokens = [] # fat list of tokens, needs someone to grab them all from the freegpt4 server lol
    try:
        t = random.choice(tokens)
        client = PoeApi(t)  
        bot = ""
        main_request = messages[-1]["content"]
        rest_string = hh.stringify_messages(messages=messages)
        
        if "claude" in model:
            if (len(rest_string) > 33333):
                bot = "a2_100k"
                rest_string = await hh.progressive_summarize_text(
                    rest_string, min(len(rest_string), 100000) # bullshitting values as im not sure of token->characters conversion
                )
            else:
                bot = "a2"
                rest_string = await hh.progressive_summarize_text(
                    rest_string, min(len(rest_string), 25000) # bullshitting values as im not sure of token->characters conversion
                )
        elif "4" in model:
            bot = "mistral-medium"
            rest_string = await hh.progressive_summarize_text(
                    rest_string, min(len(rest_string), 50000) # bullshitting values as im not sure of token->characters conversion
                )
        elif "3" in model:
            bot = "gemini-pro"
            rest_string = await hh.progressive_summarize_text(
                    rest_string, min(len(rest_string), 30000) # bullshitting values as im not sure of token->characters conversion
                )
        else:
            pass # this should not happen xP
        
        message = f"IGNORE PREVIOUS MESSAGES.\n\nYour current message context: {rest_string}\n\nThe most recent message: {main_request}\n\n"
        # Non-streamed example:
        for chunk in client.send_message(bot, message):
            pass
        return chunk["text"]
    except:
        return None
    
    

async def streaming_response(
    streaming: bool, response: str, model: str
) -> Generator[str, Any, None]:
    id_val = hh.generate_random_string(length=28)
    t = int(time.time())
    if not streaming:
        data = {
            "id": f"chatcmpl-{id_val}",
            "object": "chat.completion",
            "created": t,
            "model": model,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "choices": [
                {
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
        }
        yield json.dumps(data, separators=(",", ":"))
    else:
        async for token in hh.chunk_string(response, 6):
            data = {
                "id": f"chatcmpl-{id_val}",
                "object": "chat.completion.chunk",
                "created": t,
                "model": model,
                "choices": [
                    {
                        "delta": {"content": f"{token}"},
                        "index": 0,
                        "finish_reason": None,
                    },
                ],
            }
            yield "data: %s\n\n" % json.dumps(data, separators=(",", ":"))
            await asyncio.sleep(0.002)
        end_completion_data = {
            "id": f"chatcmpl-{id_val}",
            "object": "chat.completion.chunk",
            "created": t,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        content = json.dumps(end_completion_data, separators=(",", ":"))
        yield f"data: {content}\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=6969, log_level="info")