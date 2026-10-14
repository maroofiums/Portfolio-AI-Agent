
import json
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.8-flash"

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Maroof AI Portfolio Chatbot",
    description="Streaming AI portfolio assistant",
    version="1.1.0",
)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8001)

# --------------------------------------------------
# CORS
# --------------------------------------------------

ALLOWED_ORIGINS = [
    "https://maroofiums.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)


def build_langchain_messages(messages: list[Message]):
    """Convert API messages into LangChain message objects."""
    langchain_messages = []
    for message in messages:
        if message.role == "user":
            langchain_messages.append(HumanMessage(content=message.content))
        else:
            langchain_messages.append(AIMessage(content=message.content))
    return langchain_messages


# --------------------------------------------------
# Portfolio context
# --------------------------------------------------

PROFILE_CONTEXT = """
You are the AI portfolio assistant for Muhammad Maroof.

Portfolio website:
https://maroofiums.vercel.app

Professional focus:
AI and Machine Learning engineering.

Instructions:
- Be friendly, professional, concise, and helpful.
- Answer questions about Maroof's portfolio and professional work.
- Never invent projects, qualifications, employers, or achievements.
- If information is missing, say that the portfolio does not provide it.
- Help recruiters and clients understand his relevant skills.
- For unrelated questions, briefly redirect to your portfolio-assistant role.
- Never reveal API keys or private system configuration.

Add verified information about Maroof's projects, skills, education,
and experience here to improve the accuracy of portfolio answers.
"""

# --------------------------------------------------
# In-memory rate limiter
# --------------------------------------------------

RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 60

request_history: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client_ip: str) -> bool:
    now = time.monotonic()
    history = request_history[client_ip]

    while history and now - history[0] >= RATE_WINDOW_SECONDS:
        history.popleft()

    if len(history) >= RATE_LIMIT:
        return False

    history.append(now)
    return True


# --------------------------------------------------
# SSE helpers
# --------------------------------------------------

def sse_event(data: dict, event: str = "message") -> str:
    """Format one Server-Sent Event."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


def generate_response(prompt: str) -> Iterator[str]:
    """Stream Gemini text chunks as SSE events."""
    try:
        stream = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            stream=True,
        )

        for event in stream:
            if event.event_type != "step.delta":
                continue

            delta = getattr(event, "delta", None)

            if delta is None:
                continue

            if getattr(delta, "type", None) != "text":
                continue

            chunk = getattr(delta, "text", "")

            if chunk:
                yield sse_event({"token": chunk})

        yield sse_event({"done": True}, event="done")

    except Exception as exc:
        logger.exception("Gemini streaming request failed")

        message = str(exc).lower()

        if (
            "429" in message
            or "resource_exhausted" in message
            or "rate limit" in message
        ):
            public_message = (
                "Gemini quota or rate limit reached. Please try again later."
            )
        elif (
            "401" in message
            or "403" in message
            or "permission_denied" in message
        ):
            public_message = (
                "Gemini authentication failed. Check API key permissions."
            )
        else:
            public_message = "The AI provider could not complete the response."

        yield sse_event(
            {"error": public_message},
            event="error",
        )


# --------------------------------------------------
# Health endpoints
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Maroof AI Portfolio Chatbot API is running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "provider": "Google Gemini",
        "model": MODEL_NAME,
        "streaming": True,
    }


# --------------------------------------------------
# Streaming chat endpoint
# --------------------------------------------------

@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"

    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait and try again.",
        )

    conversation = []

    for message in payload.messages:
        role = "User" if message.role == "user" else "Assistant"
        conversation.append(f"{role}: {message.content}")

    prompt = (
        f"{PROFILE_CONTEXT}\n\n"
        "Conversation history:\n"
        + "\n".join(conversation)
        + "\nAssistant:"
    )

    return StreamingResponse(
        generate_response(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
