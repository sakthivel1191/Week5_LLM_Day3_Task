
import os
import uuid

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenRouter connection
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "openai/gpt-4o-mini"

# Create FastAPI application
app = FastAPI(
    title="Week 5 Day 3 - AI Chat API",
    description="FastAPI and Pydantic practice",
    version="1.0"
)


# --------------------------------
# PYDANTIC MODELS
# --------------------------------

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=2000
    )


class ChatResponse(BaseModel):

    reply: str
    status: str


# --------------------------------
# BONUS: MIDDLEWARE
# --------------------------------

@app.middleware("http")
async def add_request_id(request: Request, call_next):

    # Generate unique request ID
    request_id = str(uuid.uuid4())

    # Process request
    response = await call_next(request)

    # Add response header
    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------
# GET ROUTE
# --------------------------------

@app.get("/sessions/{session_id}")
def get_session(session_id: str):

    return {
        "session_id": session_id,
        "status": "Session found"
    }


# --------------------------------
# POST ROUTE
# --------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    # Validate using model_validate()
    validated_request = ChatRequest.model_validate(
        request.model_dump()
    )

    # Convert model into dictionary
    request_data = validated_request.model_dump()

    user_message = request_data["message"]

    # Send message to OpenRouter
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    return ChatResponse(
        reply=answer,
        status="success"
    )


# --------------------------------
# HOME ROUTE
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to Week 5 Day 3 FastAPI project"
    }
