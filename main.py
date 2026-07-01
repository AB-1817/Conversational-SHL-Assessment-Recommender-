import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from agent.agent import SHLAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, val: str) -> str:
        if val not in ("user", "assistant", "system"):
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        return val

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, val: list[Message]) -> list[Message]:
        if not val:
            raise ValueError("Messages list cannot be empty")
        if val[-1].role != "user":
            raise ValueError("The last message in history must be from user")
        return val

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    log.info("Loading SHL agent model and index...")
    _agent = SHLAgent()
    yield
    log.info("Service shutdown completed")

app = FastAPI(
    title="SHL Advisor API",
    lifespan=lifespan
)

@app.exception_handler(Exception)
async def catch_all_errors(request: Request, exc: Exception):
    log.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again."}
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent is not ready yet")

    history = [{"role": m.role, "content": m.content} for m in request.messages]
    
    result = await _agent.chat(history)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"]
            )
            for r in result.get("recommendations", [])
            if isinstance(r, dict)
        ],
        end_of_conversation=bool(result.get("end_of_conversation", False))
    )

