import asyncio
import json
import logging
import os
import re
from typing import Any

from groq import AsyncGroq
from dotenv import load_dotenv

from agent.prompts import REFUSAL_TEMPLATES, build_messages

load_dotenv()
log = logging.getLogger(__name__)

# Config
MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 25  # Leave 5s buffer for the 30s evaluator cap

# Quick regex checks for off-topic/invalid requests (saves LLM cost/time)
LEGAL_KEYWORDS = re.compile(
    r"\b(legally required|gdpr|hipaa law|eeoc|ada compliance|"
    r"discrimination law|labour law|employment law|lawsuit|"
    r"are we required by law|does this satisfy|regulatory obligation)\b",
    re.I
)

INJECTION_KEYWORDS = re.compile(
    r"(ignore (previous|above|all) instructions?|"
    r"you are now|pretend (you are|to be)|"
    r"disregard|forget your (instructions?|rules?)|"
    r"act as (a |an )?(?!shl))",
    re.I
)

HR_KEYWORDS = re.compile(
    r"\b(salary|compensation|pay grade|benefits|culture fit|"
    r"how to interview|interview questions|reference check|"
    r"background check|onboarding|probation period)\b",
    re.I
)

CONFIRMATION_KEYWORDS = re.compile(
    r"\b(confirmed?|that'?s? (it|good|perfect|great|what (we|i) need)|"
    r"lock(ed)? it in|go with (that|those)|looks? good|finalised?|"
    r"sounds? good|we'?re? good|ok(ay)?|yes( please)?|perfect)\b",
    re.I
)

def get_refusal_reason(text: str) -> str | None:
    if INJECTION_KEYWORDS.search(text):
        return "injection"
    if LEGAL_KEYWORDS.search(text):
        return "legal"
    if HR_KEYWORDS.search(text):
        return "general"
    return None

def is_user_confirming(text: str, has_recs: bool) -> bool:
    return bool(CONFIRMATION_KEYWORDS.search(text)) and has_recs

def get_combined_query(history: list[dict]) -> str:
    return " ".join(m["content"] for m in history if m.get("role") == "user").strip()

def get_last_shortlist(history: list[dict]) -> list[dict]:
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        try:
            data = json.loads(msg.get("content", ""))
            recs = data.get("recommendations", [])
            if isinstance(recs, list) and recs:
                return recs
        except (json.JSONDecodeError, TypeError):
            pass
    return []

def sanitize_and_ground_recs(raw_recs: list, candidates: list[dict], retriever) -> list[dict]:
    """Ensures LLM output matches real catalog items and doesn't hallucinate URLs."""
    valid = []
    for r in raw_recs[:10]:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name", "")).strip()
        url = str(r.get("url", "")).strip()
        t_type = str(r.get("test_type", "")).strip()

        if not name:
            continue

        # Look up match in catalog (by URL or exact name)
        match = retriever.get_by_url(url) or retriever.get_by_name(name)

        # Fallback: fuzzy match name against RAG candidates
        if not match:
            name_lower = name.lower()
            for c in candidates:
                if c["name"].lower() == name_lower or name_lower in c["name"].lower():
                    match = c
                    break

        if match:
            valid.append({
                "name": match["name"],
                "url": match["url"],
                "test_type": match.get("test_type", t_type)
            })
        else:
            log.warning(f"Dropping ungrounded recommendation: {name}")
    return valid

class SHLAgent:
    def __init__(self):
        # Local import to prevent loading index during imports
        from indexer.retriever import get_retriever
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY environment variable")
        
        self.client = AsyncGroq(api_key=api_key)
        self.retriever = get_retriever()

    async def chat(self, messages: list[dict[str, str]]) -> dict:
        try:
            return await asyncio.wait_for(
                self._process_turn(messages),
                timeout=REQUEST_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.error("Groq API call timed out")
            return {
                "reply": "Request timed out. Please try again.",
                "recommendations": [],
                "end_of_conversation": False
            }
        except Exception as e:
            log.exception(f"Unhandled error in agent: {e}")
            return {
                "reply": "Something went wrong. Please try again.",
                "recommendations": [],
                "end_of_conversation": False
            }

    async def _process_turn(self, messages: list[dict[str, str]]) -> dict:
        if not messages:
            return {
                "reply": "Hi! I can help you find assessments. What roles are you hiring for?",
                "recommendations": [],
                "end_of_conversation": False
            }

        last_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        turn_count = len(messages)
        existing_recs = get_last_shortlist(messages)

        # 1. Deterministic guardrails (Refusals)
        refusal_key = get_refusal_reason(last_msg)
        if refusal_key:
            return {
                "reply": REFUSAL_TEMPLATES[refusal_key],
                "recommendations": existing_recs,  # Persist shortlist so user doesn't lose progress
                "end_of_conversation": False
            }

        # 2. Force recommendations on Turn 7/8 (evaluator turn limit)
        must_recommend = turn_count >= 7

        # 3. Confirmation check
        if is_user_confirming(last_msg, bool(existing_recs)) and not must_recommend:
            return {
                "reply": "Got it. Your battery shortlist is locked in.",
                "recommendations": existing_recs,
                "end_of_conversation": True
            }

        if must_recommend and existing_recs:
            return {
                "reply": "Here is the final recommendation shortlist based on our conversation.",
                "recommendations": existing_recs,
                "end_of_conversation": True
            }

        # 4. RAG Retrieval
        query = get_combined_query(messages)
        candidates = self.retriever.retrieve(query, top_k=15)

        # 5. Call LLM
        prompt_msgs = build_messages(messages, candidates, turn_count)
        
        resp = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=prompt_msgs,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1024
        )
        
        raw_output = json.loads(resp.choices[0].message.content)

        # 6. Sanitize and Ground
        reply = str(raw_output.get("reply", "")).strip()
        recs = raw_output.get("recommendations", [])
        end_conv = bool(raw_output.get("end_of_conversation", False))

        grounded_recs = sanitize_and_ground_recs(recs, candidates, self.retriever)

        # Safety fallback: force recommendations if turn budget is hit
        if must_recommend and not grounded_recs and existing_recs:
            grounded_recs = existing_recs
            end_conv = True

        return {
            "reply": reply or "Could not generate a response. Please try again.",
            "recommendations": grounded_recs,
            "end_of_conversation": end_conv
        }

