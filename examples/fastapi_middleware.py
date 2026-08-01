#!/usr/bin/env python3
"""Drop-in FastAPI dependency that guards a chat endpoint.

Requires: pip install fastapi uvicorn
Optional — only for the example, not a package dependency.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentinel_llm_guardrails import LLMGuardrails

app = FastAPI(title="guardrails-demo")
guards = LLMGuardrails(daily_request_limit=500, injection_score_threshold=0.45)


class ChatIn(BaseModel):
    user_id: str = "anonymous"
    message: str = Field(min_length=1, max_length=32_000)


class ChatOut(BaseModel):
    reply: str
    redacted: bool = False
    warnings: list[str] = []


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn) -> Any:
    checked = guards.check_input(body.message, user_id=body.user_id)
    if not checked.allowed:
        raise HTTPException(status_code=400, detail=checked.blocked_reason)

    prompt = checked.redacted_input or body.message
    # Replace with your model call:
    model_text = f"Echo: {prompt[:500]}"

    out = guards.check_output(model_text)
    if not out.allowed:
        raise HTTPException(status_code=502, detail=out.blocked_reason)

    return ChatOut(
        reply=model_text,
        redacted=checked.redacted_input is not None,
        warnings=list(checked.warnings) + list(out.warnings),
    )


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "stats": guards.get_stats()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
