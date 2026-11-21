"""LLM turn loop: streams text, calls tools, hands finished sentences to a speaker."""
import json
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from .chunker import SentenceChunker
from .tools import TOOL_SCHEMAS, run_tool

MAX_TOOL_ROUNDS = 4

Emit = Callable[[dict[str, Any]], Awaitable[None]]
Speak = Callable[[str], Awaitable[None]]


def system_prompt(company: str) -> str:
    return (
        f"You are a phone support agent for {company}. You are speaking out loud on a live call.\n"
        "- Reply in one to three short sentences. No lists, markdown, emojis or URLs.\n"
        "- Say numbers naturally, for example 'twenty nine ninety nine' or 'five to seven business days'.\n"
        "- Ask for one thing at a time. Confirm the order number before taking an action.\n"
        "- Use tools for order facts and actions, never invent order data.\n"
        "- Refunds above 100 dollars: do not refund, open a high priority ticket and say so.\n"
        "- The caller can interrupt you. If your last line is marked [interrupted], do not repeat it.\n"
        "- The transcript comes from speech recognition and may contain small errors."
    )


async def run_turn(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    emit: Emit,
    speak: Speak,
    spoken: list[str],
) -> None:
    """Append assistant/tool messages to `messages`. `spoken` collects emitted text
    so the caller can record a partial answer if the turn is cancelled."""
    for _ in range(MAX_TOOL_ROUNDS):
        stream = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, stream=True
        )
        chunker = SentenceChunker()
        text, calls = "", {}
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta.content:
                text += delta.content
                await emit({"type": "agent_delta", "text": delta.content})
                for sentence in chunker.feed(delta.content):
                    spoken.append(sentence)
                    await speak(sentence)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                slot["id"] = tc.id or slot["id"]
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["args"] += tc.function.arguments or ""
        for sentence in chunker.flush():
            spoken.append(sentence)
            await speak(sentence)

        if not calls:
            messages.append({"role": "assistant", "content": text})
            return
        ordered = [calls[i] for i in sorted(calls)]
        messages.append({
            "role": "assistant", "content": text or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                           for c in ordered],
        })
        for c in ordered:
            result = run_tool(c["name"], c["args"])
            await emit({"type": "tool", "name": c["name"], "args": c["args"], "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(result)})
