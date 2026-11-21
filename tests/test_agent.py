import json
from types import SimpleNamespace as NS

import pytest

from app.agent import run_turn
from app.chunker import SentenceChunker
from app.server import _drop_incomplete_tool_round
from app.tools import run_tool


def chunk_all(parts):
    c, out = SentenceChunker(), []
    for p in parts:
        out += c.feed(p)
    return out + c.flush()


def test_chunker_splits_streamed_sentences():
    parts = ["Sure, I can ch", "eck that order for you. ", "It is in transit", " and arrives Thursday."]
    assert chunk_all(parts) == ["Sure, I can check that order for you.", "It is in transit and arrives Thursday."]


def test_chunker_does_not_flush_tiny_first_sentence():
    assert chunk_all(["Hi. How can I help you with your order today?"]) == [
        "Hi. How can I help you with your order today?"]


def test_chunker_flushes_trailing_text_without_punctuation():
    assert chunk_all(["One moment please"]) == ["One moment please"]


def test_tool_lookup_found_and_missing():
    assert run_tool("lookup_order", '{"order_number": "48,213"}')["status"] == "in transit"
    assert run_tool("lookup_order", '{"order_number": "1"}')["found"] is False


def test_tool_refund_rules():
    assert run_tool("issue_refund", '{"order_number":"50977","amount":29.99,"reason":"dup"}')["ok"]
    assert not run_tool("issue_refund", '{"order_number":"50977","amount":500,"reason":"x"}')["ok"]
    assert not run_tool("issue_refund", '{"order_number":"0","amount":5,"reason":"x"}')["ok"]


def test_tool_bad_input_never_raises():
    assert not run_tool("nope", "{}")["ok"]
    assert not run_tool("lookup_order", "{not json")["ok"]
    assert not run_tool("lookup_order", '{"wrong": 1}')["ok"]


def test_drop_incomplete_tool_round():
    calls = {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]}
    base = [{"role": "user", "content": "hi"}]
    assert _drop_incomplete_tool_round(base + [calls, {"role": "tool"}]) == base
    full = base + [calls, {"role": "tool"}, {"role": "tool"}]
    assert _drop_incomplete_tool_round(full) == full


def _event(content=None, tool=None):
    delta = NS(content=content, tool_calls=[tool] if tool else None)
    return NS(choices=[NS(delta=delta)])


def _tc(index, id_=None, name=None, args=None):
    return NS(index=index, id=id_, function=NS(name=name, arguments=args))


class FakeClient:
    def __init__(self, rounds):
        self._rounds = iter(rounds)
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **_):
        events = next(self._rounds)

        async def gen():
            for e in events:
                yield e
        return gen()


@pytest.mark.asyncio
async def test_run_turn_calls_tool_then_speaks_answer():
    rounds = [
        [_event(tool=_tc(0, "c1", "lookup_order", '{"order_')), _event(tool=_tc(0, None, None, 'number":"48213"}'))],
        [_event("Your order is in transit "), _event("and arrives Thursday. Anything else?")],
    ]
    emitted, spoken_out, spoken = [], [], []

    async def emit(p): emitted.append(p)
    async def speak(s): spoken_out.append(s)

    msgs = [{"role": "user", "content": "where is 48213"}]
    await run_turn(FakeClient(rounds), "m", msgs, emit, speak, spoken)

    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(msgs[2]["content"])["status"] == "in transit"
    assert spoken_out == ["Your order is in transit and arrives Thursday.", "Anything else?"]
    assert any(e["type"] == "tool" and e["name"] == "lookup_order" for e in emitted)


@pytest.mark.asyncio
async def test_speaker_falls_back_to_browser_voice_when_tts_fails():
    import asyncio
    from app.config import Settings
    from app.server import CallSession

    sent = []

    class WS:
        async def send_text(self, t): sent.append(json.loads(t))
        async def send_bytes(self, b): sent.append("audio")

    class Boom:
        def __call__(self, **_): raise RuntimeError("404 deployment missing")

    client = NS(audio=NS(speech=NS(with_streaming_response=NS(create=Boom()))))
    cfg = Settings("azure", "k", "https://x", "v", "chat", "tts", "alloy", True, "asr", "English", "Co")
    s = CallSession(WS(), cfg, None, client)
    q: asyncio.Queue = asyncio.Queue()
    for sentence in ("First sentence here.", "Second one.", None):
        await q.put(sentence)
    metrics: dict = {}
    await s._speaker(q, 0.0, metrics)

    kinds = [m["type"] for m in sent if isinstance(m, dict)]
    assert kinds.count("tts_fallback") == 1  # reported once, not per sentence
    assert [m["text"] for m in sent if isinstance(m, dict) and m["type"] == "speak_text"] == [
        "First sentence here.", "Second one."]
    assert "first_audio_ms" in metrics
