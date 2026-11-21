"""FastAPI app: one WebSocket per call. Binary frames in = one utterance (16 kHz PCM16)."""
import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from .agent import run_turn, system_prompt
from .asr import RumbleASR
from .config import Settings, load_settings
from .tts import TTS_SAMPLE_RATE, stream_speech

log = logging.getLogger("server")
STATIC = Path(__file__).resolve().parents[1] / "static"


def _drop_incomplete_tool_round(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """After a cancel, remove a trailing assistant tool_calls message whose results are missing."""
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("tool_calls"):
            answered = sum(1 for m in msgs[i + 1:] if m["role"] == "tool")
            return msgs[:i] if answered < len(msgs[i]["tool_calls"]) else msgs
    return msgs


class CallSession:
    def __init__(self, ws: WebSocket, cfg: Settings, asr: RumbleASR, client: AsyncOpenAI) -> None:
        self.ws, self.cfg, self.asr, self.client = ws, cfg, asr, client
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(cfg.company)}]
        self.task: asyncio.Task | None = None
        self.tts_ok = True  # flips to False after the first TTS failure; browser voice takes over
        self._send_lock = asyncio.Lock()

    async def emit(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def _send_audio(self, data: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(data)

    async def cancel(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task
        self.task = None

    def start_utterance(self, pcm: bytes) -> None:
        self.task = asyncio.create_task(self._handle(pcm))

    async def _handle(self, pcm: bytes) -> None:
        t0 = time.perf_counter()
        metrics: dict[str, float] = {}
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        spoken: list[str] = []
        work = list(self.messages)
        speaker = asyncio.create_task(self._speaker(queue, t0, metrics))
        try:
            text = await self.asr.transcribe(pcm)
            metrics["asr_ms"] = (time.perf_counter() - t0) * 1000
            if not text:
                return
            await self.emit({"type": "transcript", "role": "user", "text": text})
            work.append({"role": "user", "content": text})
            t_llm = time.perf_counter()
            first = False

            async def emit(p: dict[str, Any]) -> None:
                nonlocal first
                if p["type"] == "agent_delta" and not first:
                    first = True
                    metrics["llm_first_ms"] = (time.perf_counter() - t_llm) * 1000
                await self.emit(p)

            await run_turn(self.client, self.cfg.chat_model, work, emit, queue.put, spoken)
            self.messages[:] = work
            await queue.put(None)
            await speaker
            await self.emit({"type": "agent_done"})
            await self.emit({"type": "metrics", **{k: round(v) for k, v in metrics.items()}})
        except asyncio.CancelledError:
            self.messages[:] = _drop_incomplete_tool_round(work)
            if spoken:
                self.messages.append(
                    {"role": "assistant", "content": " ".join(spoken) + " [interrupted]"})
            raise
        except Exception as exc:  # surfaced to the UI, never swallowed
            log.exception("turn failed")
            await self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            if not speaker.done():
                speaker.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await speaker

    async def _speaker(self, queue: asyncio.Queue, t0: float, metrics: dict[str, float]) -> None:
        while (sentence := await queue.get()) is not None:
            if self.tts_ok:
                try:
                    await self.emit({"type": "audio_start", "sample_rate": TTS_SAMPLE_RATE})
                    async for chunk in stream_speech(
                            self.client, self.cfg.tts_model, self.cfg.tts_voice, sentence,
                            self.cfg.tts_instructions):
                        metrics.setdefault("first_audio_ms", (time.perf_counter() - t0) * 1000)
                        await self._send_audio(chunk)
                    continue
                except Exception as exc:  # cancellation is a BaseException and passes through
                    log.warning("TTS failed, using browser voice: %s", exc)
                    self.tts_ok = False
                    await self.emit({"type": "tts_fallback",
                                     "message": f"{type(exc).__name__}: {exc}"[:300]})
            metrics.setdefault("first_audio_ms", (time.perf_counter() - t0) * 1000)
            await self.emit({"type": "speak_text", "text": sentence})


def create_app() -> FastAPI:
    cfg = load_settings()
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        def load() -> RumbleASR:
            asr = RumbleASR(cfg.asr_model, cfg.asr_language)
            asr.warmup()
            return asr
        # load in the background so /health and the UI are reachable while the model loads
        loader = asyncio.create_task(asyncio.to_thread(load))
        loader.add_done_callback(lambda t: setattr(app.state, "asr", t.result()))
        yield
        loader.cancel()

    app = FastAPI(title="Realtime Support Agent", lifespan=lifespan)
    app.state.asr = None

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"asr_ready": app.state.asr is not None,
                             "device": getattr(app.state.asr, "device", None),
                             "has_api_key": bool(cfg.api_key), "provider": cfg.provider,
                             "chat_model": cfg.chat_model, "tts_model": cfg.tts_model})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        if not cfg.api_key:
            await ws.send_text(json.dumps({"type": "error", "message": "API key is not set in agent/.env (AZURE_OPENAI_API_KEY)"}))
            await ws.close()
            return
        session = CallSession(ws, cfg, app.state.asr, cfg.make_client())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await session.cancel()
                    session.start_utterance(msg["bytes"])
                elif msg.get("text"):
                    if json.loads(msg["text"]).get("type") == "interrupt":
                        await session.cancel()
                        await session.emit({"type": "stopped"})
        except WebSocketDisconnect:
            pass
        finally:
            await session.cancel()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
