"""Rumble-2-Flash speech recognition, loaded once and run off the event loop."""
import asyncio
import logging
import threading

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

log = logging.getLogger("asr")
SAMPLE_RATE = 16_000


class RumbleASR:
    def __init__(self, model_id: str, language: str) -> None:
        self._language = language
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        log.info("loading %s on %s", model_id, self._device)
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = (
            AutoModelForMultimodalLM.from_pretrained(model_id, dtype=self._dtype)
            .to(self._device)
            .eval()
        )
        self._lock = threading.Lock()  # one GPU job at a time
        self.device = self._device

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        with self._lock:
            inputs = self._processor.apply_transcription_request(
                audio=audio, language=self._language
            ).to(self._device, dtype=self._dtype)
            with torch.inference_mode():
                out = self._model.generate(**inputs, max_new_tokens=256)
            gen = out[:, inputs["input_ids"].shape[-1]:]
            return self._processor.decode(gen, return_format="transcription_only")[0].strip()

    async def transcribe(self, pcm16: bytes) -> str:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size < SAMPLE_RATE * 0.3:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def warmup(self) -> None:
        self._transcribe_sync(np.zeros(SAMPLE_RATE, dtype=np.float32))
