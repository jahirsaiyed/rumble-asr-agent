"""OpenAI text-to-speech streamed as 24 kHz mono 16-bit PCM."""
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

TTS_SAMPLE_RATE = 24_000
_CHUNK = 4800  # 100 ms
_STYLE = "Warm, calm, concise customer support agent. Speak at a natural conversational pace."


async def stream_speech(
    client: AsyncOpenAI, model: str, voice: str, text: str, instructions: bool = True
) -> AsyncIterator[bytes]:
    """Yield even-length PCM chunks so the client never splits a sample."""
    carry = b""
    extra = {"instructions": _STYLE} if instructions else {}
    async with client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=text, response_format="pcm", **extra
    ) as resp:
        async for chunk in resp.iter_bytes(_CHUNK):
            data = carry + chunk
            if len(data) % 2:
                data, carry = data[:-1], data[-1:]
            else:
                carry = b""
            if data:
                yield data
