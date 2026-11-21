"""Verify the configured chat and TTS deployments respond. Run: python check_azure.py"""
import asyncio

from app.config import load_settings


async def main() -> None:
    cfg = load_settings()
    print(f"provider={cfg.provider} endpoint={cfg.azure_endpoint or 'openai'} "
          f"api_version={cfg.azure_api_version}")
    if not cfg.api_key:
        raise SystemExit("API key missing in .env")
    client = cfg.make_client()

    try:
        r = await client.chat.completions.create(
            model=cfg.chat_model, messages=[{"role": "user", "content": "Say OK."}], max_completion_tokens=30)
        print(f"chat  OK  deployment={cfg.chat_model!r} -> {r.choices[0].message.content!r}")
    except Exception as exc:
        print(f"chat  FAIL deployment={cfg.chat_model!r}: {type(exc).__name__}: {exc}")

    try:
        total = 0
        async with client.audio.speech.with_streaming_response.create(
                model=cfg.tts_model, voice=cfg.tts_voice, input="Hello.", response_format="pcm") as resp:
            async for chunk in resp.iter_bytes():
                total += len(chunk)
        print(f"tts   OK  deployment={cfg.tts_model!r} -> {total} bytes of audio")
    except Exception as exc:
        print(f"tts   FAIL deployment={cfg.tts_model!r}: {type(exc).__name__}: {exc}")


asyncio.run(main())
