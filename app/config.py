"""Runtime configuration, read once from the environment (.env supported).

Azure OpenAI is used when AZURE_OPENAI_ENDPOINT is set, otherwise plain OpenAI.
On Azure, the "model" values are your *deployment names*.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class Settings:
    provider: str  # "azure" | "openai"
    api_key: str
    azure_endpoint: str
    azure_api_version: str
    chat_model: str
    tts_model: str
    tts_voice: str
    tts_instructions: bool
    asr_model: str
    asr_language: str
    company: str

    def make_client(self) -> AsyncOpenAI:
        if self.provider == "azure":
            return AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version=self.azure_api_version,
            )
        return AsyncOpenAI(api_key=self.api_key)


def load_settings() -> Settings:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    azure = bool(endpoint)
    e = os.environ.get
    if azure:
        chat = e("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        tts = e("AZURE_OPENAI_TTS_DEPLOYMENT", "gpt-4o-mini-tts")
        key = e("AZURE_OPENAI_API_KEY", "").strip()
    else:
        chat = e("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        tts = e("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        key = e("OPENAI_API_KEY", "").strip()
    return Settings(
        provider="azure" if azure else "openai",
        api_key=key,
        azure_endpoint=endpoint,
        azure_api_version=e("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        chat_model=chat,
        tts_model=tts,
        tts_voice=e("TTS_VOICE", "alloy"),
        # only the gpt-4o-mini-tts family accepts style instructions
        tts_instructions="4o-mini-tts" in tts.lower(),
        asr_model=e("ASR_MODEL", "Reson8/Rumble-2-Flash"),
        asr_language=e("ASR_LANGUAGE", "English"),
        company=e("COMPANY_NAME", "Northwind"),
    )
