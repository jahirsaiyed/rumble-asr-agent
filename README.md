# Live support agent

Browser mic -> Rumble-2-Flash (local GPU) -> OpenAI chat + mock tools -> OpenAI TTS -> browser speaker.
Talking over the agent interrupts it.

## Run

```powershell
cd D:\Garage\Rumble-ASR\agent
# edit .env: set AZURE_OPENAI_API_KEY and your chat/TTS deployment names
..\.venv\Scripts\python check_azure.py    # verifies both deployments respond
..\.venv\Scripts\python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 (Chrome or Edge), wait for "ASR: Rumble on cuda", press START CALL.
Run only one server at a time: the model uses most of a 4 GB GPU.

## Layout

- `app/server.py`  WebSocket call session, interruption, latency metrics
- `app/asr.py`     Rumble-2-Flash wrapper
- `app/agent.py`   streaming LLM turn loop with tool calls
- `app/tools.py`   mock order/refund/ticket tools (replace with real systems)
- `app/tts.py`     OpenAI TTS streamed as PCM
- `static/index.html`  UI, voice-activity detection, playback

Settings (env or `.env`): `AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_TTS_DEPLOYMENT`, `TTS_VOICE`, `ASR_LANGUAGE`, `COMPANY_NAME`.

## Test

```powershell
..\.venv\Scripts\python -m pytest -q
```
