# Live support agent

Browser mic -> Rumble-2-Flash (local GPU) -> OpenAI chat + mock tools -> OpenAI TTS -> browser speaker.
Talking over the agent interrupts it.

## Install

Python 3.12. Dependencies are pinned in `requirements.txt` (`transformers` 5.x is required for this model).

```powershell
python -m venv .venv
# NVIDIA GPU (recommended): install the CUDA build of torch first
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

CPU-only machines (and Macs) can skip the CUDA line. The app then runs on CPU, which is slower.

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
