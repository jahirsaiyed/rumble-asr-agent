# Live support agent

Browser mic -> Rumble-2-Flash (speech-to-text, runs locally) -> Azure OpenAI or OpenAI chat with mock
support tools -> text-to-speech -> browser speaker. Talking over the agent interrupts it.
If cloud text-to-speech is unavailable, the browser's built-in voice reads the replies instead.

## Quick start

Requires Python 3.12, git, and Chrome or Edge. An NVIDIA GPU is recommended; CPU works but is slower.

```bash
git clone https://github.com/jahirsaiyed/rumble-asr-agent.git
cd rumble-asr-agent
```

**1. Create the environment and install**

Windows (PowerShell):
```powershell
python -m venv .venv
# NVIDIA GPU only: install the CUDA build of torch first (skip this line for CPU-only)
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:
```bash
python3.12 -m venv .venv
# NVIDIA GPU only (Linux): pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip install -r requirements.txt
```

**2. Configure credentials** (the key is never committed)

```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Edit `.env`:
- Azure OpenAI: set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and the names of your chat and
  text-to-speech **deployments** (`AZURE_OPENAI_CHAT_DEPLOYMENT`, `AZURE_OPENAI_TTS_DEPLOYMENT`).
- Plain OpenAI instead: delete the `AZURE_*` lines and set `OPENAI_API_KEY`
  (optionally `OPENAI_CHAT_MODEL`, `OPENAI_TTS_MODEL`).

Check that the deployments respond (Windows: `.venv\Scripts\python`):
```bash
.venv/bin/python check_azure.py
```
If the speech check fails, the app still works using the browser voice.

**3. Run**

```bash
.venv/bin/python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, wait for "ASR: Rumble on cuda" (or cpu), then press START CALL.
The first start downloads the speech model from Hugging Face, so it needs internet and takes longer.
Run only one server at a time: the model uses most of a 4 GB GPU.

Microphone access works on `localhost`. Opening the page from another machine over plain `http://`
is blocked by browsers and would need HTTPS. Use headphones to avoid the agent hearing itself.

## Layout

- `app/server.py`  WebSocket call session, interruption, latency metrics
- `app/asr.py`     Rumble-2-Flash wrapper
- `app/agent.py`   streaming LLM turn loop with tool calls
- `app/tools.py`   mock order/refund/ticket tools (replace with real systems)
- `app/tts.py`     text-to-speech streamed as PCM
- `static/index.html`  UI, voice-activity detection, playback, browser-voice fallback

Other settings (env or `.env`): `AZURE_OPENAI_API_VERSION`, `TTS_VOICE`, `ASR_LANGUAGE`, `COMPANY_NAME`.

## Test

```bash
.venv/bin/python -m pytest -q
```
