# Alya TTS Microservice

A standalone microservice for handling Text-to-Speech (TTS) using Retrieval-based Voice Conversion (RVC). 

This repository is designed as a backend component for:
👉 **[Alya-Bot-Telegram](https://github.com/Afdaan/Alya-Bot-Telegram)**

## Architecture

This service acts as a specialized worker to isolate heavy CPU/RAM tasks (RVC & Torch) from the main chat logic. By running this as a separate microservice:
- The main bot can use more recent library versions (e.g., Torch 2.6+ for NLP).
- Voice generation won't interfere with the responsiveness of AI text chat.
- It features **Resource Optimization**: Models are only loaded into memory when needed and automatically unloaded after 5 minutes of inactivity.

## Setup & Execution

1. **Navigate to this folder:**
   ```bash
   cd Alya-TTS
   ```

2. **Create a new Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Venv:**
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. **Install Dependencies:**
   If you are on Windows, it is recommended to downgrade `pip` to version 24.0 to avoid build errors with `fairseq`:
   ```bash
   python -m pip install pip==24.0
   pip install -r requirements.txt
   ```
   *Note: If you still encounter `fairseq` installation errors, ensure you are using Python 3.9 or 3.10 for better compatibility.*

5. **Start the Service:**
   ```bash
   python tts_service.py
   ```

## Voice Model Setup

Before running the service, you need to set up the RVC voice models.
Please follow the instructions in:
👉 **[setup/VOICE_MODEL_SETUP.md](setup/VOICE_MODEL_SETUP.md)**

You can also use the automatic setup scripts:
- **Windows**: `powershell ./setup/setup_voice_models.ps1`
- **Linux**: `bash ./setup/setup_voice_models.sh`

## Configuration

The service reads the `.env` file in this directory to access the `TELEGRAM_BOT_TOKEN`.
By default, it runs on port **5001**. You can override this in the `.env` file by adding `TTS_PORT=your_port_number`.

Ensure the main bot's `.env` has:
`TTS_SERVICE_URL=http://localhost:5001`

## API Endpoints

- `GET /health` - Check service status and model state (`dormant` vs `loaded`).
- `POST /tts` - Dispatch a TTS generation job.

## Features

- **Lazy Loading**: RVC models are loaded on the first request (Cold Start).
- **Auto-Unload**: Automatically frees up RAM/VRAM after 5 minutes of idle time.
- **Direct Delivery**: Sends voice notes directly to Telegram chat IDs.
