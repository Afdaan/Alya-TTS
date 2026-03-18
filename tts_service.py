import os
# Optimizations for CPU Inference on limited-core servers (MUST be set before torch/numpy imports)
cpu_threads = os.getenv("RVC_CPU_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = cpu_threads
os.environ["MKL_NUM_THREADS"] = cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = cpu_threads

import sys
import asyncio
import logging
import gc
import time
from pathlib import Path

# Aggressive garbage collection for strict RAM limits (e.g., 4GB instances)
gc.set_threshold(100, 10, 10)
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from telegram import Bot
from dotenv import load_dotenv

# Ensure libs directory is in sys.path for local vendorized packages (e.g. rvc_python)
BASE_DIR = Path(__file__).resolve().parent
LIBS_PATH = BASE_DIR / "libs"
if str(LIBS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBS_PATH))

# Load environment variables
load_dotenv()

# Setup logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTS-Service")

app = FastAPI(title="Alya TTS Microservice (Resource Balanced)")

class TTSRequest(BaseModel):
    text: str
    voice_lang: str = "en"
    user_lang: str = "en"
    chat_id: int
    reply_to_message_id: Optional[int] = None
    bot_token: Optional[str] = None
    loading_message_id: Optional[int] = None

class ModelManager:
    """Manages lazy loading and auto-unloading of heavy RVC models."""
    def __init__(self):
        self._voice_processor = None
        self.last_active_time = 0
        from config.settings import TTS_IDLE_TIMEOUT
        self.idle_timeout = TTS_IDLE_TIMEOUT
        self.lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()  # Serialize inference to prevent CPU/RAM thrashing

    async def get_processor(self):
        async with self.lock:
            if self._voice_processor is None:
                logger.info("⚡ Cold Start: Loading RVC & Torch models into memory...")
                from utils.voice_processor import VoiceProcessor
                self._voice_processor = VoiceProcessor()
                from config.settings import DEFAULT_LANGUAGE
                self.default_lang = DEFAULT_LANGUAGE
            
            self.last_active_time = time.time()
            return self._voice_processor

    async def cleanup_if_idle(self):
        """Periodically checks if models should be unloaded."""
        while True:
            await asyncio.sleep(60) # Check every minute
            if self._voice_processor and (time.time() - self.last_active_time > self.idle_timeout):
                async with self.lock:
                    if self._voice_processor and (time.time() - self.last_active_time > self.idle_timeout):
                        logger.info("😴 Idle timeout reached. Unloading models to free up system resources...")
                        
                        try:
                            self._voice_processor.cleanup()
                        except Exception as e:
                            logger.error(f"Error during model cleanup: {e}")
                            
                        self._voice_processor = None
                        
                        gc.collect()
                        gc.collect()
                        
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except: pass
                        
                        logger.info("✨ Cleanup finished. System should notice reduced RAM usage shortly.")

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(model_manager.cleanup_if_idle())
    logger.info("🚀 TTS Service started and monitoring idle resource...")

async def process_tts_job(request: TTSRequest):
    """Background task to generate TTS and send to Telegram."""
    try:
        token = request.bot_token or os.getenv("BOT_TOKEN")
        if not token:
            logger.error("No bot token provided")
            return

        bot = Bot(token=token)
        
        processor = await model_manager.get_processor()
        
        # Use voice_lang; fallback to user_lang when voice_lang is unset (default 'en')
        target_lang = request.voice_lang
        if target_lang == "en" and request.user_lang and request.user_lang != "en":
            target_lang = request.user_lang
            logger.info(f"Using user_lang ({target_lang}) as fallback for default voice_lang")

        logger.info(f"Queueing voice generation for chat {request.chat_id} (lang: {target_lang})")
        
        # Lock inference to prevent CPU thread thrashing and memory OOM on concurrent requests
        async with model_manager.inference_lock:
            logger.info(f"Starting voice generation for chat {request.chat_id}")
            model_manager.last_active_time = time.time()  # Prevent idle timeout if request was queued for a long time
            voice_path = await processor.text_to_speech(request.text, target_lang)
            model_manager.last_active_time = time.time()  # Reset idle timer after generation finishes
        
        if not voice_path or not os.path.exists(voice_path):
            logger.error(f"Failed to generate voice for chat {request.chat_id}")
            return
            
        if request.loading_message_id:
            try:
                await bot.delete_message(chat_id=request.chat_id, message_id=request.loading_message_id)
            except Exception as e:
                logger.warning(f"Failed to delete loading message {request.loading_message_id}: {e}")

        with open(voice_path, 'rb') as vf:
            from config.settings import TTS_SEND_TIMEOUT
            await bot.send_voice(
                chat_id=request.chat_id,
                voice=vf,
                caption=f"🎙️ Alya's voice ({target_lang.upper()})",
                reply_to_message_id=request.reply_to_message_id,
                read_timeout=TTS_SEND_TIMEOUT,
                write_timeout=TTS_SEND_TIMEOUT,
                connect_timeout=60
            )
            
        if os.path.exists(voice_path):
            os.unlink(voice_path)
            
        logger.info(f"✅ Voice sent successfully to {request.chat_id}")
        
    except Exception as e:
        logger.error(f"❌ Error in processing job: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    status = "loaded" if model_manager._voice_processor else "dormant"
    return {
        "status": "ok", 
        "model_state": status,
        "seconds_since_last_use": int(time.time() - model_manager.last_active_time) if model_manager.last_active_time > 0 else 0
    }

@app.post("/tts")
async def dispatch_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_tts_job, request)
    return {"status": "accepted", "message": "TTS job received"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TTS_PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
