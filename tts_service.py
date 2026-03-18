import os
# Optimizations for CPU Inference on limited-core servers (MUST be set before torch/numpy imports)
cpu_threads = os.getenv("RVC_CPU_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = cpu_threads
os.environ["MKL_NUM_THREADS"] = cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = cpu_threads

# 1. Standard library imports
import sys
import asyncio
import logging
import gc
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

# 2. Third-party library imports
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from telegram import Bot
from dotenv import load_dotenv

# Aggressive garbage collection for strict RAM limits (e.g., 4GB instances)
gc.set_threshold(100, 10, 10)

# --- Global Subprocess State ---
_worker_processor = None

def _run_tts_worker(text: str, lang: str) -> Optional[str]:
    """Runs completely isolated inside a subprocess. Solves PyTorch memory leak strictly."""
    global _worker_processor
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if _worker_processor is None:
        from utils.voice_processor import VoiceProcessor
        import logging
        logging.getLogger().setLevel(logging.INFO)
        _worker_processor = VoiceProcessor()
        
    return loop.run_until_complete(_worker_processor.text_to_speech(text, lang))

# Ensure libs directory is in sys.path for local vendorized packages (e.g. rvc_python)
BASE_DIR = Path(__file__).resolve().parent
LIBS_PATH = BASE_DIR / "libs"
if str(LIBS_PATH) not in sys.path:
    sys.path.insert(0, str(LIBS_PATH))

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
    """Manages lazy loading and auto-unloading of heavy RVC models via Subprocess isolation."""
    def __init__(self):
        self._executor = None
        self.last_active_time = 0
        from config.settings import TTS_IDLE_TIMEOUT
        self.idle_timeout = TTS_IDLE_TIMEOUT
        self.lock = asyncio.Lock()
        self.inference_lock = asyncio.Lock()  # Serialize inference to prevent CPU/RAM thrashing

    async def get_executor(self):
        async with self.lock:
            if self._executor is None:
                logger.info("⚡ Cold Start: Spawning isolated TTS process (100% Leak-Proof Protocol)...")
                ctx = mp.get_context("spawn")
                self._executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
            
            self.last_active_time = time.time()
            return self._executor

    async def cleanup_if_idle(self):
        """Periodically checks if models should be unloaded."""
        while True:
            await asyncio.sleep(10) # Check every 10 seconds for aggressiveness
            if self._executor and (time.time() - self.last_active_time > self.idle_timeout):
                async with self.lock:
                    if self._executor and (time.time() - self.last_active_time > self.idle_timeout):
                        logger.info("😴 Idle timeout reached. Terminating TTS subprocess to guarantee 100% RAM release...")
                        
                        try:
                            # Fast shutdown terminates the subprocess directly, letting the OS reclaim all RAM
                            self._executor.shutdown(wait=False, cancel_futures=True)
                        except Exception as e:
                            logger.error(f"Error during executor shutdown: {e}")
                            
                        self._executor = None
                        
                        gc.collect()
                        logger.info("✨ Process terminated. RAM is completely returned to OS.")

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
        
        # Use voice_lang; fallback to user_lang when voice_lang is unset (default 'en')
        target_lang = request.voice_lang
        if target_lang == "en" and request.user_lang and request.user_lang != "en":
            target_lang = request.user_lang
            logger.info(f"Using user_lang ({target_lang}) as fallback for default voice_lang")

        logger.info(f"Queueing voice generation for chat {request.chat_id} (lang: {target_lang})")
        
        # Lock inference to prevent CPU thread thrashing and memory OOM on concurrent requests
        async with model_manager.inference_lock:
            executor = await model_manager.get_executor()
            logger.info(f"Starting voice generation for chat {request.chat_id} using Subprocess")
            model_manager.last_active_time = time.time()  # Prevent idle timeout if request was queued for a long time
            
            loop = asyncio.get_running_loop()
            voice_path = await loop.run_in_executor(
                executor, 
                _run_tts_worker, 
                request.text, 
                target_lang
            )
            
            model_manager.last_active_time = time.time()  # Reset idle timer after generation finishes
        
        if not voice_path or not os.path.exists(voice_path):
            logger.error(f"Failed to generate voice for chat {request.chat_id}")
            return
            
        if request.loading_message_id:
            try:
                await bot.delete_message(chat_id=request.chat_id, message_id=request.loading_message_id)
            except Exception as e:
                logger.warning(f"Failed to delete loading message {request.loading_message_id}: {e}")

        from telegram.error import RetryAfter, TelegramError
        
        try:
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
            logger.info(f"✅ Voice sent successfully to {request.chat_id}")
        except RetryAfter as e:
            logger.warning(f"⚠️ Telegram Flood Control hit ({e.retry_after}s). Dropping TTS job for chat {request.chat_id}.")
            try:
                msg = "🎙️ <i>Gomen, the voice service is currently rate limited by Telegram. Please try again later!</i>"
                await bot.send_message(
                    chat_id=request.chat_id,
                    text=msg,
                    parse_mode="HTML",
                    reply_to_message_id=request.reply_to_message_id
                )
            except Exception as notify_err:
                logger.error(f"Failed to send rate-limit notification: {notify_err}")
        except TelegramError as e:
            logger.error(f"❌ Failed to send voice message: {type(e).__name__} - {e}")
            try:
                msg = "🎙️ <i>Gomen, an error occurred while attempting to send the voice note.</i>"
                await bot.send_message(
                    chat_id=request.chat_id,
                    text=msg,
                    parse_mode="HTML",
                    reply_to_message_id=request.reply_to_message_id
                )
            except Exception:
                pass
        finally:
            if os.path.exists(voice_path):
                try:
                    os.unlink(voice_path)
                except OSError as e:
                    logger.warning(f"Failed to delete temp voice file: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error in processing job: {e}", exc_info=True)

@app.get("/health")
async def health_check():
    status = "loaded" if model_manager._executor else "dormant"
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
