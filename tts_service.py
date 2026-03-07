import os
import asyncio
import logging
import gc
import time
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from telegram import Bot
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTS-Service")

# Load environment variables
load_dotenv()

app = FastAPI(title="Alya TTS Microservice (Resource Balanced)")

class TTSRequest(BaseModel):
    text: str
    voice_lang: str = "en"
    user_lang: str = "id"
    chat_id: int
    reply_to_message_id: Optional[int] = None
    bot_token: Optional[str] = None

class ModelManager:
    """Manages lazy loading and auto-unloading of heavy RVC models."""
    def __init__(self):
        self._voice_processor = None
        self.last_active_time = 0
        self.idle_timeout = 300  # 5 minutes (adjust as needed)
        self.lock = asyncio.Lock()

    async def get_processor(self):
        async with self.lock:
            if self._voice_processor is None:
                logger.info("⚡ Cold Start: Loading RVC & Torch models into memory...")
                from utils.voice_processor import VoiceProcessor
                self._voice_processor = VoiceProcessor()
            
            self.last_active_time = time.time()
            return self._voice_processor

    async def cleanup_if_idle(self):
        """Periodically checks if models should be unloaded."""
        while True:
            await asyncio.sleep(60) # Check every minute
            if self._voice_processor and (time.time() - self.last_active_time > self.idle_timeout):
                async with self.lock:
                    if time.time() - self.last_active_time > self.idle_timeout:
                        logger.info("😴 Idle timeout reached. Unloading models to free up system resources...")
                        self._voice_processor = None
                        # Force garbage collection
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except: pass

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    # Start the idle cleanup monitor
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
        
        # Get processor (Trigger lazy load if needed)
        processor = await model_manager.get_processor()
        
        logger.info(f"Generating voice for chat {request.chat_id}...")
        voice_path = await processor.text_to_speech(request.text, request.voice_lang)
        
        if not voice_path or not os.path.exists(voice_path):
            logger.error(f"Failed to generate voice for chat {request.chat_id}")
            return
            
        with open(voice_path, 'rb') as vf:
            await bot.send_voice(
                chat_id=request.chat_id,
                voice=vf,
                caption=f"🎙️ Alya's voice ({request.voice_lang.upper()})",
                reply_to_message_id=request.reply_to_message_id
            )
            
        # Cleanup file after sending
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
