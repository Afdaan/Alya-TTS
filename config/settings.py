"""
Configuration settings for Alya TTS Microservice.
Only contains settings relevant to voice processing and RVC.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# General
DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Voice Model
VOICE_ENABLED: bool = os.getenv("VOICE_ENABLED", "true").lower() == "true"
VOICE_MODEL_DIR: str = os.getenv("VOICE_MODEL_DIR", "alya_voice")
VOICE_MODEL_PATH: str = os.path.join(VOICE_MODEL_DIR, "alya.pth")
VOICE_INDEX_PATH: str = os.path.join(VOICE_MODEL_DIR, "added_IVF777_Flat_nprobe_1_alya_v2.index")

# RVC Engine
RVC_ENABLED: bool = os.getenv("RVC_ENABLED", "true").lower() == "true"
RVC_DEVICE: str = os.getenv("RVC_DEVICE", "cpu")
RVC_CPU_THREADS: int = int(os.getenv("RVC_CPU_THREADS", "4"))
RVC_IS_HALF: bool = os.getenv("RVC_IS_HALF", "false").lower() == "true"
RVC_PITCH_CHANGE: int = int(os.getenv("RVC_PITCH_CHANGE", "0"))
RVC_F0_METHOD: str = os.getenv("RVC_F0_METHOD", "rmvpe")
RVC_INDEX_RATE: float = float(os.getenv("RVC_INDEX_RATE", "0.75"))
RVC_VOLUME_ENVELOPE: float = float(os.getenv("RVC_VOLUME_ENVELOPE", "1.0"))
RVC_PROTECT: float = float(os.getenv("RVC_PROTECT", "0.33"))
RVC_RESAMPLE_SR: int = int(os.getenv("RVC_RESAMPLE_SR", "0"))
RVC_QUEUE_SIZE: int = int(os.getenv("RVC_QUEUE_SIZE", "2"))

# TTS Service
TTS_PORT: int = int(os.getenv("TTS_PORT", "5001"))
TTS_IDLE_TIMEOUT: int = int(os.getenv("TTS_IDLE_TIMEOUT", "60"))  # Unload from RAM after 60s of inactivity
TTS_SEND_TIMEOUT: int = int(os.getenv("TTS_SEND_TIMEOUT", "120"))