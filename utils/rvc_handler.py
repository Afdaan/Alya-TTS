import logging
import os
import asyncio
from pathlib import Path
from typing import Optional

from config.settings import (
    RVC_DEVICE, RVC_PITCH_CHANGE,
    RVC_F0_METHOD, RVC_INDEX_RATE, RVC_VOLUME_ENVELOPE, RVC_PROTECT,
    RVC_RESAMPLE_SR
)

logger = logging.getLogger(__name__)

class RVCHandler:
    """Handler for RVC voice conversion."""
    
    def __init__(self, model_path: Path, index_path: Path):
        self.model_path = Path(model_path)
        self.index_path = Path(index_path)
        self.rvc = None
        self.is_available = False
        self.device = RVC_DEVICE
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialize RVC components."""
        try:
            if not self.model_path.exists() or not self.index_path.exists():
                logger.error(f"❌ RVC model or index files missing: {self.model_path} or {self.index_path}")
                return
            
            import sys
            import importlib.util
            from pathlib import Path

            # Absolute path to libs
            current_file = Path(__file__).resolve()
            base_dir = current_file.parent.parent
            libs_path = base_dir / "libs"
            rvc_root = libs_path / "rvc_python"
            
            if not rvc_root.exists():
                logger.error(f"❌ rvc_python not found in {libs_path}")
                return

            if str(libs_path) not in sys.path:
                sys.path.insert(0, str(libs_path))
            
            # Helper to check for submodules
            if not (rvc_root / "lib" / "__init__.py").exists():
                logger.warning(f"⚠️ Warning: {rvc_root / 'lib' / '__init__.py'} missing. Attempting to fix...")
                (rvc_root / "lib" / "__init__.py").touch()

            # Force reload if it was already loaded from site-packages
            if 'rvc_python' in sys.modules:
                for mod in list(sys.modules.keys()):
                    if mod.startswith('rvc_python'):
                        del sys.modules[mod]
            
            # Log for debugging
            logger.info(f"🔍 RVC Root: {rvc_root}")
            
            import torch
            from rvc_python.infer import RVCInference

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                torch.set_num_threads(min(4, torch.get_num_threads()))
            self.device = device
            
            self.rvc = RVCInference(device=self.device)
            self.rvc.load_model(str(self.model_path))
            self.is_available = True
            logger.info(f"✅ RVC Handler ready on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RVC: {e}", exc_info=True)
            self.is_available = False
            
    async def convert_voice(self, audio_path: str, output_path: str) -> bool:
        """Convert voice using RVC model."""
        if not self.is_available or not self.rvc:
            return False
        
        if not os.path.exists(audio_path):
            return False
            
        try:
            # Run inference in thread pool with timeout
            await asyncio.wait_for(
                asyncio.to_thread(self.rvc.infer_file, audio_path, output_path),
                timeout=60.0
            )
            return os.path.exists(output_path)
        except Exception as e:
            logger.error(f"❌ RVC Conversion error: {e}")
            return False
