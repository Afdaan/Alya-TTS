import sys
import os
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test-RVC")

async def test_rvc():
    try:
        # 1. Check paths
        root = Path(__file__).resolve().parent
        model_path = root / "alya_voice" / "alya.pth"
        index_path = root / "alya_voice" / "added_IVF777_Flat_nprobe_1_alya_v2.index"
        
        print(f"🔍 Checking model files in {root}/alya_voice...")
        if not model_path.exists():
            print(f"❌ Missing: {model_path}")
        else:
            print(f"✅ Found: {model_path.name}")
            
        if not index_path.exists():
            print(f"❌ Missing: {index_path}")
        else:
            print(f"✅ Found: {index_path.name}")
            
        # 2. Try to initialize RVCHandler
        print("\n⚡ Initializing RVC Handler...")
        from utils.rvc_handler import RVCHandler
        
        handler = RVCHandler(model_path, index_path)
        
        if handler.is_available:
            print("✅ RVC Handler: PASS")
            
            # 3. Simple test (if any wav exists in tmp)
            test_wav = root / "tmp" / "test_input.wav"
            if test_wav.exists():
                print(f"\n🎙️ Testing conversion of {test_wav.name}...")
                output_wav = root / "tmp" / "test_output.wav"
                success = await handler.convert_voice(str(test_wav), str(output_wav))
                if success:
                    print(f"✅ Conversion: PASS -> {output_wav}")
                else:
                    print("❌ Conversion: FAIL")
            else:
                print("\nℹ️ Skipping conversion test (no tmp/test_input.wav found)")
        else:
            print("❌ RVC Handler: FAIL")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rvc())
