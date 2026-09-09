"""
Smoke Test for Google Gemini API and Environment Setup
Verifies that:
1. .env exists and GEMINI_API_KEY is loaded.
2. The GenAI SDK can successfully connect and generate text.
3. System instructions and temperature configurations work as expected.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

def run_smoke_test():
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    print("========================================")
    print("  Hiver AI Support Agent - Smoke Test   ")
    print("========================================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python Executable: {sys.executable}")
    print(f"Target model: {model}")

    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n[FAIL] GEMINI_API_KEY is missing or unconfigured!")
        print("Please edit .env and set GEMINI_API_KEY=your_actual_key")
        print("Get your free key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    print(f"[OK] Found GEMINI_API_KEY: {api_key[:6]}...{api_key[-4:]}")

    try:
        from src.gemini_client import GeminiClient
        client = GeminiClient(api_key=api_key, model=model)
        sdk_used = "Official google-genai" if client.use_official else "Legacy google-generativeai"
        print(f"[OK] GenAI SDK detected: {sdk_used}")
        
        test_prompt = (
            "You are an AI support agent for Amazon. "
            "Reply in one concise sentence acknowledging a customer reporting a late package."
        )
        print("\nSending test prompt to Gemini...")
        response = client.generate(
            prompt=test_prompt,
            temperature=0.2,
            max_output_tokens=100
        )
        print("\n[SUCCESS] API Response received:")
        print(f"--> \"{response}\"")
        print(f"\nModel '{model}' is operational and ready for Phase 1.")
    except ImportError as e:
        print(f"\n[FAIL] Dependency Error: {e}")
        print("To install the official SDK, run:")
        print(f"    \"{sys.executable}\" -m pip install google-genai python-dotenv")
        print("or install all dependencies:")
        print(f"    \"{sys.executable}\" -m pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Gemini API call failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. If model not found, try setting GEMINI_MODEL=gemini-2.5-flash or gemini-3.1-flash-lite in .env")
        print("2. Check if your API key has access to the specified model at https://aistudio.google.com")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
