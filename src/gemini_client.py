"""
Unified Gemini API Client Wrapper
Prefers the official `google-genai` SDK (v1.0+), with automatic compatibility
fallback to `google-generativeai` if running in an environment where only the
legacy package is installed. Includes exponential backoff for free-tier rate limits.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SDK Auto-detection: prefer official google-genai
USE_OFFICIAL_SDK = False
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    USE_OFFICIAL_SDK = True
except ImportError:
    try:
        import google.generativeai as legacy_genai
        USE_OFFICIAL_SDK = False
    except ImportError:
        raise ImportError(
            "\n[MISSING DEPENDENCY] Google GenAI SDK is not installed in this Python environment.\n"
            "Please run:\n"
            "    pip install google-genai\n"
            "or install all pinned requirements:\n"
            "    pip install -r requirements.txt\n"
        )

# Default model recommendation for Google AI Studio free tier.
# gemini-3.1-flash-lite offers 15 RPM / 500 RPD (highest daily quota).
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


class GeminiClient:
    """
    Minimal, explicit wrapper around Google's Gemini models.
    Supports official `google-genai` and falls back gracefully to `google-generativeai`.
    Handles rate-limit retries (HTTP 429) with exponential backoff.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GEMINI_API_KEY is not set or is still the placeholder. "
                "Please add your actual key to .env or pass it to GeminiClient(api_key=...)."
            )
        self.model = model or DEFAULT_MODEL
        self.use_official = USE_OFFICIAL_SDK

        if self.use_official:
            self.client = genai.Client(api_key=self.api_key)
        else:
            legacy_genai.configure(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
        response_mime_type: Optional[str] = None,
        max_retries: int = 4,
        initial_backoff: float = 2.0,
    ) -> str:
        """
        Send a prompt to the Gemini model and return the generated text.

        Parameters:
            prompt: User/instruction prompt to send.
            system_instruction: Optional system instruction for grounding/persona.
            temperature: Sampling temperature (lower = more deterministic).
            max_output_tokens: Token cap on response.
            response_mime_type: E.g., 'application/json' for structured output.
            max_retries: Number of retry attempts on rate limit (HTTP 429) or transient API errors.
            initial_backoff: Initial sleep duration in seconds for exponential backoff.

        Returns:
            Extracted text content from the response.
        """
        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            try:
                if self.use_official:
                    # Official google-genai path
                    config_args: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    }
                    if system_instruction:
                        config_args["system_instruction"] = system_instruction
                    if response_mime_type:
                        config_args["response_mime_type"] = response_mime_type

                    config = types.GenerateContentConfig(**config_args)
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                    return ""
                else:
                    # Legacy google-generativeai fallback path
                    gen_model = legacy_genai.GenerativeModel(
                        model_name=self.model,
                        system_instruction=system_instruction,
                    )
                    gen_config: Dict[str, Any] = {
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    }
                    if response_mime_type:
                        gen_config["response_mime_type"] = response_mime_type

                    response = gen_model.generate_content(
                        prompt,
                        generation_config=gen_config,
                    )
                    if response and response.text:
                        return response.text.strip()
                    return ""

            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                    or "rate" in err_str.lower()
                )
                if attempt < max_retries:
                    sleep_time = backoff
                    logger.warning(
                        f"[Gemini API Retry] Attempt {attempt}/{max_retries}. "
                        f"{'Rate limit hit. ' if is_rate_limit else f'Error: {err_str[:80]}... '} "
                        f"Backing off for {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0
                else:
                    logger.error(f"[Gemini API Failed] Exhausted {max_retries} attempts: {e}")
                    raise


# Convenience function for quick functional calls
_default_client: Optional[GeminiClient] = None


def generate(prompt: str, **kwargs) -> str:
    """Convenience module-level function sharing a cached client instance."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client.generate(prompt, **kwargs)


if __name__ == "__main__":
    print("Testing GeminiClient initialization...")
    try:
        client = GeminiClient()
        sdk_type = "official google-genai" if client.use_official else "legacy google-generativeai"
        print(f"Initialized GeminiClient with model: {client.model} (SDK: {sdk_type})")
    except Exception as err:
        print(f"Note: {err}")
