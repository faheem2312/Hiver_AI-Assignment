"""
Gemini-Powered Intent Classifier for @AmazonHelp
=================================================
Classifies incoming customer tweets into the fixed 8-intent taxonomy.
Features:
1. Grounded in Intent enum (src/intents.py) as single source of truth.
2. Few-shot examples drawn strictly from training/grounding corpus (zero golden-set leakage).
3. Structured JSON output enforcement (intent, confidence, reasoning).
4. Exponential backoff and retry for rate limits & malformed outputs.
5. Disk caching to conserve Gemini API quota across development and evaluation.
"""

import os
import re
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent
from src.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "classification_cache.json"


# =====================================================================
# 1. Few-Shot Exemplars (Drawn Strictly from Grounding Corpus)
# =====================================================================
FEW_SHOT_EXAMPLES = [
    {
        "text": "@AmazonHelp My order #112-9847291 tracker says 'out for delivery' since yesterday morning with no update!",
        "intent": Intent.ORDER_TRACKING_DELAY.value,
        "confidence": 0.98,
        "reasoning": "Customer inquiring about tracking status of delayed out-for-delivery parcel."
    },
    {
        "text": "@AmazonHelp Dropped off my return at UPS 5 days ago. When will the refund hit my bank statement?",
        "intent": Intent.REFUND_RETURN_INQUIRY.value,
        "confidence": 0.96,
        "reasoning": "Customer asking about return processing and refund credit timeline."
    },
    {
        "text": "@AmazonHelp ceramic bowl arrived completely shattered into pieces! Glass was everywhere in the box!",
        "intent": Intent.DAMAGED_WRONG_ITEM.value,
        "confidence": 0.99,
        "reasoning": "Report of physically broken merchandise upon delivery."
    },
    {
        "text": "@AmazonHelp Was charged $14.99 for Prime today but never authorized auto-renewal. Please cancel immediately.",
        "intent": Intent.PRIME_MEMBERSHIP_BILLING.value,
        "confidence": 0.97,
        "reasoning": "Dispute over unexpected Prime subscription auto-renewal charge."
    },
    {
        "text": "@AmazonHelp I am not receiving the OTP 2FA verification code to log in to my account. Tried 5 times!",
        "intent": Intent.ACCOUNT_LOGIN_SECURITY.value,
        "confidence": 0.98,
        "reasoning": "Customer experiencing Two-Factor Authentication login block."
    },
    {
        "text": "@AmazonHelp Fire Stick screen is completely frozen on the logo and the remote will not respond.",
        "intent": Intent.PRODUCT_TECH_SUPPORT.value,
        "confidence": 0.95,
        "reasoning": "Hardware/device technical troubleshooting request."
    },
    {
        "text": "@AmazonHelp Your delivery van driver smashed into my gate causing $2000 in damage and drove off! Calling my lawyer!",
        "intent": Intent.ESCALATION_HIGH_RISK.value,
        "confidence": 0.99,
        "reasoning": "Severe real-world property damage and legal action threat."
    }
]


class IntentClassifier:
    """Classifies customer tweets into fixed intents using Gemini Flash."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_cache: bool = True):
        self.client = GeminiClient(api_key=api_key, model=model)
        self.use_cache = use_cache
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self.valid_intents = set(Intent.list_all())

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads cached classification results from disk."""
        if self.use_cache and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load classification cache: {e}")
        return {}

    def _save_cache(self):
        """Persists classification cache to disk."""
        if self.use_cache:
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not save classification cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Returns normalized MD5 hash for query text."""
        normalized = " ".join(text.strip().lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _build_prompt(self, text: str) -> str:
        """Constructs few-shot structured prompt."""
        taxonomy_text = Intent.get_prompt_taxonomy()

        few_shot_str = "\n\n".join([
            f"Customer Tweet: \"{ex['text']}\"\n"
            f'{{\n  "intent": "{ex["intent"]}",\n  "confidence": {ex["confidence"]},\n  "reasoning": "{ex["reasoning"]}"\n}}'
            for ex in FEW_SHOT_EXAMPLES
        ])

        prompt = f"""You are an expert intent classifier for @AmazonHelp customer support on Twitter.
Analyze the incoming customer tweet and classify it into EXACTLY ONE of the allowed intent categories below.

### Allowed Intent Taxonomy:
{taxonomy_text}

### Reference Examples:
{few_shot_str}

### Instructions:
1. Select the single best intent matching the customer's core objective.
2. Assign a confidence score between 0.00 and 1.00.
3. Return ONLY a valid JSON object with keys: "intent", "confidence", "reasoning".
4. Do NOT add preamble, conversational filler, or markdown fences other than the JSON itself.

Customer Tweet to Classify:
"{text}"
"""
        return prompt

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extracts and validates JSON from model response."""
        # Strip markdown code fences if present
        clean_text = re.sub(r"^```(?:json)?\s*", "", response_text.strip(), flags=re.IGNORECASE)
        clean_text = re.sub(r"\s*```$", "", clean_text)
        clean_text = clean_text.strip()

        try:
            data = json.loads(clean_text)
            intent = data.get("intent", "").strip()
            confidence = float(data.get("confidence", 0.0))
            reasoning = data.get("reasoning", "").strip()

            # Validate intent against authoritative enum
            if intent not in self.valid_intents:
                # Attempt case-insensitive match
                for valid in self.valid_intents:
                    if valid.lower() == intent.lower():
                        intent = valid
                        break

            if intent in self.valid_intents:
                return {
                    "intent": intent,
                    "confidence": min(1.0, max(0.0, confidence)),
                    "reasoning": reasoning
                }
        except Exception as e:
            logger.debug(f"JSON parsing error: {e} on raw response: '{response_text}'")
        return None

    def classify(self, text: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Classifies a customer tweet into an Intent with confidence and reasoning.
        Returns:
            {"intent": str, "confidence": float, "reasoning": str, "cached": bool}
        """
        cache_key = self._get_cache_key(text)
        if self.use_cache and cache_key in self.cache:
            cached_res = self.cache[cache_key].copy()
            cached_res["cached"] = True
            return cached_res

        prompt = self._build_prompt(text)

        for attempt in range(1, max_retries + 1):
            try:
                response_text = self.client.generate(
                    prompt=prompt,
                    temperature=0.0,  # Deterministic output for classification
                    max_output_tokens=250,
                    response_mime_type="application/json"
                )
                parsed = self._parse_json_response(response_text)
                if parsed:
                    result = {
                        "intent": parsed["intent"],
                        "confidence": parsed["confidence"],
                        "reasoning": parsed["reasoning"],
                        "cached": False
                    }
                    if self.use_cache:
                        self.cache[cache_key] = result
                        self._save_cache()
                    return result

                logger.warning(f"[Classifier] Retry {attempt}/{max_retries} due to unparseable response: {response_text[:60]}")
            except Exception as e:
                logger.warning(f"[Classifier] Error on attempt {attempt}: {e}")

        # Fallback on parsing failure
        logger.error(f"[Classifier Fallback] Failed to parse classification for: '{text[:50]}...'")
        fallback_res = {
            "intent": Intent.OTHER_GENERAL.value,
            "confidence": 0.10,
            "reasoning": "Unclear inquiry; fallback applied due to malformed model response.",
            "cached": False
        }
        return fallback_res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify customer support tweets using Gemini.")
    parser.add_argument("--text", type=str, default=None, help="Customer tweet text to classify.")
    args = parser.parse_args()

    sample_query = args.text or "@AmazonHelp where is my package? tracker says delivered 2 days ago but nothing arrived at my porch!"

    print("==================================================")
    print("      @AmazonHelp Intent Classification Test      ")
    print("==================================================")
    print(f"Input: \"{sample_query}\"\n")

    classifier = IntentClassifier()
    result = classifier.classify(sample_query)

    print(f"Predicted Intent : {result['intent']}")
    print(f"Confidence       : {result['confidence']:.2f}")
    print(f"Reasoning        : {result['reasoning']}")
    print(f"Cache Hit        : {result.get('cached', False)}")
    print("=" * 50)
