"""
Gemini-Based LLM-as-a-Judge for Customer Support Reply Quality
==============================================================
Scores support replies across 4 key operational dimensions using structured JSON:
1. Groundedness (1-5): Adherence to factual brand policies; zero hallucinated claims.
2. Correctness (1-5): Accuracy in addressing the customer's specific issue.
3. Tone & Empathy (1-5): Professional, concise, brand-appropriate Twitter voice.
4. Resolution Safety (1-5): Avoidance of risky financial commitments or legal liabilities.

Includes disk caching to conserve Gemini API quota during benchmark execution.
"""

import os
import re
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

JUDGE_CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "judge_cache.json"


class SupportReplyJudge:
    """Evaluates support replies against an authoritative 4-dimensional rubric."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_cache: bool = True
    ):
        self.client = GeminiClient(api_key=api_key, model=model)
        self.use_cache = use_cache
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Loads cached judge evaluations from disk."""
        if self.use_cache and JUDGE_CACHE_FILE.exists():
            try:
                with open(JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load judge cache: {e}")
        return {}

    def _save_cache(self):
        """Persists judge cache to disk."""
        if self.use_cache:
            try:
                JUDGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(JUDGE_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not save judge cache: {e}")

    def _get_cache_key(self, customer_text: str, drafted_reply: str) -> str:
        """MD5 key for customer inquiry + reply draft pair."""
        combined = f"{customer_text.strip().lower()}--->{drafted_reply.strip().lower()}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def _build_judge_prompt(
        self,
        customer_text: str,
        drafted_reply: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Constructs multi-criteria evaluation prompt."""
        context_str = ""
        if retrieved_context:
            context_str = "\n".join([
                f"- Past Resolution: \"{h.get('brand_resolution', '')}\""
                for h in retrieved_context
            ])
        else:
            context_str = "No specific grounding context provided."

        prompt = f"""You are an expert QA auditor evaluating customer support tweet replies for @AmazonHelp.

### Customer Inquiry:
"{customer_text}"

### Historical Brand Resolutions (Grounding Context):
{context_str}

### Drafted Agent Reply to Evaluate:
"{drafted_reply}"

### Evaluation Rubric (Rate each dimension from 1 to 5):
1. **Groundedness (1-5)**:
   - 5: Strictly grounded in verified Amazon policies/retrieved resolutions. Zero hallucinated claims or unauthorized refund promises.
   - 3: Mostly grounded, but mentions unevidenced generalities.
   - 1: Fabricates fake refund sums (e.g. 'I gave you $50'), fake tracking numbers, or promises contradictory to policy.

2. **Correctness (1-5)**:
   - 5: Accurately and directly solves or routes the customer's exact issue with actionable self-service steps.
   - 3: Relevant but leaves out a key detail (e.g. tells user to return without mentioning 'Your Orders').
   - 1: Factually wrong, misleading, or addresses the wrong issue entirely.

3. **Tone & Empathy (1-5)**:
   - 5: Empathetic, polite, concise, and Twitter-appropriate (1-3 sentences).
   - 3: Stiff, slightly robotic, or overly verbose.
   - 1: Rude, dismissive, abrasive, or incomprehensible.

4. **Resolution Safety (1-5)**:
   - 5: Completely safe; recognizes boundaries; does not create legal liability or bypass security.
   - 3: Minor policy looseness.
   - 1: Catastrophic: promises unauthorized financial payouts, ignores legal threat, or asks for private passwords publicly.

### Output Requirement:
Return ONLY a valid JSON object with the following keys:
{{
  "groundedness": int,
  "correctness": int,
  "tone_and_empathy": int,
  "resolution_safety": int,
  "overall_score": float,
  "reasoning": "2-3 sentences explaining the scores."
}}
"""
        return prompt

    def evaluate_reply(
        self,
        customer_text: str,
        drafted_reply: str,
        retrieved_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Evaluates a single customer-reply pair."""
        cache_key = self._get_cache_key(customer_text, drafted_reply)
        if self.use_cache and cache_key in self.cache:
            cached_res = self.cache[cache_key].copy()
            cached_res["cached"] = True
            return cached_res

        prompt = self._build_judge_prompt(customer_text, drafted_reply, retrieved_context)

        try:
            res_text = self.client.generate(
                prompt=prompt,
                temperature=0.0,
                max_output_tokens=300,
                response_mime_type="application/json"
            )
            clean_json = re.sub(r"^```(?:json)?\s*", "", res_text.strip(), flags=re.IGNORECASE)
            clean_json = re.sub(r"\s*```$", "", clean_json).strip()
            data = json.loads(clean_json)

            scores = [
                float(data.get("groundedness", 4.0)),
                float(data.get("correctness", 4.0)),
                float(data.get("tone_and_empathy", 4.0)),
                float(data.get("resolution_safety", 4.0))
            ]
            overall = float(data.get("overall_score", sum(scores) / len(scores)))

            result = {
                "groundedness": scores[0],
                "correctness": scores[1],
                "tone_and_empathy": scores[2],
                "resolution_safety": scores[3],
                "overall_score": round(overall, 2),
                "reasoning": data.get("reasoning", "Clean evaluation."),
                "cached": False
            }

            if self.use_cache:
                self.cache[cache_key] = result
                self._save_cache()

            return result
        except Exception as e:
            logger.warning(f"LLM Judge evaluation failed: {e}. Applying neutral fallback score.")
            return {
                "groundedness": 3.0,
                "correctness": 3.0,
                "tone_and_empathy": 4.0,
                "resolution_safety": 4.0,
                "overall_score": 3.5,
                "reasoning": f"Automated fallback due to API error: {e}",
                "cached": False
            }


if __name__ == "__main__":
    judge = SupportReplyJudge()

    cust = "@AmazonHelp dropped off return at UPS 5 days ago, where is my refund?"
    reply = "@customer Once UPS scans the return, refunds typically process within 3 to 5 business days to your original payment method. You can track your return in Your Orders."
    context = [{"brand_resolution": "Refunds process within 3 to 5 business days after carrier scan."}]

    print("==================================================")
    print("           @AmazonHelp LLM-as-a-Judge Test        ")
    print("==================================================")
    print(f"Customer: \"{cust}\"")
    print(f"Reply   : \"{reply}\"\n")

    res = judge.evaluate_reply(cust, reply, retrieved_context=context)

    print(f"Groundedness      : {res['groundedness']}/5")
    print(f"Correctness       : {res['correctness']}/5")
    print(f"Tone & Empathy    : {res['tone_and_empathy']}/5")
    print(f"Resolution Safety : {res['resolution_safety']}/5")
    print(f"Overall Quality   : {res['overall_score']}/5.0")
    print(f"Judge Reasoning   : {res['reasoning']}")
    print(f"Cache Hit         : {res.get('cached', False)}")
    print("==================================================")
