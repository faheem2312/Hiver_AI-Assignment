"""
Retrieval-Grounded Reply Drafting Engine for @AmazonHelp
=========================================================
Generates customer support replies grounded in historical brand resolutions.

Features:
1. Context-grounded prompting: strictly adopts policies, troubleshooting steps,
   and tone from retrieved historical resolutions.
2. Anti-hallucination guardrails: forbids fabricating financial commitments,
   refund amounts, or delivery guarantees not evidenced in the retrieved context.
3. Twitter-appropriate tone: concise (1-3 sentences), empathetic, and professional.
4. Hash-based disk caching: prevents redundant LLM calls during evaluations.
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DRAFT_CACHE_FILE = PROJECT_ROOT / "data" / "processed" / "draft_cache.json"


class ReplyDrafter:
    """Generates grounded customer support replies using Gemini Flash."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_cache: bool = True
    ):
        self.client = GeminiClient(api_key=api_key, model=model)
        self.use_cache = use_cache
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Loads cached draft replies from disk."""
        if self.use_cache and DRAFT_CACHE_FILE.exists():
            try:
                with open(DRAFT_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load draft cache: {e}")
        return {}

    def _save_cache(self):
        """Persists draft replies cache to disk."""
        if self.use_cache:
            try:
                DRAFT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(DRAFT_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not save draft cache: {e}")

    def _get_cache_key(self, customer_text: str, retrieved_resolutions: List[Dict[str, Any]]) -> str:
        """Generates hash key combining customer inquiry and retrieved context."""
        retrieved_signatures = "-".join([h.get("brand_resolution", "")[:30] for h in retrieved_resolutions])
        combined = f"{customer_text.strip().lower()}--{retrieved_signatures}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()

    def _build_prompt(
        self,
        customer_text: str,
        retrieved_resolutions: List[Dict[str, Any]],
        thread_context: str = "",
        predicted_intent: str = ""
    ) -> str:
        """Builds strict grounding prompt."""
        grounding_snippets = []
        for i, hit in enumerate(retrieved_resolutions, 1):
            grounding_snippets.append(
                f"[Example #{i}]\n"
                f"Past Customer Query: \"{hit.get('customer_text', '')}\"\n"
                f"Past Brand Resolution: \"{hit.get('brand_resolution', '')}\""
            )

        grounding_context = "\n\n".join(grounding_snippets) if grounding_snippets else "No past examples found."

        system_instruction = (
            "You are an AI customer support assistant for @AmazonHelp on Twitter.\n"
            "Your job is to draft a helpful, professional, and concise reply to the customer's tweet.\n\n"
            "### STRICT GROUNDING & SAFETY RULES:\n"
            "1. Ground your advice, troubleshooting steps, and policy explanations EXCLUSIVELY in the historical brand resolutions provided below.\n"
            "2. DO NOT fabricate or promise specific financial adjustments (e.g. never say 'I have refunded $50' or 'here is a gift card').\n"
            "3. DO NOT promise specific delivery dates unless stated in the grounding examples.\n"
            "4. Point customers to self-service paths mentioned in the examples (e.g., 'Your Orders > Return or Replace Items' or checking with neighbors/lockers).\n"
            "5. Keep the reply concise (under 280 characters / 2-3 sentences), empathetic, and Twitter-appropriate.\n"
            "6. Output ONLY the drafted reply text, without quotes, notes, or explanations."
        )

        user_content = f"""### Historical Brand Resolutions for Similar Issues:
{grounding_context}

### Conversation Context:
{thread_context or "New customer inquiry."}

### Customer Tweet to Reply to:
"{customer_text}"

Drafted Reply (concise, grounded, brand-appropriate):"""

        return system_instruction, user_content

    def draft(
        self,
        customer_text: str,
        retrieved_resolutions: List[Dict[str, Any]],
        thread_context: str = "",
        predicted_intent: str = "",
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Drafts a grounded reply for the customer query.
        Returns:
            {"reply": str, "cached": bool}
        """
        cache_key = self._get_cache_key(customer_text, retrieved_resolutions)
        if self.use_cache and cache_key in self.cache:
            return {"reply": self.cache[cache_key], "cached": True}

        system_inst, prompt = self._build_prompt(
            customer_text=customer_text,
            retrieved_resolutions=retrieved_resolutions,
            thread_context=thread_context,
            predicted_intent=predicted_intent
        )

        try:
            drafted_text = self.client.generate(
                prompt=prompt,
                system_instruction=system_inst,
                temperature=temperature,
                max_output_tokens=180
            )

            # Clean output of stray quotes
            clean_reply = drafted_text.strip().strip('"').strip("'")

            if self.use_cache:
                self.cache[cache_key] = clean_reply
                self._save_cache()

            return {"reply": clean_reply, "cached": False}
        except Exception as e:
            logger.error(f"Error generating grounded reply: {e}")
            # Fallback to general brand acknowledgment
            fallback_reply = (
                "We sincerely apologize for the inconvenience. Please check 'Your Orders' for live updates, "
                "or visit amazon.com/help for immediate self-service assistance."
            )
            return {"reply": fallback_reply, "cached": False}


if __name__ == "__main__":
    import argparse
    from src.retrieve import ResolutionRetriever

    parser = argparse.ArgumentParser(description="Draft grounded Amazon support reply.")
    parser.add_argument("--text", type=str, default=None, help="Customer message.")
    args = parser.parse_args()

    sample_query = args.text or "@AmazonHelp I dropped off my return package at UPS 4 days ago. When do I get my money back?"

    print("==================================================")
    print("      @AmazonHelp Grounded Reply Drafting Test    ")
    print("==================================================")
    print(f"Customer Tweet: \"{sample_query}\"\n")

    # Retrieve grounding context
    retriever = ResolutionRetriever()
    hits = retriever.retrieve(sample_query, top_k=2)

    print("--- Retrieved Grounding Context ---")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['brand_resolution']}")

    # Draft reply
    drafter = ReplyDrafter()
    res = drafter.draft(sample_query, retrieved_resolutions=hits)

    print("\n--- Drafted Reply (Grounded) ---")
    print(f"\"{res['reply']}\"")
    print(f"Cache Hit: {res['cached']}")
    print("==================================================")
