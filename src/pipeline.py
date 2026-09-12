"""
End-to-End Customer Support Pipeline for @AmazonHelp
====================================================
Orchestrates:
1. Intent Classification (Gemini 3.1 Flash Lite + Few-Shot)
2. Historical Knowledge Retrieval (FAISS / Vectorized NumPy Cosine Search)
3. Grounded Reply Drafting (Retrieval-Augmented Gemini Prompting)
4. Safety & Policy Escalation Decision (Rule-Based Gates + LLM Fallback)

Produces a standardized, auditable response schema.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classify import IntentClassifier
from src.retrieve import ResolutionRetriever
from src.draft_reply import ReplyDrafter
from src.escalate import EscalationPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AmazonSupportPipeline:
    """Master agent coordinating classification, retrieval, drafting, and escalation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        use_cache: bool = True
    ):
        logger.info("Initializing @AmazonHelp AI Support Pipeline...")
        self.classifier = IntentClassifier(api_key=api_key, model=model, use_cache=use_cache)
        self.retriever = ResolutionRetriever()
        self.drafter = ReplyDrafter(api_key=api_key, model=model, use_cache=use_cache)
        self.escalator = EscalationPolicy()
        logger.info("Pipeline initialized successfully and ready for inference.")

    def run(
        self,
        customer_text: str,
        message_id: str = "",
        thread_context: str = "",
        top_k_retrieve: int = 3
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline for an incoming customer tweet.

        Returns:
            Structured dictionary with classification, retrieval, drafted reply,
            escalation decision, and execution metrics.
        """
        start_time = time.perf_counter()

        # Step 1: Intent Classification
        clf_result = self.classifier.classify(customer_text)
        intent = clf_result["intent"]
        confidence = clf_result["confidence"]
        clf_reasoning = clf_result["reasoning"]

        # Step 2: Historical Knowledge Retrieval (with intent boosting)
        retrieved_hits = self.retriever.retrieve(
            query=customer_text,
            top_k=top_k_retrieve,
            predicted_intent=intent
        )
        top_similarity = retrieved_hits[0]["similarity"] if retrieved_hits else 0.0

        # Step 3: Retrieval-Grounded Reply Drafting
        draft_result = self.drafter.draft(
            customer_text=customer_text,
            retrieved_resolutions=retrieved_hits,
            thread_context=thread_context,
            predicted_intent=intent
        )
        drafted_reply = draft_result["reply"]

        # Step 4: Escalation Policy Decision
        action, escalation_reason = self.escalator.decide(
            customer_text=customer_text,
            predicted_intent=intent,
            intent_confidence=confidence,
            retrieval_similarity=top_similarity,
            thread_context=thread_context
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "message_id": message_id,
            "customer_text": customer_text,
            "predicted_intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": clf_reasoning,
            "top_retrieval_similarity": top_similarity,
            "retrieved_resolutions": retrieved_hits,
            "drafted_reply": drafted_reply,
            "predicted_action": action,
            "escalation_reason": escalation_reason,
            "latency_ms": round(latency_ms, 1),
            "cached_classification": clf_result.get("cached", False),
            "cached_draft": draft_result.get("cached", False)
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run end-to-end @AmazonHelp support pipeline.")
    parser.add_argument("--text", type=str, default=None, help="Incoming customer tweet.")
    args = parser.parse_args()

    test_tweet = args.text or "@AmazonHelp I dropped off my return at UPS 5 days ago. When do I get my money back?"

    print("==================================================")
    print("      @AmazonHelp End-to-End Pipeline Test        ")
    print("==================================================")
    print(f"Customer Tweet : \"{test_tweet}\"\n")

    pipeline = AmazonSupportPipeline()
    result = pipeline.run(test_tweet)

    print("\n--- Pipeline Execution Output ---")
    print(f"1. Predicted Intent      : {result['predicted_intent']} (Conf: {result['intent_confidence']:.2f})")
    print(f"   Classification Reason : {result['intent_reasoning']}")
    print(f"2. Top Retrieval Sim     : {result['top_retrieval_similarity']:.3f}")
    print(f"3. Escalation Decision   : [{result['predicted_action'].upper()}]")
    print(f"   Escalation Reason     : {result['escalation_reason']}")
    print(f"4. Grounded Draft Reply  :\n   \"{result['drafted_reply']}\"")
    print(f"5. Total Latency         : {result['latency_ms']} ms")
    print("==================================================")
