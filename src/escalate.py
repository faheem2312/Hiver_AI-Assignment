"""
Escalation Policy Engine for @AmazonHelp AI Support Agent
=========================================================
Determines whether an incoming customer inquiry can be safely auto-handled
or must be escalated to a human support specialist, with human-readable rationale.

Design Principles:
1. Two-Tier Architecture: Deterministic rule-based checks first (safety, legal,
   theft, low confidence, missing grounding), with LLM fallback only for
   genuinely ambiguous border cases.
2. Safety-First Asymmetric Risk: A False Auto-Handle (mishandling stolen property,
   legal threats, or angry disputes) is far costlier than a False Escalation.
3. Transparent Rationale: Every decision produces a clear, human-readable reason
   explaining the trigger.
"""

import re
import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent
from src.gemini_client import GeminiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EscalationPolicy:
    """Evaluates support inquiries for human escalation triggers."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        similarity_threshold: float = 0.40,
        high_value_threshold: float = 100.0,
        enable_llm_fallback: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.high_value_threshold = high_value_threshold
        self.enable_llm_fallback = enable_llm_fallback
        self.gemini_client: Optional[GeminiClient] = None

        # Risk regex patterns with explicit categories
        self.risk_patterns = [
            ("LEGAL_THREAT", re.compile(r"\b(lawyer|attorney|legal action|court|sue|suing|lawsuit)\b", re.I)),
            ("CRIME_THEFT", re.compile(r"\b(police|stolen|theft|burglar|package thief|porch pirate)\b", re.I)),
            ("PROPERTY_DAMAGE", re.compile(r"\b(property damage|backed into|drove into|smashed my|damaged my fence|hit my car)\b", re.I)),
            ("EXTREME_FRUSTRATION", re.compile(r"\b(human supervisor now|manager immediately|unacceptable service|furious|livid)\b", re.I)),
        ]

    def _extract_monetary_amount(self, text: str) -> Optional[float]:
        """Extracts claimed dollar values to identify high-stake financial requests."""
        # Matches $100, $2,400, $14.99, etc.
        matches = re.findall(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text)
        amounts = []
        for m in matches:
            try:
                amt = float(m.replace(",", ""))
                amounts.append(amt)
            except ValueError:
                continue
        return max(amounts) if amounts else None

    def _llm_fallback_decision(self, text: str, intent: str, context: str) -> Tuple[str, str]:
        """Invokes Gemini fallback only for borderline ambiguous inquiries."""
        if self.gemini_client is None:
            self.gemini_client = GeminiClient()

        prompt = f"""You are a risk supervisor for @AmazonHelp customer support.
Determine whether this customer inquiry should be 'auto' (safe for automated self-service guidance)
or 'escalate' (requires a human specialist due to complexity, anger, or dispute).

Customer Tweet: "{text}"
Predicted Category: {intent}
Thread Context: {context or "None"}

Policy:
- 'auto': Standard tracking inquiries, return window questions, device troubleshooting, routine FAQ.
- 'escalate': Unresolved multi-turn disputes, missing high-value orders, customer anger, edge-case policy ambiguity.

Return ONLY a valid JSON object with keys 'action' ('auto' or 'escalate') and 'reason' (one clear sentence).
"""
        try:
            res_text = self.gemini_client.generate(
                prompt=prompt,
                temperature=0.0,
                max_output_tokens=150,
                response_mime_type="application/json"
            )
            data = json.loads(res_text.strip().strip("`").replace("json\n", ""))
            action = data.get("action", "escalate").lower()
            if action not in ["auto", "escalate"]:
                action = "escalate"
            reason = data.get("reason", "LLM fallback determined human review required.")
            return action, f"LLM boundary judgment: {reason}"
        except Exception as e:
            logger.warning(f"Escalation LLM fallback failed: {e}. Defaulting conservatively to escalate.")
            return "escalate", "Conservative fallback: unable to confirm automated resolution safety."

    def decide(
        self,
        customer_text: str,
        predicted_intent: str,
        intent_confidence: float,
        retrieval_similarity: float,
        thread_context: str = ""
    ) -> Tuple[str, str]:
        """
        Executes the hierarchical escalation policy.

        Returns:
            (action: 'auto' | 'escalate', reason: human_readable_string)
        """
        # 1. Mandatory Sensitive Intent Check
        if predicted_intent == Intent.ESCALATION_HIGH_RISK.value:
            return "escalate", "Mandatory policy: High-risk incident requiring senior human specialist."

        if predicted_intent == Intent.ACCOUNT_LOGIN_SECURITY.value:
            return "escalate", "Mandatory policy: Account credentials & Two-Factor OTP require verified human channel."

        # 2. Safety, Legal, Crime & Damage Risk Checks
        for risk_type, pattern in self.risk_patterns:
            match = pattern.search(customer_text)
            if match:
                term = match.group(0)
                reasons = {
                    "LEGAL_THREAT": f"Legal threat detected ('{term}'); escalated to executive relations.",
                    "CRIME_THEFT": f"Crime or theft reported ('{term}'); requires carrier depot investigation.",
                    "PROPERTY_DAMAGE": f"Property damage signal detected ('{term}'); routed to claims team.",
                    "EXTREME_FRUSTRATION": f"Acute customer escalation demand ('{term}'); routed to human supervisor.",
                }
                return "escalate", reasons.get(risk_type, f"Critical risk trigger detected: '{term}'.")

        # 3. Financial Exposure Gate (High-Value Items > $100)
        max_amount = self._extract_monetary_amount(customer_text)
        if max_amount is not None and max_amount >= self.high_value_threshold:
            return "escalate", (
                f"High-value claim detected (${max_amount:,.2f}); exceeds "
                f"automated self-service threshold (${self.high_value_threshold:,.0f})."
            )

        # 4. Model & Grounding Confidence Gates
        if intent_confidence < self.confidence_threshold:
            return "escalate", (
                f"Low classification confidence ({intent_confidence:.2f} < {self.confidence_threshold}); "
                "intent is ambiguous and requires human clarification."
            )

        if retrieval_similarity < self.similarity_threshold:
            return "escalate", (
                f"Low knowledge retrieval similarity ({retrieval_similarity:.2f} < {self.similarity_threshold}); "
                "no verified historical brand resolution found in knowledge base."
            )

        # 5. Borderline Ambiguity Check (Trigger LLM Fallback if enabled)
        is_borderline_confidence = (self.confidence_threshold <= intent_confidence < 0.72)
        is_borderline_similarity = (self.similarity_threshold <= retrieval_similarity < 0.48)
        if self.enable_llm_fallback and (is_borderline_confidence or is_borderline_similarity):
            return self._llm_fallback_decision(customer_text, predicted_intent, thread_context)

        # 6. Standard Safe Automated Resolution
        return "auto", "Routine self-service inquiry with high confidence and verified grounding."


if __name__ == "__main__":
    policy = EscalationPolicy()

    test_cases = [
        ("@AmazonHelp where is my order? tracker says out for delivery", "ORDER_TRACKING_DELAY", 0.95, 0.78),
        ("@AmazonHelp I want a refund for my stolen laptop worth $2,400", "ORDER_TRACKING_DELAY", 0.92, 0.72),
        ("@AmazonHelp delivery driver smashed into my gate! calling my lawyer!", "ESCALATION_HIGH_RISK", 0.99, 0.81),
        ("@AmazonHelp locked out of my account cannot get OTP code", "ACCOUNT_LOGIN_SECURITY", 0.94, 0.70),
        ("@AmazonHelp is the sky blue today?", "OTHER_GENERAL", 0.35, 0.22),
    ]

    print("==================================================")
    print("        @AmazonHelp Escalation Policy Test        ")
    print("==================================================")
    for text, intent, conf, sim in test_cases:
        action, reason = policy.decide(text, intent, conf, sim)
        print(f"Tweet  : \"{text}\"")
        print(f"Action : [{action.upper()}]")
        print(f"Reason : {reason}\n")
    print("==================================================")
