"""
Baseline Systems for @AmazonHelp Evaluation
============================================
Implements two non-LLM baseline systems to establish performance benchmarks:

1. Trivial Baseline:
   - Intent: Majority class ("ORDER_TRACKING_DELAY").
   - Action: Always escalate ("escalate").
   - Reply: One static canned generic brush-off.

2. Simple Rule-Based Baseline:
   - Intent: Keyword & regex matching heuristics.
   - Action: Threshold on urgency/risk keywords.
   - Reply: Static per-intent FAQ template (zero LLM calls).

Both systems output the exact same schema as the main AI pipeline for apples-to-apples comparison.
"""

import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent


# =====================================================================
# 1. Trivial Baseline
# =====================================================================
class TrivialBaseline:
    """Always predicts majority intent, canned reply, and escalates everything."""

    def __init__(self):
        self.name = "Trivial Baseline (Majority / Always-Escalate)"
        self.majority_intent = Intent.ORDER_TRACKING_DELAY.value
        self.canned_reply = (
            "Thank you for reaching out to Amazon Help. A customer service associate "
            "will review your inquiry and follow up shortly."
        )

    def predict(self, text: str, message_id: str = "") -> Dict[str, Any]:
        return {
            "message_id": message_id,
            "predicted_intent": self.majority_intent,
            "predicted_action": "escalate",
            "escalation_reason": "Trivial baseline policy: always route 100% of customer messages to human agents.",
            "drafted_reply": self.canned_reply,
        }


# =====================================================================
# 2. Simple Rule-Based Baseline
# =====================================================================
class SimpleBaseline:
    """Regex/keyword intent matching with per-intent FAQ templates and urgency heuristic."""

    def __init__(self):
        self.name = "Simple Rule-Based Baseline (Regex / Templates)"

        # Keyword mapping to Intent
        self.intent_rules = [
            (
                Intent.ESCALATION_HIGH_RISK.value,
                [r"\blawyer\b", r"\blegal\b", r"\bpolice\b", r"\bsue\b", r"\bproperty\s+damage\b",
                 r"\bvan\b", r"\bstolen\b", r"\$2,?400", r"\bfraud\b", r"\btheft\b", r"\bcourt\b"]
            ),
            (
                Intent.ACCOUNT_LOGIN_SECURITY.value,
                [r"\botp\b", r"\bverification\s*code\b", r"\b2fa\b", r"\btwo-step\b",
                 r"\blocked\s*out\b", r"\bhacked\b", r"\bpassword\b", r"\bunauthorized\b"]
            ),
            (
                Intent.PRODUCT_TECH_SUPPORT.value,
                [r"\becho\b", r"\balexa\b", r"\bblinking\s+yellow\b", r"\bfire\s*stick\b",
                 r"\bremote\b", r"\bfrozen\b", r"\bkindle\b", r"\bdevice\b", r"\bdot\b"]
            ),
            (
                Intent.PRIME_MEMBERSHIP_BILLING.value,
                [r"\bprime\b", r"\bmembership\b", r"\bauto-?renewal\b", r"\bsubscription\b",
                 r"\bcharged\b", r"\bbilling\b", r"\bfee\b", r"\bdouble\s*charged\b"]
            ),
            (
                Intent.DAMAGED_WRONG_ITEM.value,
                [r"\bdamaged\b", r"\bbroken\b", r"\bshattered\b", r"\bdefective\b",
                 r"\bwrong\s+item\b", r"\btorn\b", r"\bcracked\b", r"\bpieces\b", r"\bglass\b"]
            ),
            (
                Intent.REFUND_RETURN_INQUIRY.value,
                [r"\brefund\b", r"\breturn\b", r"\bdropped\s+off\b", r"\bups\b",
                 r"\bkohl'?s\b", r"\breturn\s*window\b", r"\bcredited\b", r"\b30-day\b"]
            ),
            (
                Intent.ORDER_TRACKING_DELAY.value,
                [r"\btracking\b", r"\btrack\b", r"\bdelay\b", r"\blate\b", r"\bcarrier\b",
                 r"\bout\s+for\s+delivery\b", r"\bwhere\s+is\s+my\b", r"\bparcel\b", r"\bpackage\b", r"\bestimated\b"]
            ),
        ]

        # Deterministic per-intent response templates (zero LLM)
        self.templates = {
            Intent.ORDER_TRACKING_DELAY.value: (
                "You can track your package live under 'Your Orders'. Carriers deliver until 9 PM local time. "
                "If it has not arrived by tomorrow, you can request a replacement or refund directly in the app."
            ),
            Intent.REFUND_RETURN_INQUIRY.value: (
                "Eligible items can be returned within 30 days via 'Your Orders' > 'Return or Replace Items'. "
                "Refunds are credited to your original payment method within 3 to 5 business days of carrier drop-off."
            ),
            Intent.DAMAGED_WRONG_ITEM.value: (
                "We sincerely apologize for the damaged or incorrect item. Please visit 'Your Orders' to select the item "
                "and request an immediate replacement or refund with a prepaid return label."
            ),
            Intent.PRIME_MEMBERSHIP_BILLING.value: (
                "To manage your Prime membership, review charges, or cancel auto-renewal, visit Account > Prime Membership. "
                "If unused, eligible subscription fees will be automatically refunded."
            ),
            Intent.ACCOUNT_LOGIN_SECURITY.value: (
                "If you are experiencing OTP or login issues, please check your network connection or select 'Try another way' "
                "to receive your code via email. For security alerts, update your password at amazon.com/security."
            ),
            Intent.PRODUCT_TECH_SUPPORT.value: (
                "Try restarting your device: unplug the power adapter for 40 seconds, then reconnect. For Echo devices "
                "with a blinking yellow light, say 'Alexa, read my notifications'."
            ),
            Intent.ESCALATION_HIGH_RISK.value: (
                "We take safety and critical incident reports very seriously. A senior customer support specialist has been "
                "alerted and will contact you directly to assist."
            ),
            Intent.OTHER_GENERAL.value: (
                "Thank you for contacting Amazon Help. Please visit our Help & Customer Service portal for further assistance."
            ),
        }

        # Keywords that trigger escalation in the simple baseline
        self.escalation_keywords = [
            "lawyer", "legal", "police", "stolen", "property damage", "fraud", "sue",
            "unauthorized", "urgent", "supervisor", "hacked", "court", "attorney"
        ]

    def predict(self, text: str, message_id: str = "") -> Dict[str, Any]:
        lower_text = text.lower()

        # 1. Predict Intent via keyword matching
        predicted_intent = Intent.OTHER_GENERAL.value
        for intent_name, patterns in self.intent_rules:
            for pat in patterns:
                if re.search(pat, lower_text):
                    predicted_intent = intent_name
                    break
            if predicted_intent != Intent.OTHER_GENERAL.value:
                break

        # 2. Predict Action via heuristic triggers
        has_escalation_kw = any(re.search(rf"\b{re.escape(kw)}\b", lower_text) for kw in self.escalation_keywords)
        is_sensitive_intent = predicted_intent in [
            Intent.ESCALATION_HIGH_RISK.value,
            Intent.ACCOUNT_LOGIN_SECURITY.value
        ]

        if has_escalation_kw or is_sensitive_intent:
            action = "escalate"
            reason = f"Triggered by sensitivity rule (intent={predicted_intent} or risk keywords detected)."
        else:
            action = "auto"
            reason = "Standard self-service query eligible for automated resolution."

        reply = self.templates.get(predicted_intent, self.templates[Intent.OTHER_GENERAL.value])

        return {
            "message_id": message_id,
            "predicted_intent": predicted_intent,
            "predicted_action": action,
            "escalation_reason": reason,
            "drafted_reply": reply,
        }


# =====================================================================
# 3. Evaluation Metrics Computation
# =====================================================================
def compute_baseline_metrics(golden_df: pd.DataFrame, preds_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes standardized metrics across intent accuracy and escalation decisions."""
    merged = pd.merge(golden_df, preds_df, on="message_id")
    n = len(merged)

    # 1. Intent Accuracy
    correct_intents = (merged["true_intent"] == merged["predicted_intent"]).sum()
    intent_acc = correct_intents / n if n > 0 else 0.0

    # 2. Escalation Precision, Recall, F1
    # True positives: true='escalate' AND pred='escalate'
    tp = ((merged["true_action"] == "escalate") & (merged["predicted_action"] == "escalate")).sum()
    fp = ((merged["true_action"] == "auto") & (merged["predicted_action"] == "escalate")).sum()
    fn = ((merged["true_action"] == "escalate") & (merged["predicted_action"] == "auto")).sum()
    tn = ((merged["true_action"] == "auto") & (merged["predicted_action"] == "auto")).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 3. Asymmetric Safety Cost:
    # A False Auto-Handle (FN) is severely dangerous (ignoring angry/stolen/legal issue).
    # Cost = (5 * FN) + (1 * FP)
    asymmetric_cost = (5 * fn) + (1 * fp)

    return {
        "n_samples": n,
        "intent_accuracy": intent_acc,
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": f1,
        "false_auto_count": int(fn),
        "false_escalate_count": int(fp),
        "asymmetric_cost": int(asymmetric_cost),
    }


def run_baselines():
    golden_path = PROJECT_ROOT / "eval" / "golden_set.csv"
    if not golden_path.exists():
        print(f"[ERROR] Golden set not found at {golden_path}. Run 'python eval/sample_golden_set.py' first.")
        sys.exit(1)

    golden_df = pd.read_csv(golden_path)
    print("==================================================")
    print("        Evaluating Non-LLM Baseline Systems       ")
    print("==================================================")
    print(f"Golden Set Size: {len(golden_df)} items\n")

    models = [
        ("Trivial Baseline", TrivialBaseline()),
        ("Simple Rule-Based", SimpleBaseline()),
    ]

    results = []
    for model_name, model in models:
        preds = []
        for _, row in golden_df.iterrows():
            pred = model.predict(text=row["text"], message_id=row["message_id"])
            preds.append(pred)

        pred_df = pd.DataFrame(preds)
        metrics = compute_baseline_metrics(golden_df, pred_df)
        metrics["model"] = model_name
        results.append(metrics)

        # Cache baseline predictions for comparison
        out_name = f"baseline_{model_name.lower().replace(' ', '_').replace('-', '_')}_preds.csv"
        pred_df.to_csv(PROJECT_ROOT / "eval" / out_name, index=False)

    # Format Headline Table
    summary_df = pd.DataFrame(results)[[
        "model", "intent_accuracy", "escalation_precision", "escalation_recall", "escalation_f1", "false_auto_count", "asymmetric_cost"
    ]]

    summary_df.columns = [
        "System Baseline", "Intent Acc", "Esc. Precision", "Esc. Recall", "Esc. F1", "False Auto (FN)", "Safety Cost"
    ]

    # Format Headline Table (pure Python, zero extra dependency)
    try:
        print(summary_df.to_markdown(index=False, floatfmt=".3f"))
    except Exception:
        print(summary_df.to_string(index=False))
    print("\n[NOTE] False Auto (FN) is the costliest error: an angry/legal/stolen issue handled incorrectly as self-service.")
    print("Safety Cost Formula: (5 * False_Auto) + (1 * False_Escalate)")
    print("=" * 50)


if __name__ == "__main__":
    run_baselines()
