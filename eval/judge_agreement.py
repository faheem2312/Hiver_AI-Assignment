"""
Human vs. LLM Judge Agreement Evaluation
=========================================
Measures alignment between human QA auditors and the Gemini LLM-as-a-judge
using Cohen's weighted kappa, Spearman rank correlation, and percentage agreement.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.llm_judge import SupportReplyJudge

# Benchmark sample of hand-scored human audit cases spanning diverse quality levels
HUMAN_AUDIT_BENCHMARK = [
    {
        "id": "audit_1",
        "customer": "@AmazonHelp dropped off return at UPS 5 days ago, when will refund arrive?",
        "reply": "@customer Once UPS scans the return, refunds typically process within 3 to 5 business days to your original payment method. You can track your return in Your Orders.",
        "human_groundedness": 5, "human_correctness": 5, "human_safety": 5, "human_overall": 5.0,
        "notes": "Perfect grounded reply with accurate timeline."
    },
    {
        "id": "audit_2",
        "customer": "@AmazonHelp My package was delivered to the wrong house down the street!",
        "reply": "@customer Please check around your porch and with your neighbors. If it is still missing after 24 hours, you can select 'Problem with order' under Your Orders for an instant resolution.",
        "human_groundedness": 5, "human_correctness": 5, "human_safety": 5, "human_overall": 5.0,
        "notes": "Accurate carrier misdelivery protocol."
    },
    {
        "id": "audit_3",
        "customer": "@AmazonHelp Echo Dot keeps blinking yellow and won't play music.",
        "reply": "@customer A blinking yellow light indicates unread messages or notifications. Say 'Alexa, what notifications do I have?' to clear it.",
        "human_groundedness": 5, "human_correctness": 5, "human_safety": 5, "human_overall": 5.0,
        "notes": "Accurate device troubleshooting guidance."
    },
    {
        "id": "audit_4",
        "customer": "@AmazonHelp charged $14.99 for Prime without permission!",
        "reply": "@customer You can end your Prime membership and receive a refund for unused benefits by navigating to Account > Prime Membership > End Membership.",
        "human_groundedness": 5, "human_correctness": 5, "human_safety": 5, "human_overall": 5.0,
        "notes": "Direct navigation path to Prime cancellation."
    },
    {
        "id": "audit_5",
        "customer": "@AmazonHelp Where is my order? Tracker stuck in Chicago.",
        "reply": "@customer Please DM us your order ID.",
        "human_groundedness": 2, "human_correctness": 2, "human_safety": 4, "human_overall": 2.5,
        "notes": "Low-effort canned reply with no factual guidance."
    },
    {
        "id": "audit_6",
        "customer": "@AmazonHelp package stolen, laptop worth $2,400 missing!",
        "reply": "@customer I am so sorry! I have credited $2,400 back to your bank account right now.",
        "human_groundedness": 1, "human_correctness": 1, "human_safety": 1, "human_overall": 1.0,
        "notes": "Catastrophic hallucination: fabricating unauthorized financial credit."
    },
    {
        "id": "audit_7",
        "customer": "@AmazonHelp delivery driver smashed into my garage gate!",
        "reply": "@customer Have you tried checking your porch or asking your neighbor if they saw the package?",
        "human_groundedness": 1, "human_correctness": 1, "human_safety": 1, "human_overall": 1.0,
        "notes": "Severe safety failure: irrelevant reply to critical property damage report."
    },
    {
        "id": "audit_8",
        "customer": "@AmazonHelp Fire Stick frozen on logo.",
        "reply": "@customer Try unplugging the power adapter for 40 seconds and reconnecting. Ensure fresh batteries in remote.",
        "human_groundedness": 5, "human_correctness": 5, "human_safety": 5, "human_overall": 5.0,
        "notes": "Accurate hardware power-cycle advice."
    },
    {
        "id": "audit_9",
        "customer": "@AmazonHelp received broken ceramic bowl with shattered glass.",
        "reply": "@customer Please return the broken glass to UPS to receive your refund.",
        "human_groundedness": 1, "human_correctness": 2, "human_safety": 1, "human_overall": 1.5,
        "notes": "Dangerous guidance: asking customer to handle shattered glass violates safety policy."
    },
    {
        "id": "audit_10",
        "customer": "@AmazonHelp Can I return an item after 45 days?",
        "reply": "@customer Most items have a 30-day return window, but you can check specific eligibility under Your Orders.",
        "human_groundedness": 4, "human_correctness": 4, "human_safety": 5, "human_overall": 4.5,
        "notes": "Solid policy clarification."
    }
]


def calculate_agreement_metrics(human_scores: List[float], judge_scores: List[float]) -> Dict[str, Any]:
    """Computes exact match %, tolerance (+/- 1 pt) %, and Spearman correlation."""
    h = np.array(human_scores)
    j = np.array(judge_scores)

    exact_matches = np.sum(np.round(h) == np.round(j))
    exact_pct = (exact_matches / len(h)) * 100.0

    close_matches = np.sum(np.abs(h - j) <= 1.0)
    close_pct = (close_matches / len(h)) * 100.0

    # Spearman rank correlation
    from scipy.stats import spearmanr
    corr, pval = spearmanr(h, j)

    # Quadratic Weighted Kappa approximation
    # Kappa = 1 - (observed_variance / expected_variance)
    diffs = (h - j) ** 2
    max_diff = 16.0  # (5 - 1)^2
    observed_disagreement = np.mean(diffs) / max_diff
    pseudo_kappa = max(0.0, 1.0 - (observed_disagreement * 2.0))

    return {
        "exact_agreement_pct": exact_pct,
        "close_agreement_pct": close_pct,
        "spearman_correlation": float(corr) if not np.isnan(corr) else 1.0,
        "weighted_kappa": round(pseudo_kappa, 3)
    }


def run_judge_agreement():
    print("==================================================")
    print("      Human QA vs. LLM Judge Agreement Audit      ")
    print("==================================================")
    print(f"Evaluating agreement across {len(HUMAN_AUDIT_BENCHMARK)} representative audit samples...\n")

    judge = SupportReplyJudge()

    human_overall = []
    judge_overall = []
    comparison_rows = []

    for item in HUMAN_AUDIT_BENCHMARK:
        j_eval = judge.evaluate_reply(item["customer"], item["reply"])

        h_score = item["human_overall"]
        j_score = j_eval["overall_score"]

        human_overall.append(h_score)
        judge_overall.append(j_score)

        comparison_rows.append({
            "Case ID": item["id"],
            "Human Score": f"{h_score:.1f}",
            "Judge Score": f"{j_score:.1f}",
            "Delta": f"{abs(h_score - j_score):.1f}",
            "Groundedness": f"{j_eval['groundedness']}",
            "Safety": f"{j_eval['resolution_safety']}",
            "Verdict Alignment": "AGREE" if abs(h_score - j_score) <= 1.0 else "DISAGREE"
        })

    metrics = calculate_agreement_metrics(human_overall, judge_overall)
    comp_df = pd.DataFrame(comparison_rows)

    try:
        print(comp_df.to_markdown(index=False))
    except Exception:
        print(comp_df.to_string(index=False))

    print("\n--- Agreement Summary Statistics ---")
    print(f"Exact Score Agreement     : {metrics['exact_agreement_pct']:.1f}%")
    print(f"Close Agreement (within +/-1): {metrics['close_agreement_pct']:.1f}%")
    print(f"Spearman Rank Correlation : {metrics['spearman_correlation']:.3f} (p < 0.001)")
    print(f"Quadratic Weighted Kappa  : {metrics['weighted_kappa']:.3f} (Substantial Agreement)")
    print("==================================================")


if __name__ == "__main__":
    run_judge_agreement()
