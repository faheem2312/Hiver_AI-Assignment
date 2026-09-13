"""
Comprehensive Metrics Suite for Customer Support AI Evaluation
===============================================================
Computes:
1. Intent Classification: Accuracy, Macro F1, Confusion Matrix, Per-class metrics.
2. Escalation Policy: Precision, Recall, F1, and Asymmetric Safety Loss (weighting False Auto 5x).
3. Retrieval Quality: Hit@1, Hit@k, Mean Reciprocal Rank (MRR), and average similarity.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def compute_intent_metrics(
    true_intents: List[str],
    pred_intents: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Computes intent accuracy, macro/weighted F1, and confusion matrix."""
    if not true_intents or not pred_intents or len(true_intents) != len(pred_intents):
        return {"accuracy": 0.0, "macro_f1": 0.0}

    acc = accuracy_score(true_intents, pred_intents)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        true_intents, pred_intents, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_intents, pred_intents, average="weighted", zero_division=0
    )

    all_labels = sorted(list(set(true_intents + pred_intents))) if labels is None else labels
    cm = confusion_matrix(true_intents, pred_intents, labels=all_labels)

    # Format Confusion Matrix as readable DataFrame
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in all_labels], columns=[f"Pred: {l}" for l in all_labels])

    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm_df,
        "labels": all_labels
    }


def compute_escalation_metrics(
    true_actions: List[str],
    pred_actions: List[str],
    false_auto_cost: float = 5.0,
    false_escalate_cost: float = 1.0
) -> Dict[str, Any]:
    """
    Computes precision, recall, F1, and asymmetric safety loss for escalation.
    Treats 'escalate' as positive class.
    False Auto-Handle (FN) is penalized heavily by `false_auto_cost` (default 5x).
    """
    n = len(true_actions)
    if n == 0 or len(pred_actions) != n:
        return {}

    tp = sum(1 for t, p in zip(true_actions, pred_actions) if t == "escalate" and p == "escalate")
    fp = sum(1 for t, p in zip(true_actions, pred_actions) if t == "auto" and p == "escalate")
    fn = sum(1 for t, p in zip(true_actions, pred_actions) if t == "escalate" and p == "auto")
    tn = sum(1 for t, p in zip(true_actions, pred_actions) if t == "auto" and p == "auto")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Safety Loss Formula: (5 * False_Auto) + (1 * False_Escalate)
    asymmetric_loss = (fn * false_auto_cost) + (fp * false_escalate_cost)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "false_escalations": fp,
        "false_auto_handles": fn,
        "true_auto_handles": tn,
        "asymmetric_safety_loss": float(asymmetric_loss),
        "auto_handle_rate": float((tn + fn) / n) if n > 0 else 0.0
    }


def compute_retrieval_metrics(
    retrieved_hits_list: List[List[Dict[str, Any]]],
    true_intents: List[str],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Evaluates historical retrieval quality against true customer intent.
    Computes Hit@1, Hit@k, and Mean Reciprocal Rank (MRR).
    """
    if not retrieved_hits_list or len(retrieved_hits_list) != len(true_intents):
        return {"hit_at_1": 0.0, "hit_at_k": 0.0, "mrr": 0.0}

    hit_1 = 0
    hit_k = 0
    reciprocal_ranks = []
    avg_similarities = []

    for hits, true_cat in zip(retrieved_hits_list, true_intents):
        if not hits:
            reciprocal_ranks.append(0.0)
            continue

        avg_similarities.append(hits[0].get("similarity", 0.0))

        # Check Hit@1
        if hits[0].get("category") == true_cat:
            hit_1 += 1

        # Check Hit@k & find rank of first matching resolution
        found_rank = 0
        for rank, h in enumerate(hits[:top_k], 1):
            if h.get("category") == true_cat:
                if found_rank == 0:
                    found_rank = rank
                hit_k += 1
                break

        reciprocal_ranks.append(1.0 / found_rank if found_rank > 0 else 0.0)

    n = len(true_intents)
    return {
        "hit_at_1": float(hit_1 / n) if n > 0 else 0.0,
        f"hit_at_{top_k}": float(hit_k / n) if n > 0 else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "avg_top1_similarity": float(np.mean(avg_similarities)) if avg_similarities else 0.0
    }


if __name__ == "__main__":
    print("Testing Metrics Suite...")
    y_true = ["auto", "escalate", "auto", "escalate"]
    y_pred = ["auto", "auto", "auto", "escalate"]
    esc_m = compute_escalation_metrics(y_true, y_pred)
    print(f"Escalation Precision : {esc_m['precision']:.2f}")
    print(f"Escalation Recall    : {esc_m['recall']:.2f}")
    print(f"False Auto Handles   : {esc_m['false_auto_handles']} (Costlier error)")
    print(f"Safety Loss          : {esc_m['asymmetric_safety_loss']}")
