"""
Stratified Golden Evaluation Set Sampler
========================================
Extracts a clean, stratified 150-200 example evaluation benchmark from
`data/processed/amazon_threads_subsample.csv`.

Crucial Data Hygiene:
- The sampled golden examples are strictly HELD OUT and saved to `eval/golden_set.csv`.
- The remaining 4,800+ examples are saved to `data/processed/amazon_grounding_corpus.csv`.
  This guarantees zero data leakage into the FAISS retrieval index.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intents import Intent


def sample_golden_set(
    total_samples: int = 200,
    random_seed: int = 42,
    auto_populate_reference: bool = True
):
    data_path = PROJECT_ROOT / "data" / "processed" / "amazon_threads_subsample.csv"
    if not data_path.exists():
        print(f"[ERROR] Processed data not found at {data_path}. Run 'python src/data_prep.py' first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print("==================================================")
    print("       Sampling Stratified Golden Evaluation Set  ")
    print("==================================================")
    print(f"Total available threads: {len(df):,}")

    # Identify category/stratification column
    strat_col = "category_tag" if "category_tag" in df.columns else "filter_reason"

    # Stratified sampling across categories
    unique_cats = df[strat_col].unique()
    per_cat_quota = max(1, total_samples // len(unique_cats))

    sampled_dfs = []
    for cat in unique_cats:
        cat_df = df[df[strat_col] == cat]
        n_sample = min(len(cat_df), per_cat_quota)
        sampled_dfs.append(cat_df.sample(n=n_sample, random_state=random_seed))

    golden_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    if len(golden_df) > total_samples:
        golden_df = golden_df.iloc[:total_samples]

    # Clean split: Remaining examples form the grounding corpus (ZERO overlap)
    held_out_ids = set(golden_df["thread_id"])
    grounding_df = df[~df["thread_id"].isin(held_out_ids)].reset_index(drop=True)

    grounding_corpus_path = PROJECT_ROOT / "data" / "processed" / "amazon_grounding_corpus.csv"
    grounding_df.to_csv(grounding_corpus_path, index=False)
    print(f"[ISOLATION] Saved {len(grounding_df):,} held-out threads to grounding corpus: {grounding_corpus_path.name}")

    # Build the Golden Set Schema
    golden_rows = []
    for idx, row in golden_df.iterrows():
        msg_id = row["customer_tweet_id"]
        text = row["customer_text"]
        thread_context = f"Brand: {row['brand_text']}" if pd.notna(row.get("brand_text")) else ""
        category = row.get("category_tag", "UNKNOWN")

        # Ground-truth defaults for reference baseline execution
        if category == "ORDER_TRACKING_DELAY":
            t_intent = Intent.ORDER_TRACKING_DELAY.value
            t_action = "auto"
            has_grounding = "Y"
        elif category == "REFUND_RETURN_INQUIRY":
            t_intent = Intent.REFUND_RETURN_INQUIRY.value
            t_action = "auto"
            has_grounding = "Y"
        elif category == "DAMAGED_WRONG_ITEM":
            t_intent = Intent.DAMAGED_WRONG_ITEM.value
            t_action = "auto"
            has_grounding = "Y"
        elif category == "PRIME_MEMBERSHIP_BILLING":
            t_intent = Intent.PRIME_MEMBERSHIP_BILLING.value
            t_action = "auto"
            has_grounding = "Y"
        elif category == "ACCOUNT_LOGIN_SECURITY":
            t_intent = Intent.ACCOUNT_LOGIN_SECURITY.value
            t_action = "escalate"
            has_grounding = "Y"
        elif category == "PRODUCT_TECH_SUPPORT":
            t_intent = Intent.PRODUCT_TECH_SUPPORT.value
            t_action = "auto"
            has_grounding = "Y"
        elif category == "ESCALATION_HIGH_RISK":
            t_intent = Intent.ESCALATION_HIGH_RISK.value
            t_action = "escalate"
            has_grounding = "N"
        else:
            t_intent = Intent.OTHER_GENERAL.value
            t_action = "escalate"
            has_grounding = "N"

        golden_rows.append({
            "message_id": msg_id,
            "text": text,
            "thread_context": thread_context,
            "true_intent": t_intent if auto_populate_reference else "",
            "has_good_grounding_example": has_grounding if auto_populate_reference else "",
            "true_action": t_action if auto_populate_reference else "",
            "notes": f"Category: {category}" if auto_populate_reference else ""
        })

    out_df = pd.DataFrame(golden_rows)
    eval_dir = PROJECT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    golden_set_path = eval_dir / "golden_set.csv"
    out_df.to_csv(golden_set_path, index=False)
    print(f"[OK] Successfully sampled {len(out_df)} golden evaluation items.")
    print(f"Saved Golden Set to: {golden_set_path}")

    # Print Class Balance
    if auto_populate_reference:
        print("\nGolden Set Intent Distribution:")
        print(out_df["true_intent"].value_counts().to_string())
        print("\nGolden Set Action Distribution:")
        print(out_df["true_action"].value_counts().to_string())
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stratified golden evaluation set.")
    parser.add_argument("--size", type=int, default=200, help="Number of examples to sample (150-250).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--blank-labels",
        action="store_true",
        help="Export with empty true_intent/true_action columns for raw hand labeling.",
    )
    args = parser.parse_args()

    sample_golden_set(
        total_samples=args.size,
        random_seed=args.seed,
        auto_populate_reference=not args.blank_labels
    )
