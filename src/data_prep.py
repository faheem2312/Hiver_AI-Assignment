"""
Data Preparation Pipeline for Real @AmazonHelp Twitter Customer Support Dataset
================================================================================
Processes genuine, real-world Kaggle Twitter support data (twcs_amazon_real.csv / twcs.csv):
1. Ingests real raw tweets.
2. Filters for English-language conversations.
3. Reconstructs multi-turn conversation threads using `in_response_to_tweet_id` parent-child chains.
4. Pairs inbound customer messages with real AmazonHelp brand responses.
5. Applies the non-resolution filter (`is_informative_resolution`) to separate informative
   troubleshooting guidance from generic canned brush-offs ("Please DM us / click here").
6. Saves clean real-world processed threads to data/processed/amazon_threads_subsample.csv.
7. Outputs genuine dataset statistics.
"""

import os
import re
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BRAND_HANDLE = "AmazonHelp"


# =====================================================================
# 1. Non-Resolution Filtering Heuristic (Applied to Real Brand Tweets)
# =====================================================================
# Twitter brand accounts frequently deploy low-effort canned redirects like:
# "Please DM us" or "Reach us via https://t.co/...".
# These contain zero troubleshooting value. We filter them out of the grounding corpus.

CAN_REDIRECT_PATTERNS = [
    r"^(?:hi|hello|hey)?\s*(?:there,?)?\s*(?:please\s+)?(?:send\s+us\s+a\s+dm|dm\s+us|reach\s+out\s+via\s+dm|message\s+us\s+in\s+dm)\b",
    r"^(?:please\s+)?(?:reach\s+us|contact\s+us)\s+(?:by\s+phone\s+or\s+chat\s+here|at)\s*:\s*https?://\S+\b",
    r"^please\s+provide\s+your\s+order\s+details\s+via\s+dm\b",
    r"^(?:please\s+)?click\s+here\s+to\s+chat\s*:\s*https?://\S+$",
]

INFORMATIVE_KEYWORDS = [
    "carrier", "tracking", "delivery window", "delivered", "driver", "courier",
    "porch", "neighbor", "mailbox", "business days", "refund", "replacement",
    "return window", "your orders", "manage prime", "restart", "settings",
    "unplug", "firmware", "billing", "statement", "bank", "authorization",
    "otp", "two-step", "password reset", "apologize", "problem with order",
    "dispatch", "shipped", "shipping address", "locker"
]


def is_informative_resolution(text: str) -> Tuple[bool, str]:
    """Evaluates whether a real brand response contains substantive grounding information."""
    if not isinstance(text, str):
        return False, "EMPTY_TEXT"

    clean_text = text.strip()
    words = clean_text.split()

    if len(words) < 6:
        return False, "TOO_SHORT"

    lower_text = clean_text.lower()
    for pattern in CAN_REDIRECT_PATTERNS:
        if re.search(pattern, lower_text):
            return False, "CANNED_DM_REDIRECT"

    asks_for_dm_or_link = bool(re.search(r"\b(dm|direct message|private message|reach us by phone|chat here)\b", lower_text))
    has_informative_content = any(kw in lower_text for kw in INFORMATIVE_KEYWORDS)

    if asks_for_dm_or_link and not has_informative_content and len(words) < 16:
        return False, "PURE_DM_REQUEST_NO_GUIDANCE"

    return True, "INFORMATIVE_RESOLUTION"


def is_english_tweet(text: str) -> bool:
    """Filters out non-English tweets (AmazonHelp operates globally in Japanese, German, Italian, etc.)."""
    if not isinstance(text, str):
        return False
    # Check ASCII character density to filter Japanese/CJK and other non-Latin scripts
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) == 0 or (ascii_chars / len(text)) < 0.85:
        return False
    # Common English stopwords to ensure linguistic grounding
    en_stops = {"the", "is", "at", "which", "on", "my", "to", "a", "for", "of", "with", "i", "you", "it", "have", "this", "your", "in", "and", "be", "not", "we", "me", "so", "can", "do", "no", "what", "when", "how", "why", "who", "from", "but", "an", "all", "been", "was", "will", "would", "could", "should", "our", "their", "if", "or", "as", "by", "up", "out"}
    words = set(re.findall(r"\b[a-z]{1,15}\b", text.lower()))
    if len(words.intersection(en_stops)) < 2:
        return False
    # Filter common German/Spanish/French/Italian Amazon handles or tokens
    lower = text.lower()
    non_en_tokens = ["wir bedauern", "bitte sende", "hola", "danke für", "kundenservice", "per dm", "dovreste", "grazie", "perché", "ciao", "buongiorno", "merci", "por favor", "gracias"]
    if any(tok in lower for tok in non_en_tokens):
        return False
    return True


def categorize_customer_intent(text: str) -> str:
    """Categorizes raw customer tweet into one of 8 canonical support intents."""
    if not isinstance(text, str):
        return "OTHER_GENERAL"
    t = text.lower()
    if re.search(r"\b(lawyer|attorney|police|sue|legal|court|stolen|theft|thief|fraud|crime|injury|safety hazard)\b", t):
        return "ESCALATION_HIGH_RISK"
    if re.search(r"\b(otp|2fa|verification code|locked out|hacked|password|login|log in|sign in|compromised)\b", t):
        return "ACCOUNT_LOGIN_SECURITY"
    if re.search(r"\b(prime|membership|subscription|charged|double charge|billed|billing|fee|renewal)\b", t):
        return "PRIME_MEMBERSHIP_BILLING"
    if re.search(r"\b(damaged|broken|shattered|cracked|defective|smashed|wrong item|incorrect item|missing item)\b", t):
        return "DAMAGED_WRONG_ITEM"
    if re.search(r"\b(refund|return|returned|returning|send back|cancel|cancellation)\b", t):
        return "REFUND_RETURN_INQUIRY"
    if re.search(r"\b(track|tracking|package|parcel|courier|delivery|delivered|dispatch|shipped|delay|late|carrier|arrived|order)\b", t):
        return "ORDER_TRACKING_DELAY"
    if re.search(r"\b(kindle|fire tv|firestick|alexa|echo|app|firmware|wifi|bluetooth|setup|reboot|device)\b", t):
        return "PRODUCT_TECH_SUPPORT"
    return "OTHER_GENERAL"


# =====================================================================
# 2. Ingest Real Data
# =====================================================================
def get_real_data_path() -> Path:
    """Finds raw real-world dataset on disk, or prompts to stream it."""
    candidates = [
        RAW_DATA_DIR / "twcs_amazon_real.csv",
        RAW_DATA_DIR / "twcs.csv",
        RAW_DATA_DIR / "amazon_tweets_raw.csv"
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 100000:
            return c

    # If missing, run streamer automatically
    logger.info("Real dataset missing in data/raw/. Invoking live Kaggle TWCS stream...")
    from scripts.fetch_real_amazon_data import stream_real_amazon_tweets
    stream_real_amazon_tweets(max_tweets=12000)
    return RAW_DATA_DIR / "twcs_amazon_real.csv"


def load_real_raw_tweets(csv_path: Path) -> pd.DataFrame:
    """Loads and deduplicates real tweets."""
    logger.info(f"Loading real tweets from {csv_path.name}...")
    df = pd.read_csv(
        csv_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": str,
            "created_at": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        low_memory=False
    )
    logger.info(f"Loaded {len(df):,} total raw tweets.")
    
    # Filter for English tweets
    df["is_en"] = df["text"].apply(is_english_tweet)
    df_en = df[df["is_en"] == True].drop(columns=["is_en"]).reset_index(drop=True)
    logger.info(f"Retained {len(df_en):,} real English tweets.")
    return df_en


# =====================================================================
# 3. Thread Reconstruction on Real Data
# =====================================================================
def reconstruct_real_threads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs real customer-agent dialogue pairs by linking:
    Customer Tweet (inbound=True) -> Brand Reply (in_response_to_tweet_id == customer_tweet_id)
    """
    logger.info("Reconstructing real conversation chains via in_response_to_tweet_id...")
    df["tweet_id"] = df["tweet_id"].astype(str)
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

    brand_replies = df[df["author_id"] == BRAND_HANDLE].copy()
    customer_tweets = df[df["inbound"] == True].copy()

    # Map: customer_tweet_id -> brand reply row
    brand_reply_map: Dict[str, Dict[str, Any]] = {}
    for _, row in brand_replies.iterrows():
        parent_id = row["in_response_to_tweet_id"]
        if parent_id and parent_id != "" and parent_id != "nan":
            if parent_id not in brand_reply_map:
                brand_reply_map[parent_id] = row.to_dict()

    threads = []
    for _, cust_row in customer_tweets.iterrows():
        t_id = cust_row["tweet_id"]
        if t_id in brand_reply_map:
            brand_turn = brand_reply_map[t_id]
            brand_text = brand_turn["text"]
            is_info, filter_reason = is_informative_resolution(brand_text)

            threads.append({
                "thread_id": f"th_{t_id}",
                "customer_tweet_id": t_id,
                "customer_user_id": cust_row["author_id"],
                "customer_text": cust_row["text"],
                "brand_tweet_id": brand_turn["tweet_id"],
                "brand_text": brand_text,
                "created_at": cust_row.get("created_at", ""),
                "is_informative_resolution": is_info,
                "filter_reason": filter_reason,
                "category_tag": categorize_customer_intent(cust_row["text"])
            })

    thread_df = pd.DataFrame(threads)
    # Deduplicate by customer tweet text to ensure diverse organic interactions
    thread_df = thread_df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    logger.info(f"Reconstructed {len(thread_df):,} real customer-agent conversation threads.")
    return thread_df


def print_summary_statistics(thread_df: pd.DataFrame):
    """Prints comprehensive stats of real Twitter data."""
    total_threads = len(thread_df)
    informative_count = thread_df["is_informative_resolution"].sum()
    canned_count = total_threads - informative_count

    avg_cust_len = thread_df["customer_text"].str.split().str.len().mean()
    avg_brand_len = thread_df["brand_text"].str.split().str.len().mean()

    print("\n" + "=" * 60)
    print("   Real @AmazonHelp Twitter Data Preparation Summary Stats    ")
    print("=" * 60)
    print(f"Total reconstructed real threads : {total_threads:,}")
    print(f"Informative resolutions          : {informative_count:,} ({informative_count/total_threads*100:.1f}%)")
    print(f"Filtered canned/non-res          : {canned_count:,} ({canned_count/total_threads*100:.1f}%)")
    print(f"Avg customer message words       : {avg_cust_len:.1f}")
    print(f"Avg brand reply words            : {avg_brand_len:.1f}")

    if "created_at" in thread_df.columns and not thread_df["created_at"].isna().all():
        print(f"Date range sample                : {thread_df['created_at'].iloc[0]} -> {thread_df['created_at'].iloc[-1]}")

    print("\nFilter Reason Breakdown on Real Tweets:")
    print(thread_df["filter_reason"].value_counts().to_string())
    print("=" * 60 + "\n")


def prepare_real_data(sample_size: int = 5000, force_reload: bool = True) -> pd.DataFrame:
    """Main orchestration pipeline using genuine Twitter data."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_csv = PROCESSED_DATA_DIR / "amazon_threads_subsample.csv"

    if not force_reload and processed_csv.exists():
        thread_df = pd.read_csv(processed_csv)
        print_summary_statistics(thread_df)
        return thread_df

    raw_path = get_real_data_path()
    raw_df = load_real_raw_tweets(raw_path)
    thread_df = reconstruct_real_threads(raw_df)

    if len(thread_df) > sample_size:
        thread_df = thread_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    thread_df.to_csv(processed_csv, index=False)
    logger.info(f"Saved real processed threads to {processed_csv}")
    print_summary_statistics(thread_df)
    return thread_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process real @AmazonHelp Twitter support threads.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of threads to subsample.")
    parser.add_argument("--force-reload", action="store_true", default=True, help="Force reload raw real data.")
    args = parser.parse_args()

    prepare_real_data(sample_size=args.sample_size, force_reload=args.force_reload)
