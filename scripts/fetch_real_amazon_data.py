"""
Stream and filter real @AmazonHelp tweets directly from the Kaggle TWCS dataset on Hugging Face.
Extracts genuine, noisy, real-world customer support tweets without requiring a 2.3GB manual download.
"""

import csv
import sys
import logging
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV_OUT = PROJECT_ROOT / "data" / "raw" / "twcs_amazon_real.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

HF_TWCS_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
TARGET_BRAND = "AmazonHelp"


def stream_real_amazon_tweets(max_tweets: int = 15000):
    RAW_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Connecting to live TWCS Kaggle stream on Hugging Face: {HF_TWCS_URL}")
    logger.info(f"Targeting brand: '{TARGET_BRAND}' (up to {max_tweets:,} real tweets)...")

    matched_count = 0
    total_scanned = 0

    with requests.get(HF_TWCS_URL, stream=True) as response:
        response.raise_for_status()
        
        # Read lines decoded as utf-8
        lines = (line.decode("utf-8", errors="ignore") for line in response.iter_lines())
        reader = csv.reader(lines)
        
        # Extract header
        header = next(reader)
        logger.info(f"Detected raw dataset columns: {header}")

        # Indices
        tweet_id_idx = header.index("tweet_id")
        author_id_idx = header.index("author_id")
        inbound_idx = header.index("inbound")
        created_at_idx = header.index("created_at")
        text_idx = header.index("text")
        response_tweet_id_idx = header.index("response_tweet_id")
        in_response_to_tweet_id_idx = header.index("in_response_to_tweet_id")

        with open(RAW_CSV_OUT, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(header)

            for row in reader:
                if len(row) < len(header):
                    continue
                total_scanned += 1

                author = row[author_id_idx]
                text = row[text_idx]

                # Match real tweets authored by AmazonHelp OR mentioning @AmazonHelp
                is_amazon = (author == TARGET_BRAND) or (f"@{TARGET_BRAND}".lower() in text.lower())

                if is_amazon:
                    writer.writerow(row)
                    matched_count += 1

                    if matched_count % 1000 == 0:
                        logger.info(f"Extracted {matched_count:,} real @AmazonHelp tweets (scanned {total_scanned:,} rows)...")

                    if matched_count >= max_tweets:
                        logger.info(f"Reached quota of {max_tweets:,} real @AmazonHelp tweets.")
                        break

    logger.info(f"[SUCCESS] Saved {matched_count:,} genuine @AmazonHelp tweets to {RAW_CSV_OUT}")
    return matched_count


if __name__ == "__main__":
    stream_real_amazon_tweets(max_tweets=12000)
