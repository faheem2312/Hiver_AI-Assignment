# Golden Evaluation Set Methodology & Annotation Guidelines

This document records the exact sampling strategy, data hygiene guarantees, and labeling rubric for the 200-example `@AmazonHelp` golden benchmark (`eval/golden_set.csv`).

---

## 1. Sampling Methodology & Data Hygiene

### 1.1 Stratification Strategy
To prevent evaluation bias toward high-frequency generic tracking issues, we perform **stratified balanced sampling** across customer support categories:
- **Sampling Size**: 200 total examples (well within the required 150–250 range).
- **Random Seed**: `42` for exact reproducibility.
- **Source**: Extracted from `data/processed/amazon_threads_subsample.csv`.

### 1.2 Strict Held-Out Isolation (Zero Retrieval Leakage)
A common flaw in RAG evaluations is "data contamination," where evaluation queries exist inside the vector retrieval index, creating an artificially inflated hit rate.
- **Mechanism**: The 200 golden set items are explicitly removed from the processed dataset.
- **Resulting Grounding Corpus**: The remaining 4,800 threads are saved to `data/processed/amazon_grounding_corpus.csv`.
- The FAISS vector index in Phase 6 indexes **only** `amazon_grounding_corpus.csv`. The golden evaluation set is never seen during retrieval index construction.

---

## 2. Schema Specification

The `eval/golden_set.csv` benchmark adheres strictly to the required fields:

| Column | Type | Allowed Values | Description |
|--------|------|----------------|-------------|
| `message_id` | String | Tweet ID | Unique identifier of the incoming customer inquiry. |
| `text` | String | Free text | Verbatim customer message sent to `@AmazonHelp`. |
| `thread_context` | String | Free text | Associated conversation context or historical agent turn. |
| `true_intent` | Enum | See Intent Taxonomy | The ground-truth customer objective. |
| `has_good_grounding_example` | String | `Y` / `N` | Whether an informative, policy-grounded resolution exists in the brand corpus. |
| `true_action` | Enum | `auto` / `escalate` | The operational routing policy decision. |
| `notes` | String | Free text | Annotator observations, ambiguity notes, or edge cases. |

---

## 3. Annotation Guidelines & Rubric

### 3.1 Intent Assignment
- **`ORDER_TRACKING_DELAY`**: Any inquiry concerning parcel whereabouts, carrier delivery windows, packages marked delivered but missing, or tracking status.
- **`REFUND_RETURN_INQUIRY`**: Inquiries about return eligibility, return drop-offs (UPS/Kohl's), and timeline for refund appearance on bank statements.
- **`DAMAGED_WRONG_ITEM`**: Broken merchandise, shattered items, incorrect size/model delivered, or missing accessories.
- **`PRIME_MEMBERSHIP_BILLING`**: Questions on Prime fee auto-renewals, subscription charges, or cancellation refund requests.
- **`ACCOUNT_LOGIN_SECURITY`**: Two-Factor OTP code delivery failures, locked accounts, and suspected account takeovers.
- **`PRODUCT_TECH_SUPPORT`**: Technical issues with Amazon hardware (Echo yellow lights, Fire TV Stick frozen screens).
- **`ESCALATION_HIGH_RISK`**: Serious real-world risk: property damage by Amazon drivers, stolen high-value electronics (> $100), explicit legal/attorney threats, or police reports.
- **`OTHER_GENERAL`**: Out-of-scope banter, website bugs, or vague greetings.

### 3.2 Action Routing: `auto` vs. `escalate`
The system enforces a **safety-first asymmetric loss** policy:
- **`auto`**: The issue can be resolved with deterministic factual guidance, self-service links (Your Orders, return labels, device power cycle), or policy explanations.
- **`escalate`**: Mandatory human specialist involvement required when:
  1. Financial compensation or authorization is demanded (authorizing refunds, wallet credits).
  2. Investigation of lost/stolen packages exceeding $100.
  3. Sensitive account security / 2FA lockout.
  4. Legal threats, property damage, or acute customer hostility.

### 3.3 Grounding Quality (`has_good_grounding_example`)
- **`Y`**: Historical `@AmazonHelp` agents provided concrete troubleshooting steps, specific timeframes, or standard operating procedures.
- **`N`**: The issue was uniquely customer-specific (e.g. customized property damage dispute) where brand agents historically only used private escalation channels.
