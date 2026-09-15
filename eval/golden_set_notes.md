# Golden Evaluation Set Methodology & Annotation Guidelines

This document records the exact sampling strategy, data hygiene guarantees, human hand-labeling rubric, and class distribution for the 200-example held-out `@AmazonHelp` golden benchmark (`eval/golden_set.csv`).

---

## 1. Sampling Methodology & Data Hygiene

### 1.1 Stratification & Human Labeling
To avoid the fatal flaw of **circular labeling** (where automated regex rules generate the "ground truth" and artificially inflate keyword-based baselines), all 200 rows were hand-labeled by reading each verbatim customer tweet individually:
- **Sampling Size**: 200 total examples extracted from genuine Kaggle `@AmazonHelp` support threads (`data/raw/twcs_amazon_real.csv`).
- **Elimination of Circular Labeling**: Early automated heuristic tags were discarded. Every row now contains an individual, human-written rationale note confronting tweet ambiguity, colloquial phrasing, sarcasm, and compound grievances.
- **Natural Class Distribution**: Unlike artificial equal-split datasets, human labeling revealed an organic support distribution:
  - `ORDER_TRACKING_DELAY`: 51 (25.5%)
  - `DAMAGED_WRONG_ITEM`: 27 (13.5%)
  - `ACCOUNT_LOGIN_SECURITY`: 25 (12.5%)
  - `REFUND_RETURN_INQUIRY`: 23 (11.5%)
  - `OTHER_GENERAL`: 23 (11.5%)
  - `PRODUCT_TECH_SUPPORT`: 20 (10.0%)
  - `ESCALATION_HIGH_RISK`: 17 (8.5%)
  - `PRIME_MEMBERSHIP_BILLING`: 14 (7.0%)
  - **Routing Action Split**: `auto`: 116 (58.0%) | `escalate`: 84 (42.0%)

### 1.2 Strict Held-Out Isolation (Zero Retrieval Leakage)
A common flaw in RAG evaluations is "data contamination," where evaluation queries exist inside the vector retrieval index, creating an artificially inflated hit rate.
- **Mechanism**: The 200 golden set items are explicitly removed from the processed dataset.
- **Resulting Grounding Corpus**: The remaining 2,592 threads are saved to `data/processed/amazon_grounding_corpus.csv`.
- The FAISS vector index indexes **only** `amazon_grounding_corpus.csv`. The golden evaluation set is never seen during retrieval index construction.

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
| `notes` | String | Free text | Human annotator observations, ambiguity notes, or edge cases. |

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

---

## 4. Key Ambiguity Insights from Hand Labeling

1. **The Keyword False Friend Trap**:
   - Customer Tweet: *"@AmazonHelp There is no option there, just says my refund has been issued as i requested - i did not request a refund!"*
   - Regex would match `refund` and route as routine return/refund inquiry. A human recognizes that the customer is outraged by an unauthorized order cancellation. True intent: `REFUND_RETURN_INQUIRY` with mandatory human escalation (`escalate`).
2. **Casually Mentioned Crime vs. Delivery Feedback**:
   - Customer Tweet: *"@AmazonHelp Not damaged but someone could of stole my parcel so please can you tell your delivery driver"*
   - Regex would flag `stole` and trigger high-risk crime escalation. A human reads that the customer *received* the package safely and is merely leaving driver delivery feedback (`auto`).
3. **Compound Complaints**:
   - Customers frequently stack issues: *"package was 4 days late AND the ceramic bowl was shattered inside! Cancel my Prime!"*
   - Hand-labeling classifies by the primary blocker causing customer churn (`DAMAGED_WRONG_ITEM`) and notes the secondary billing friction in the annotations.
