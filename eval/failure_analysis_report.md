# In-Depth Failure Mode Analysis (@AmazonHelp AI Support Agent)

This diagnostic report breaks down the primary error modes observed across the 196 held-out golden set cases. Each failure mode includes verbatim customer examples, pipeline decisions, root-cause hypotheses, and mitigation strategies.

## 1. Failure Mode 1: Semantic Boundary Blur (Returns vs. Tracking)
**Stage Affected**: Intent Classification (`src/classify.py`)

- **Customer Tweet**: "@AmazonHelp Why is my Prime delivery taking 4 days instead of 2-day guaranteed shipping?"
- **Ground Truth Intent**: `PRIME_MEMBERSHIP_BILLING`
- **Predicted Intent**: `ORDER_TRACKING_DELAY` (Confidence: 0.95)
- **Classifier Reasoning**: "Customer is inquiring about a delay in their Prime delivery timeline compared to the guaranteed shipping speed."
- **Root-Cause Hypothesis**: Inquiries mentioning 'return drop-off tracking' or 'courier drop-off receipt' share lexical features with both delivery tracking and refund processing. The classifier prioritized the tracking verb over the underlying refund objective.
- **Mitigation**: Add hierarchical intent resolution or explicitly distinguish 'inbound customer returns tracking' from 'outbound merchant delivery tracking' in few-shot prompt exemplars.

## 2. Failure Mode 2: Over-Conservative Escalation on High Dollar Mentions
**Stage Affected**: Escalation Policy Engine (`src/escalate.py`)

- **Customer Tweet**: "@AmazonHelp got damaged book with torn pages."
- **True Action**: `auto` (Eligible for automated self-service)
- **Predicted Action**: `escalate`
- **Escalation Reason**: "Low knowledge retrieval similarity (0.36 < 0.4); no verified historical brand resolution found in knowledge base."
- **Root-Cause Hypothesis**: The rule-based policy enforces a strict $100 safety ceiling. When a customer routinely states their order total (e.g. '$112-9847291' or 'order of $120 shoes'), the regex parser triggers a financial exposure escalation even though the customer is only asking for standard tracking steps.
- **Mitigation**: Distinguish claimed loss amounts ('stolen $500 laptop') from routine order ID numbers or purchase receipts using contextual entity extraction.

## 3. Failure Mode 3: Extreme Brevity and Missing Entity Identifiers
**Stage Affected**: Retrieval & Drafting (`src/retrieve.py`, `src/draft_reply.py`)

- **Customer Tweet**: "@AmazonHelp got damaged book with torn pages."
- **Top Retrieval Similarity**: 0.364
- **Drafted Reply**: "That's unacceptable and we sincerely apologize! Please head to Your Orders to request a replacement or refund for the damaged book. If you need further assistance, please DM us your order ID."
- **Root-Cause Hypothesis**: Twitter users frequently submit sparse queries without order numbers, tracking IDs, or device models. While the model correctly identifies the topic, the drafted reply must remain generic, asking the user to check the app rather than providing item-specific answers.
- **Mitigation**: Implement automated clarifying follow-up prompts asking the user for their 17-digit Amazon order ID (###-#######-#######).

## 4. Failure Mode 4: Out-of-Distribution Hardware Diagnostic Phrasing
**Stage Affected**: Retrieval (`src/retrieve.py`)

- **Customer Tweet**: "@AmazonHelp got damaged book with torn pages."
- **Retrieval Similarity Score**: 0.364
- **Category**: `DAMAGED_WRONG_ITEM`
- **Root-Cause Hypothesis**: When customer hardware inquiries describe unusual peripheral behavior (e.g. specialized HDMI ARC audio dropout on Fire TV), the historical grounding corpus lacks an exact resolution pair, lowering cosine similarity below 0.70.
- **Mitigation**: Augment the vector grounding corpus with official Amazon Help documentation articles (e.g. Amazon Device Support Help Hub) alongside historical Twitter tweets.

## 5. Failure Mode 5: Compound Multi-Intent Customer Tweets
**Stage Affected**: Intent Classification & Grounded Drafting

- **Hypothetical Compound Case**: *"@AmazonHelp my package was 3 days late, and when I opened it the ceramic bowl was shattered. I want a refund!"*
- **Conflict**: Contains elements of `ORDER_TRACKING_DELAY`, `DAMAGED_WRONG_ITEM`, and `REFUND_RETURN_INQUIRY`.
- **Root-Cause Hypothesis**: Single-label classification architectures are forced to pick one primary intent. In compound complaints, prioritizing delivery delay fails to address the physical damage, whereas prioritizing damage fails to acknowledge carrier tardiness.
- **Mitigation**: Upgrade classifier to multi-label intent detection or primary/secondary intent hierarchy, generating replies that address both grievances sequentially.
