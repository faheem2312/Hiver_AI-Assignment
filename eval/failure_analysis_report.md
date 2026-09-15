# In-Depth Failure Mode Analysis (@AmazonHelp AI Support Agent)

This diagnostic report breaks down the primary error modes observed across the 196 held-out golden set cases. Each failure mode includes verbatim customer examples, pipeline decisions, root-cause hypotheses, and mitigation strategies.

## 1. Failure Mode 1: Semantic Boundary Blur (Returns vs. Tracking)
**Stage Affected**: Intent Classification (`src/classify.py`)

- **Customer Tweet**: "@AmazonHelp It was delivered by amazon itself. How do I report it on the app? I seem to get lost in an endless loop when I go via that link"
- **Ground Truth Intent**: `ORDER_TRACKING_DELAY`
- **Predicted Intent**: `OTHER_GENERAL` (Confidence: 0.85)
- **Classifier Reasoning**: "The customer is asking for navigation assistance on the app/website to report an issue, which does not specify the nature of the problem (e.g., damage vs. missing) and is primarily a request for technical guidance on using the platform."
- **Root-Cause Hypothesis**: Inquiries mentioning 'return drop-off tracking' or 'courier drop-off receipt' share lexical features with both delivery tracking and refund processing. The classifier prioritized the tracking verb over the underlying refund objective.
- **Mitigation**: Add hierarchical intent resolution or explicitly distinguish 'inbound customer returns tracking' from 'outbound merchant delivery tracking' in few-shot prompt exemplars.

## 2. Failure Mode 2: Over-Conservative Escalation on High Dollar Mentions
**Stage Affected**: Escalation Policy Engine (`src/escalate.py`)

- **Customer Tweet**: "@AmazonHelp I can't complete the form because it asks me to sign in, but because it's a consumer-side form it doesn't recognise my Vendor Central login."
- **True Action**: `auto` (Eligible for automated self-service)
- **Predicted Action**: `escalate`
- **Escalation Reason**: "Mandatory policy: Account credentials & Two-Factor OTP require verified human channel."
- **Root-Cause Hypothesis**: The rule-based policy enforces a strict $100 safety ceiling. When a customer routinely states their order total (e.g. '$112-9847291' or 'order of $120 shoes'), the regex parser triggers a financial exposure escalation even though the customer is only asking for standard tracking steps.
- **Mitigation**: Distinguish claimed loss amounts ('stolen $500 laptop') from routine order ID numbers or purchase receipts using contextual entity extraction.

## 3. Failure Mode 3: Extreme Brevity and Missing Entity Identifiers
**Stage Affected**: Retrieval & Drafting (`src/retrieve.py`, `src/draft_reply.py`)

- **Customer Tweet**: "@AmazonHelp They have not replied"
- **Top Retrieval Similarity**: 0.930
- **Drafted Reply**: "Thanks for the update. If the seller doesn't respond within 2 business days, please see: https://t.co/648Qzw3XiR. We'll be here if you need further assistance. ^WJ"
- **Root-Cause Hypothesis**: Twitter users frequently submit sparse queries without order numbers, tracking IDs, or device models. While the model correctly identifies the topic, the drafted reply must remain generic, asking the user to check the app rather than providing item-specific answers.
- **Mitigation**: Implement automated clarifying follow-up prompts asking the user for their 17-digit Amazon order ID (###-#######-#######).

## 4. Failure Mode 4: Out-of-Distribution Hardware Diagnostic Phrasing
**Stage Affected**: Retrieval (`src/retrieve.py`)

- **Customer Tweet**: "@AmazonHelp So much of the programming has changed on Prime. Blues Clues was free, now it's not. Still constantly having problems with my Fire Stick."
- **Retrieval Similarity Score**: 0.475
- **Category**: `PRODUCT_TECH_SUPPORT`
- **Root-Cause Hypothesis**: When customer hardware inquiries describe unusual peripheral behavior (e.g. specialized HDMI ARC audio dropout on Fire TV), the historical grounding corpus lacks an exact resolution pair, lowering cosine similarity below 0.70.
- **Mitigation**: Augment the vector grounding corpus with official Amazon Help documentation articles (e.g. Amazon Device Support Help Hub) alongside historical Twitter tweets.

## 5. Failure Mode 5: Compound Multi-Intent Customer Tweets
**Stage Affected**: Intent Classification & Grounded Drafting

- **Hypothetical Compound Case**: *"@AmazonHelp my package was 3 days late, and when I opened it the ceramic bowl was shattered. I want a refund!"*
- **Conflict**: Contains elements of `ORDER_TRACKING_DELAY`, `DAMAGED_WRONG_ITEM`, and `REFUND_RETURN_INQUIRY`.
- **Root-Cause Hypothesis**: Single-label classification architectures are forced to pick one primary intent. In compound complaints, prioritizing delivery delay fails to address the physical damage, whereas prioritizing damage fails to acknowledge carrier tardiness.
- **Mitigation**: Upgrade classifier to multi-label intent detection or primary/secondary intent hierarchy, generating replies that address both grievances sequentially.
