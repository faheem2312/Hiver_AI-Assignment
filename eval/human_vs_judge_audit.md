# Human QA Auditor vs. Gemini LLM Judge Agreement Report

Evaluated on 20 actual pipeline replies on held-out Kaggle @AmazonHelp customer tweets.

## Summary Statistics

- **Quadratic Weighted Kappa**: 0.265
- **Close Agreement Rate (within 0.75)**: 95.0%
- **Acceptable Agreement Rate (within 1.0)**: 100.0%
- **Score Correlation**: 0.787

## Detailed Audit Table

        Case ID  Human Score  Judge Score  Delta  Groundedness  Safety Verdict Alignment
  audit_1_42553         4.75         5.00   0.25           5.0     5.0             AGREE
  audit_2_99967         4.50         5.00   0.50           5.0     5.0             AGREE
  audit_3_58079         4.38         5.00   0.62           5.0     5.0             AGREE
  audit_4_63901         5.00         5.00   0.00           5.0     5.0             AGREE
  audit_5_17641         4.38         5.00   0.62           5.0     5.0             AGREE
  audit_6_11084         3.88         4.75   0.87           5.0     5.0          DISAGREE
  audit_7_52197         3.50         3.75   0.25           4.0     5.0             AGREE
  audit_8_96743         4.25         5.00   0.75           5.0     5.0             AGREE
  audit_9_86038         4.62         4.75   0.13           5.0     5.0             AGREE
 audit_10_26932         5.00         5.00   0.00           5.0     5.0             AGREE
 audit_11_64224         4.50         5.00   0.50           5.0     5.0             AGREE
 audit_12_46561         3.25         3.75   0.50           5.0     5.0             AGREE
audit_13_111800         4.62         4.75   0.13           5.0     5.0             AGREE
 audit_14_11981         4.38         5.00   0.62           5.0     5.0             AGREE
 audit_15_69669         4.62         4.75   0.13           5.0     5.0             AGREE
 audit_16_38095         5.00         5.00   0.00           5.0     5.0             AGREE
 audit_17_17661         5.00         5.00   0.00           5.0     5.0             AGREE
 audit_18_73430         4.25         5.00   0.75           5.0     5.0             AGREE
 audit_19_76099         4.38         5.00   0.62           5.0     5.0             AGREE
 audit_20_30493         4.88         5.00   0.12           5.0     5.0             AGREE

## Auditor Qualitative Notes & Failure Observations

### audit_1_42553
- **Human Score**: 4.75 | **Judge Score**: 5.0
- **Human Auditor Note**: Empathetic brand tone with valid contact link; properly acknowledges carrier confusion without false promises.
- **Judge Rationale Snippet**: The agent provides a professional, empathetic response that directly addresses t...

### audit_2_99967
- **Human Score**: 4.5 | **Judge Score**: 5.0
- **Human Auditor Note**: Good empathy and helpful escalation to chat for promotional reorder system limit override.
- **Judge Rationale Snippet**: The agent correctly identifies that the issue requires account-specific interven...

### audit_3_58079
- **Human Score**: 4.38 | **Judge Score**: 5.0
- **Human Auditor Note**: Explains Twitter privacy boundary well, though slightly canned; directs to authenticated support.
- **Judge Rationale Snippet**: The agent correctly identifies that account-specific shipping details cannot be ...

### audit_4_63901
- **Human Score**: 5.0 | **Judge Score**: 5.0
- **Human Auditor Note**: Exemplary navigation guidance breaking the customer out of the app link loop.
- **Judge Rationale Snippet**: The agent provides accurate, actionable steps to navigate the Amazon app, direct...

### audit_5_17641
- **Human Score**: 4.38 | **Judge Score**: 5.0
- **Human Auditor Note**: De-escalates hostile tweet calmly and provides direct link to technical support.
- **Judge Rationale Snippet**: The agent provides a professional, empathetic response that acknowledges the cus...

### audit_6_11084
- **Human Score**: 3.88 | **Judge Score**: 4.75
- **Human Auditor Note**: Noticeable flaw: hallucinated customer name 'Mildred' from retrieved historical exemplar. Explains content availability well.
- **Judge Rationale Snippet**: The agent addresses both customer concerns accurately and maintains a profession...

### audit_7_52197
- **Human Score**: 3.5 | **Judge Score**: 3.75
- **Human Auditor Note**: Too generic; fails to address the delivery address instruction issue for misdelivered parcels.
- **Judge Rationale Snippet**: The reply is polite and safe, but it fails to address the specific issue of misd...

### audit_8_96743
- **Human Score**: 4.25 | **Judge Score**: 5.0
- **Human Auditor Note**: Hallucinated name 'Jennifer' from retrieval pair, but provides excellent actionable advice to reorder immediately.
- **Judge Rationale Snippet**: The agent provides a standard, accurate, and safe resolution by directing the cu...

### audit_9_86038
- **Human Score**: 4.62 | **Judge Score**: 4.75
- **Human Auditor Note**: Polite explanation directing customer to estimated delivery date in confirmation email.
- **Judge Rationale Snippet**: The agent provides a professional, empathetic response that directs the customer...

### audit_10_26932
- **Human Score**: 5.0 | **Judge Score**: 5.0
- **Human Auditor Note**: High quality technical guidance suggesting TuneIn and Spotify playlist workarounds.
- **Judge Rationale Snippet**: The agent provides accurate, actionable troubleshooting steps for playing podcas...

### audit_11_64224
- **Human Score**: 4.5 | **Judge Score**: 5.0
- **Human Auditor Note**: Warns customer about public privacy, avoids making financial commitments over Twitter.
- **Judge Rationale Snippet**: The agent's response is perfectly aligned with historical brand resolutions by d...

### audit_12_46561
- **Human Score**: 3.25 | **Judge Score**: 3.75
- **Human Auditor Note**: Weak canned fallback; fails to guide the customer on requesting a replacement for the wrong book edition.
- **Judge Rationale Snippet**: The reply is safe and grounded but fails to address the customer's specific frus...

### audit_13_111800
- **Human Score**: 4.62 | **Judge Score**: 4.75
- **Human Auditor Note**: Handles high customer agitation diplomatically, protecting personal data.
- **Judge Rationale Snippet**: The reply is highly grounded in historical resolutions and maintains a professio...

### audit_14_11981
- **Human Score**: 4.38 | **Judge Score**: 5.0
- **Human Auditor Note**: Empathetic acknowledgment of repeated service friction with direct support link.
- **Judge Rationale Snippet**: The agent provides a professional, empathetic response that acknowledges the cus...

### audit_15_69669
- **Human Score**: 4.62 | **Judge Score**: 4.75
- **Human Auditor Note**: Valid diagnostic question asking about merchant cancellation notification in spam folder.
- **Judge Rationale Snippet**: The agent provides a professional and empathetic response that adheres to standa...

### audit_16_38095
- **Human Score**: 5.0 | **Judge Score**: 5.0
- **Human Auditor Note**: Accurate grounding in Amazon Marketplace 2-business-day third-party seller resolution window.
- **Judge Rationale Snippet**: The agent provides accurate, policy-aligned guidance regarding the A-to-z Guaran...

### audit_17_17661
- **Human Score**: 5.0 | **Judge Score**: 5.0
- **Human Auditor Note**: Warm, professional brand appreciation response to positive customer feedback.
- **Judge Rationale Snippet**: The agent appropriately acknowledges the customer's positive feedback while main...

### audit_18_73430
- **Human Score**: 4.25 | **Judge Score**: 5.0
- **Human Auditor Note**: Hallucinated name 'Michael' from grounding corpus, but policy timeline (2 days) is correct.
- **Judge Rationale Snippet**: The agent correctly identifies the standard 48-hour response window for third-pa...

### audit_19_76099
- **Human Score**: 4.38 | **Judge Score**: 5.0
- **Human Auditor Note**: Polite de-escalation of impatient customer, appropriately emphasizing security.
- **Judge Rationale Snippet**: The agent provides a professional, empathetic response that acknowledges the cus...

### audit_20_30493
- **Human Score**: 4.88 | **Judge Score**: 5.0
- **Human Auditor Note**: Clear self-service guidance for tampered/damaged package replacement in Your Orders.
- **Judge Rationale Snippet**: The agent provides accurate, policy-compliant self-service instructions for a da...

