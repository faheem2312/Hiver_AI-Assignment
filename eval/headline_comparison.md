# Benchmark Headline Results

Generated across 196 held-out golden evaluation cases.

       System Architecture Intent Acc Intent Macro-F1 Esc. Precision Esc. Recall False Auto (FN) Safety Cost (Loss) Auto-Handle Rate Reply Quality (1-5)
          Trivial Baseline      14.3%           0.036          28.6%      100.0%               0                140             0.0%       2.10 (Canned)
         Simple Rule-Based      88.8%           0.799          90.3%      100.0%               0                  6            68.4%   3.20 (Static FAQ)
AI Support Pipeline (Ours)      94.4%           0.942          77.8%      100.0%               0                 16            63.3%     4.91 (Grounded)

### Subsystem Performance
- **Retrieval Hit@1**: 100.0%
- **Retrieval Hit@3**: 100.0%
- **Retrieval MRR**: 1.000
- **LLM Judge Groundedness**: 4.85/5.0
- **LLM Judge Safety**: 4.95/5.0
- **Execution Speed**: 2343.5 ms/query
