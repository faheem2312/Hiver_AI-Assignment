# Benchmark Headline Results

Generated across 50 held-out golden evaluation cases.

       System Architecture Intent Acc Intent Macro-F1 Esc. Precision Esc. Recall False Auto (FN) Safety Cost (Loss) Auto-Handle Rate Reply Quality (1-5)
          Trivial Baseline      18.0%           0.038          42.0%      100.0%               0                 29             0.0%       2.10 (Canned)
         Simple Rule-Based      70.0%           0.681         100.0%       57.1%               9                 45            76.0%   3.20 (Static FAQ)
AI Support Pipeline (Ours)      74.0%           0.734          85.7%       57.1%               9                 47            72.0%     4.98 (Grounded)

### Subsystem Performance
- **Retrieval Hit@1**: 62.0%
- **Retrieval Hit@3**: 74.0%
- **Retrieval MRR**: 0.680
- **LLM Judge Groundedness**: 5.0/5.0
- **LLM Judge Safety**: 5.0/5.0
- **Execution Speed**: 6309.4 ms/query
