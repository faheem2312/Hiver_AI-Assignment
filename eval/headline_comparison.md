# Benchmark Headline Results

Generated across 50 held-out golden evaluation cases.

       System Architecture Intent Acc Intent Macro-F1 Esc. Precision Esc. Recall False Auto (FN) Safety Cost (Loss) Auto-Handle Rate Reply Quality (1-5)
          Trivial Baseline       6.0%           0.014          44.0%      100.0%               0                 28             0.0%       2.10 (Canned)
         Simple Rule-Based      84.0%           0.813          91.7%       50.0%              11                 56            76.0%   3.20 (Static FAQ)
AI Support Pipeline (Ours)      58.0%           0.569          86.7%       59.1%               9                 47            70.0%     4.90 (Grounded)

### Subsystem Performance
- **Retrieval Hit@1**: 58.0%
- **Retrieval Hit@3**: 66.0%
- **Retrieval MRR**: 0.617
- **LLM Judge Groundedness**: 4.93/5.0
- **LLM Judge Safety**: 5.0/5.0
- **Execution Speed**: 7740.2 ms/query
