# Running Decision Log

Every non-obvious design choice made throughout the project is recorded here chronologically with a direct rationale.

| # | Date | Decision | Why (Rationale) |
|---|------|----------|-----------------|
| 1 | 2026-09-09 | Brand chosen: `@AmazonHelp` | High volume of real customer friction with sharp policy boundaries (self-service vs. human-only financial concessions). |
| 2 | 2026-09-09 | Use official `google-genai` SDK (not legacy `google-generativeai`) | Google deprecated `google-generativeai`; the new SDK is the official path forward for Gemini 2.x/1.5 models. |
| 3 | 2026-09-09 | Default model recommendation: `gemini-3.1-flash-lite` | Highest free-tier allowance (15 RPM / 500 RPD vs 5 RPM / 20 RPD on standard flash), preventing quota starvation during golden set evaluation. |
| 4 | 2026-09-09 | Centralized exponential backoff retry in `gemini_client.py` | Google AI Studio free-tier enforces strict RPM limits; deterministic retry avoids benchmark crashes. |
| 5 | 2026-09-09 | Local `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings | Avoids consuming Gemini API quota on embeddings and guarantees reproducible, zero-network-latency retrieval. |
| 6 | 2026-09-09 | Plain Python + in-memory FAISS/NumPy (no LangChain / LlamaIndex) | Guarantees code readability, zero abstraction bloat, and easy live debugging during interviews. |
| 7 | 2026-09-09 | Project directory placed on Desktop (`Desktop/hiver-ai-agent`) | Explicit user preference for direct accessibility and manual Git commit/push control. |
| 8 | 2026-09-09 | Support flexible `GEMINI_MODEL` in `.env` | Allows swapping seamlessly between `gemini-3.1-flash` (higher reasoning) and `gemini-3.1-flash-lite` (higher RPD for full eval runs). |
| 9 | 2026-09-10 | Canned response filter in `src/data_prep.py` (`is_informative_resolution`) | Pure "DM us" tweets contain zero factual guidance; excluding them keeps the retrieval corpus informative and prevents the agent from generating trivial brush-offs. |
| 10 | 2026-09-10 | Parent-child thread reconstruction via `in_response_to_tweet_id` | Flat Twitter feeds must be converted into customer inquiry + brand resolution pairs to enable grounded RAG. |
| 11 | 2026-09-10 | Chunked ingest & cached 5,000 thread subsample (`amazon_threads_subsample.csv`) | Bypasses the 2.3GB dataset bottleneck, allowing the entire pipeline and eval harness to reproduce in <15 minutes. |
| 12 | 2026-09-10 | Fixed 8-intent Enum in `src/intents.py` as single source of truth | Prevents string label drift across classifier, retrieval booster, and evaluation metrics; easily imported across all modules. |
| 13 | 2026-09-10 | Local Sentence-Transformers with TF-IDF fallback for cluster discovery | Guarantees zero API quota burn during exploratory unsupervised clustering and runs in seconds on local CPU. |
| 14 | 2026-09-10 | Policy routing metadata embedded directly in Intent Enum | Codifies business boundaries (`default_action`, `is_high_risk`) so self-service vs. human escalation logic remains transparent and auditable. |
| 15 | 2026-09-10 | Stratified 200-sample Golden Set with fixed seed (42) | Ensures balanced statistical power across all 8 problem domains while respecting Gemini free-tier daily quotas during eval runs. |
| 16 | 2026-09-10 | Strict physical isolation of held-out set from grounding corpus | Excludes the 200 golden items from `amazon_grounding_corpus.csv` (4,800 items) to prevent retrieval data leakage and artificial hit-rate inflation. |
| 17 | 2026-09-10 | Trivial baseline: majority intent + 100% escalation | Establishes the floor: tests whether an AI agent actually reduces human escalation without collapsing accuracy. |
| 18 | 2026-09-10 | Simple baseline: regex keyword matching + static FAQ templates | Benchmarks against standard rule-based chatbots to quantify the exact marginal gain of LLM intent reasoning & RAG. |
| 19 | 2026-09-10 | Asymmetric escalation cost metric (`5*FN + 1*FP`) | False Auto-Handles (ignoring fraud, legal, theft) incur catastrophic churn/risk compared to harmless false escalations. |
