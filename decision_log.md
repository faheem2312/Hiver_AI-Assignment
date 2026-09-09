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
