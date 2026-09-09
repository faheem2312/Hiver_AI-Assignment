# Hiver AI Customer Support Agent (`@AmazonHelp`)

An intelligent, retrieval-grounded customer support pipeline built for `@AmazonHelp` using Twitter Customer Support conversations. Designed for end-to-end auditability, safety-first escalation, and evaluation reproducibility in under 15 minutes.

---

## 🎯 Architecture Overview

```
                        Incoming Customer Tweet
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   1. Intent Classification    │ (Gemini Flash + Structured JSON)
                   └───────────────┬───────────────┘
                                   │
                     Intent & Cleaned Query
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    2. Historical Retrieval    │ (FAISS / MiniLM Embeddings)
                   │  Top-k Brand Resolution Pairs │
                   └───────────────┬───────────────┘
                                   │
                    Retrieved Context + Message
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│     3. Escalation Policy      │           │    4. Grounded Reply Draft    │
│  (Rules: Risk/Policy/Refund)  │           │  (Gemini strictly grounded    │
│    -> auto vs. escalate       │           │   in retrieved resolutions)   │
└───────────────┬───────────────┘           └───────────────┬───────────────┘
                │                                           │
                └─────────────────────┬─────────────────────┘
                                      ▼
                         Final Agent Output Schema:
             {intent, action, escalation_reason, drafted_reply}
```

---

## 🚀 Quickstart & Setup (Phase 0)

### 1. Prerequisites
- Python 3.10+
- Google Gemini API key ([Get a free key here](https://aistudio.google.com/app/apikey))

### 2. Environment Setup
```bash
# Clone or navigate to the repository
cd hiver-ai-agent

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets
Copy the `.env.example` template:
```bash
cp .env.example .env
```
Open `.env` and set your API key:
```env
GEMINI_API_KEY=AIzaSy...your_real_key
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run Smoke Test
Verify that your API key and model connectivity are operational:
```bash
python scripts/smoke_test.py
```

---

## 📂 Project Structure

```
hiver-ai-agent/
├── README.md                # 15-minute quickstart and evaluation instructions
├── requirements.txt         # Pinned production dependencies
├── .env.example             # Template for API credentials
├── .gitignore               # Ignores venvs, cache, and raw 3M-row data
├── decision_log.md          # Real-time architectural decisions and rationales
├── data/
│   ├── raw/                 # Ignored by git; store twcs.csv or raw downloads here
│   └── processed/           # Filtered @AmazonHelp threads & cached embeddings
├── src/
│   ├── __init__.py
│   ├── gemini_client.py     # Resilient Gemini wrapper with exponential backoff
│   ├── data_prep.py         # Thread reconstruction & resolution filtering
│   ├── intents.py           # Single source of truth intent enum
│   ├── classify.py          # Few-shot Gemini intent classifier
│   ├── retrieve.py          # FAISS index for historical resolution retrieval
│   ├── draft_reply.py       # Grounded reply generation with anti-hallucination prompts
│   ├── escalate.py          # Rule-based policy + risk detection + LLM fallback
│   └── pipeline.py          # End-to-end inference pipeline
├── eval/
│   ├── golden_set.csv       # Held-out human-curated benchmark
│   ├── golden_set_notes.md  # Sampling methodology & annotation guidelines
│   ├── baselines.py         # Trivial (majority/escalate) & Simple (regex/template)
│   ├── metrics.py           # Intent accuracy, retrieval hit rate, escalation cost-weighted F1
│   ├── llm_judge.py         # Multi-criteria Gemini evaluation rubric
│   ├── judge_agreement.py   # Human vs. LLM judge agreement metrics
│   └── run_eval.py          # Master benchmark runner producing headline comparison table
├── notebooks/
│   ├── 01_eda.ipynb         # Brand EDA & thread length analysis
│   └── 02_intent_discovery.ipynb # Unsupervised intent clustering
├── scripts/
│   └── smoke_test.py        # API verification script
└── report/
    └── report.md            # Comprehensive take-home report (max 6 pages equivalent)
```

---

## 📝 Design Principles
- **No Agent Framework Bloat**: Plain Python with explicit function calls, facilitating live whiteboarding and debugging.
- **Quota-Conscious**: Local Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings; API quota is reserved exclusively for reasoning tasks.
- **Safety First**: Asymmetric escalation weighting — false auto-handles on sensitive inquiries (lost shipments, billing disputes, angry customers) are treated as far more costly than conservative human escalations.
