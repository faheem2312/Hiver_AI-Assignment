# Hiver AI Customer Support Agent (`@AmazonHelp`)

An intelligent, retrieval-grounded customer support pipeline built for `@AmazonHelp` using Twitter Customer Support conversations. Designed for end-to-end auditability, safety-first escalation, and evaluation reproducibility in **under 15 minutes**.

---

## ⚡ Headline Performance Summary

Evaluated across the **196 held-out cases** in `eval/golden_set.csv` (strictly isolated from the 4,800 grounding resolution pairs):

| System Architecture | Intent Acc | Intent Macro-F1 | Esc. Recall | False Auto (FN) | Auto-Handle Rate | Reply Quality (1-5) | Wall-Clock Runtime |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Trivial Baseline** | 14.3% | 0.036 | 100.0% | 0 | 0.0% | 2.10 (Canned) | < 1s |
| **Simple Rule-Based** | 88.8% | 0.799 | 100.0% | 0 | 68.4% | 3.20 (Static FAQ) | < 1s |
| **AI Support Pipeline (Ours)** | **94.4%** | **0.942** | **100.0%** | **0** | **63.3%** | **4.91 / 5.0** (Grounded) | **~7.6 mins** |

- **Subsystem Metrics**: Retrieval Hit@1: **100.0%** | Retrieval MRR: **1.000** | Groundedness Score: **4.85 / 5.0** | Policy Safety Score: **4.95 / 5.0**
- **Human-Judge Alignment**: Evaluated in `eval/judge_agreement.py` — **80.0% close agreement**, **Spearman $\rho = 0.694$**, **Quadratic Weighted Kappa $\kappa = 0.864$** (Substantial Agreement).

---

## 🎯 Architecture Diagram

```
                        Incoming Customer Tweet
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   1. Intent Classification    │ (Gemini 3.1 Flash Lite + Few-Shot)
                   └───────────────┬───────────────┘
                                   │
                     Intent & Cleaned Query
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    2. Historical Retrieval    │ (FAISS / Local MiniLM Embeddings)
                   │  Top-k Brand Resolution Pairs │ (Intent-Conditioned Boosting)
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

## 🚀 15-Minute Quickstart & Headline Reproduction

Follow these steps to reproduce headline numbers from a fresh clone in **under 15 minutes**:

### 1. Prerequisites
- Python 3.10 to 3.12
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/app/apikey))

### 2. Clone & Install Dependencies
```powershell
git clone <your-repo-url>
cd hiver-ai-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets
Copy `.env.example` to `.env` and insert your Gemini API key:
```powershell
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```
*(Note: `gemini-3.1-flash-lite` provides 500 requests/day on the free tier, ensuring complete evaluation execution without rate-limit pauses).*

### 4. Verify Setup (Smoke Test)
```powershell
python scripts/smoke_test.py
```
*(Verifies official `google-genai` SDK connectivity in ~3 seconds).*

### 5. Reproduce All Headline Numbers in One Command
```powershell
python eval/run_eval.py --judge-samples 20
```
- **Expected Runtime**: **~7 to 8 minutes** on CPU.
- **Outputs**: Prints the master benchmark table comparing the pipeline against both baselines, and generates `eval/headline_comparison.md` and `eval/pipeline_predictions.csv`.

---

## 🛠️ Standalone Interactive CLI Usage

Every component is runnable standalone with zero hidden magic:

### 1. Run the Full End-to-End Pipeline on Any Tweet
```powershell
python src/pipeline.py --text "@AmazonHelp I returned my shoes 4 days ago via UPS, when will my refund show up?"
```

### 2. Test Intent Classifier Standalone
```powershell
python src/classify.py --text "@AmazonHelp where is my package? tracker says delivered but nothing arrived!"
```

### 3. Test Historical Retrieval Index Standalone
```powershell
python src/retrieve.py --query "Echo Dot keeps blinking yellow ring"
```

### 4. Test Escalation Rules Standalone
```powershell
python src/escalate.py
```

### 5. Audit Human vs. LLM Judge Agreement
```powershell
python eval/judge_agreement.py
```

### 6. Mine Top Failure Modes & Hypotheses
```powershell
python eval/failure_analysis.py
```

---

## 📂 Project Structure

```
hiver-ai-agent/
├── README.md                # 15-minute quickstart and headline reproduction guide
├── requirements.txt         # Pinned production dependencies
├── .env.example             # Template for API credentials
├── .gitignore               # Ignores secrets, caches, and large files
├── decision_log.md          # 33 chronological architectural decisions and rationales
├── data/
│   ├── raw/                 # Ignored by git; store twcs.csv here if present
│   └── processed/           # 5,000 reconstructed threads & isolated grounding corpus
├── src/
│   ├── __init__.py
│   ├── gemini_client.py     # Official google-genai wrapper with exponential backoff
│   ├── data_prep.py         # Thread reconstruction & canned resolution cleaner
│   ├── intents.py           # Definitive single-source-of-truth Intent Enum
│   ├── classify.py          # Few-shot Gemini intent classifier with structured JSON
│   ├── retrieve.py          # FAISS/NumPy historical resolution vector search
│   ├── draft_reply.py       # Grounded reply generator with strict anti-hallucination prompts
│   ├── escalate.py          # Multi-tier escalation engine (safety gates + LLM fallback)
│   └── pipeline.py          # Master end-to-end agent orchestrator
├── eval/
│   ├── golden_set.csv       # 196 held-out human-curated benchmark cases
│   ├── golden_set_notes.md  # Sampling methodology & annotation guidelines
│   ├── baselines.py         # Trivial (majority/escalate) & Simple (regex/template) baselines
│   ├── metrics.py           # Intent accuracy, retrieval hit rate, escalation safety loss
│   ├── llm_judge.py         # Multi-criteria Gemini evaluation rubric (1-5)
│   ├── judge_agreement.py   # Human vs. LLM judge alignment metrics (Cohen's kappa)
│   ├── run_eval.py          # Master benchmark runner producing headline comparison table
│   └── failure_analysis.py  # Diagnostic miner extracting top 5 failure modes
├── notebooks/
│   ├── 01_eda.ipynb         # Brand EDA & thread length analysis
│   └── 02_intent_discovery.ipynb # Unsupervised intent clustering
├── scripts/
│   ├── smoke_test.py        # API verification script
│   └── discover_intents.py  # Standalone intent discovery CLI runner
└── report/
    └── report.md            # Comprehensive formal report (All 6 mandatory sections)
```

---

## 📄 Deliverables Index

- **Formal Report**: [`report/report.md`](file:///report/report.md) — Covers Problem Framing, Results vs. Baselines, Top 5 Failure Modes with Hypotheses, *"What is misleading about my headline number?"*, and 1-Week Roadmap.
- **Decision Log**: [`decision_log.md`](file:///decision_log.md) — 33 logged non-obvious engineering decisions recorded in real-time.
- **Golden Evaluation Set**: [`eval/golden_set.csv`](file:///eval/golden_set.csv) & [`eval/golden_set_notes.md`](file:///eval/golden_set_notes.md).
- **Benchmark Comparisons**: [`eval/headline_comparison.md`](file:///eval/headline_comparison.md).
- **Diagnostic Failures**: [`eval/failure_analysis_report.md`](file:///eval/failure_analysis_report.md).
