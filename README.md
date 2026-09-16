# GloCare AI Support Agent

AI-powered customer-support agent built for the **Hiver SDE Intern Take-Home Assignment**.

The agent:
- classifies customer messages into support intents,
- retrieves relevant historical GloCare responses,
- generates a grounded reply,
- sanitizes unsupported claims,
- decides whether to auto-handle or escalate to a human.

The project is an **offline prototype** built to demonstrate a runnable support pipeline and a measurable evaluation process.

---

## Assignment Deliverables

This repository keeps the assignment deliverables clearly separated:

| Deliverable | Where it is covered |
|---|---|
| **D1 — Repo + Runnable Pipeline** | This README + `src/` |
| **D2 — Golden Evaluation Set** | `docs/golden_evaluation_set.md` + `results/golden_set.csv` |
| **D3 — Evaluation Harness** | `docs/evaluation.md` + evaluation scripts under `src/` |
| **D4 — Report** | D4 section in this README |

**D2 and D3 are intentionally documented in the `docs/` folder rather than duplicating their full content here.**

---

# D1 — Repository + Runnable Pipeline

## 1. Problem & Approach

For this project, **GloCare** was selected from the Customer Support on Twitter dataset.

The pipeline is:

```text
Customer Message
      ↓
TF-IDF + Logistic Regression
      ↓
Intent + Confidence
      ↓
Semantic Retrieval (ChromaDB)
      ↓
Top-3 Historical Examples
      ↓
Ollama Qwen3 4B
      ↓
Reply Sanitization
      ↓
Escalation Decision
      ↓
Final Support Response
```

The system is designed as an **offline historical-data prototype**. It does not have access to live customer accounts, payments, subscriptions, network systems, CRM systems, or internal support tools.

### What does "good" mean for this prototype?

A useful support system should:

1. identify the customer's support need correctly,
2. retrieve relevant historical GloCare evidence,
3. generate a reply that is grounded in that evidence,
4. avoid customer-specific or current-state claims that the prototype cannot verify,
5. provide a useful next step when direct resolution is not possible,
6. escalate cases when the system lacks sufficient confidence or capability.

For intent classification, **Macro F1** is the primary metric because the 10 intent classes are imbalanced. Accuracy, per-class metrics, and the confusion matrix are used as secondary diagnostics.

### What I chose not to build

This prototype intentionally does **not** implement:

- customer account / CRM access,
- payment or recharge processing,
- subscription management,
- live network or service-status checks,
- refunds,
- account-specific investigation,
- real operational escalation to internal GloCare systems.

Historical responses are treated as evidence of how similar cases were handled historically, **not as evidence that this prototype can perform those actions today**.

---

## 2. Dataset

Source: **Customer Support on Twitter** (Kaggle, `thoughtvector/customer-support-on-twitter`).

For this assignment, the GloCare subset contains:

- **6,300** customer → GloCare response pairs
- **6,161** unique customer messages
- **132** customer messages with multiple GloCare replies
- **139** duplicate customer tweet IDs
- **201** duplicate customer texts
- no missing values
- no full-row duplicates

The source contains noisy, multi-turn support conversations.

### Intent taxonomy

The final taxonomy contains 10 action-based intents:

```text
network_issue
data_issue
data_plan_issue
unauthorized_charge_or_vas
recharge_or_airtime_issue
sim_or_device_issue
voice_or_line_issue
bonus_or_promotion_issue
account_or_general_support
non_actionable_or_context_required
```

An intent is separated when it represents a meaningfully different support action and can be labelled consistently by a human.

---

## 3. Tech Stack

- Python
- TF-IDF + Logistic Regression
- ChromaDB
- `all-MiniLM-L6-v2` embeddings
- Ollama
- Qwen3 4B Instruct
- scikit-learn
- pandas
- joblib

Dependencies are pinned in `requirements.txt`.

---

## 4. Setup

### Create virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Ollama

Make sure Ollama is installed and the required model is available:

```powershell
ollama pull qwen3:4b-instruct
```

The prototype also requires the `all-MiniLM-L6-v2` embedding model. On first use, it may be downloaded by `sentence-transformers`.

---

## 5. Run the Agent

Start the interactive demo:

```powershell
python src/demo_agent.py
```

Example input:

```text
my internet is not working
```

The demo returns:

- predicted intent,
- confidence,
- escalation decision,
- escalation reason,
- draft reply.

Type `exit` or `quit` to stop.

---

## 6. Build / Rebuild Vector Index

The project uses a persistent ChromaDB index containing the historical GloCare responses.

To rebuild it:

```powershell
python src/build_vector_index.py
```

The index uses:

- embedding model: `all-MiniLM-L6-v2`
- collection: `glocare_historical_responses`
- top-k retrieval: 3
- duplicate retrieved customer tweet IDs are removed.

The generated `vector_db/` directory is local runtime data and is excluded from Git.

---

## 7. Reproduce Evaluation

The main end-to-end evaluation command is:

```powershell
python src/evaluate_end_to_end_golden.py
```

The intent-only evaluation is:

```powershell
python src/evaluate_intent_classifier_final.py
```

Additional evaluation scripts are available for retrieval and reply quality.

### Reproducibility note

The headline intent metrics can be reproduced from the locked golden set without retraining the full system.

The evaluation artifacts are already included under `results/`.

For a fresh environment, the main prerequisites are:

1. Python environment created,
2. `requirements.txt` installed,
3. `models/tfidf_logreg_baseline.joblib` present,
4. ChromaDB index available or rebuilt,
5. Ollama installed with `qwen3:4b-instruct` for generation-based evaluation.

The goal of the README is to make the headline evaluation path straightforward for a reviewer; the exact runtime should be verified on a clean environment before claiming a measured sub-15-minute reproduction time.

---

## 8. Repository Structure

```text
hiver-sde-assignment/
│
├── data/
│   └── glocare_pairs.csv
│
├── docs/
│   ├── annotation_guidelines.md
│   ├── golden_evaluation_set.md
│   └── evaluation.md
│
├── models/
│   └── tfidf_logreg_baseline.joblib
│
├── results/
│   └── evaluation and analysis artifacts
│
├── src/
│   ├── support_agent.py
│   ├── intent_classifier.py
│   ├── response_retriever.py
│   ├── query_vector_retrieval.py
│   ├── llm_generator.py
│   ├── reply_sanitizer.py
│   ├── escalation_policy.py
│   ├── demo_agent.py
│   └── evaluation / analysis scripts
│
├── requirements.txt
└── README.md
```

---

# D2 — Golden Evaluation Set

The complete D2 documentation is intentionally kept separate from the README.

See:

```text
docs/golden_evaluation_set.md
results/golden_set.csv
```

The golden set contains **156 locked examples (G001–G156)** and documents its sampling and labeling methodology.

---

# D3 — Evaluation Harness

The complete D3 documentation is intentionally kept separate from the README.

See:

```text
docs/evaluation.md
```

The executable evaluation harness is under:

```text
src/
```

including the golden-set, retrieval, reply-quality, and judge-validation evaluation scripts.

---

# D4 — Report

## 9. Results

### Final intent classification

The locked **156-example golden set** produced:

| Metric | Result |
|---|---:|
| Accuracy | **38.46%** |
| Macro F1 | **30.55%** |
| Weighted F1 | **37.32%** |
| Correct | **60 / 156** |

**Primary headline metric: Macro F1 = 30.55%.**

### Baselines

The baseline results below were measured on a **20-example pilot validation set**, not the locked 156-example golden set.

| Model | Accuracy | Macro F1 | Evaluation set |
|---|---:|---:|---|
| Majority-class | 40.00% | 5.71% | 20-example pilot |
| TF-IDF + Logistic Regression | 60.00% | 35.33% | 20-example pilot |
| Final classifier | 38.46% | 30.55% | 156-example golden |

These are methodological reference points rather than an apples-to-apples comparison because the baseline and final evaluation sets differ.

### Retrieval

Human review of vector retrieval produced:

- relevance 0: 13
- relevance 1: 28
- relevance 2: 115
- mean relevance: **1.6538 / 2**
- useful retrieval: **91.67%**
- clearly relevant retrieval: **73.72%**

An earlier TF-IDF retrieval baseline produced:

- mean relevance: **1.417 / 2**
- useful retrieval: **79.49%**
- clearly relevant retrieval: **62.18%**

Intent alignment should not be treated as equivalent to human retrieval relevance.

### Reply quality

The 156-example LLM-judge evaluation produced:

| Dimension | Mean |
|---|---:|
| Correctness | 2.79 / 5 |
| Groundedness | 1.28 / 5 |
| Helpfulness | 2.80 / 5 |
| Safety | 3.86 / 5 |
| Conciseness | 4.41 / 5 |
| Overall | 2.76 / 5 |

Pass rate: **17.31%**.

A separate 40-example judge-validation subset produced an LLM-judge overall mean of **4.85 / 5**, while an independent human-oriented evaluator produced approximately **3.23 / 5**.

This disagreement indicates that the LLM judge is too lenient and should not be treated as an unquestionable quality measure.

### Escalation

For the 156-example golden set:

- expected escalation: **102 / 156 = 65.38%**
- expected non-escalation: **54 / 156**
- actual escalation: **156 / 156**
- false positives: **54 / 156 = 34.62%**
- false negatives: **0**

The current policy is therefore conservative and over-escalates.

---

## 10. Failure Analysis — Top 5 Failure Modes

### 1. Intent boundary confusion

**Measured result:** 96/156 = **61.54%** of predictions were incorrect.

Strong confusion pairs include:

- `data_issue` → `network_issue`: 6
- `data_issue` → `data_plan_issue`: 5
- `network_issue` → `data_plan_issue`: 5
- `unauthorized_charge_or_vas` → `data_plan_issue`: 5

Examples include:

- **G021:** cannot browse on Glo while the same setting works on MTN → expected `data_issue`, predicted `network_issue`
- **G048:** poor data connectivity in a specific location → expected `network_issue`, predicted `data_plan_issue`
- **G079:** data-sharing reciprocity → expected `data_issue`, predicted `data_plan_issue`

**Hypothesis:** lexical overlap and missing context make closely related support intents difficult for the TF-IDF classifier to separate.

### 2. Escalation over-triggering

**Measured result:** **34.62% false positives**, with **0 false negatives**.

**Hypothesis:** the classifier's confidence is very low, so the current `0.50` threshold sends almost every case to a human.

### 3. Generic / non-actionable replies

A diagnostic detector flagged **42.31%** of replies as potentially generic/non-actionable.

This is a **diagnostic signal, not a confirmed human failure count**.

**Hypothesis:** historical support data contains many short verification-style responses, which can encourage similarly generic generated replies.

### 4. Unsupported operational/current-state claims

A diagnostic detector flagged **5.13%** of examples for possible unsupported operational/current-state claims.

Examples include historical-style statements about account or service state that the offline prototype cannot actually verify.

**Cause:** historical support agents had operational access that this prototype does not have.

**Mitigation:** the generation prompt includes a capability firewall and the output passes through a deterministic reply sanitizer.

### 5. LLM-as-a-judge over-rating

On the 40-example validation subset:

- LLM-judge overall: **4.85 / 5**
- independent human-oriented overall: **~3.23 / 5**
- exact overall agreement: **22.5%**
- groundedness exact agreement: **0%**

**Hypothesis:** the judge rewards fluent and plausible responses even when practical usefulness and grounding are weaker.

This should be treated as a calibration signal, not formal human validation.

---

## 11. What Is Misleading About My Headline Number?

The headline **30.55% Macro F1** should not be interpreted as production performance.

Important caveats:

- the locked golden set contains only 156 examples,
- the source data contains duplicate and correlated multi-turn interactions,
- broader thread/near-duplicate correlation remains possible,
- the classifier confidence is weakly discriminative,
- escalation is highly conservative,
- the LLM judge is demonstrably too lenient,
- the prototype has no live operational access,
- some unsupported current-state claims can still appear despite the capability controls.

Confidence analysis on the golden set:

| Group | Mean confidence |
|---|---:|
| Correct | 0.1391 |
| Incorrect | 0.1300 |
| Gap | 0.0091 |

This small gap shows that confidence alone is not a reliable correctness signal.

The headline therefore summarizes a **small, controlled offline evaluation**, not real-world customer-support performance.

### Bottom line

> The project demonstrates a working and measurable prototype with clear strengths, weaknesses, and safety controls. It is not production-ready customer-support automation.

---

## 12. What I Would Do With One More Week

1. Improve the `data_issue` / `network_issue` / `data_plan_issue` boundaries.
2. Use conversation/thread-level train-validation splits.
3. Calibrate the escalation threshold on a separate validation set.
4. Compare TF-IDF, semantic, and hybrid retrieval more systematically.
5. Collect stronger human ratings for reply quality.
6. Recalibrate or replace the LLM judge.
7. Add more structured evidence and stronger generation constraints.
8. Add production-style monitoring and human override metrics.

---

## 13. Decision Log

1. **Selected GloCare** because its data provided enough diverse support interactions for the assignment.
2. **Defined 10 action-based intents** instead of relying on keyword categories.
3. **Used clustering for discovery only** rather than treating unsupervised clusters as ground truth.
4. **Kept meaningful short messages** because short does not automatically mean unusable.
5. **Preferred thread-level splitting** because the source contains multi-turn conversations.
6. **Used Macro F1 as the primary metric** because of class imbalance.
7. **Kept escalation explicit** because safe handling requires a clear human-handoff policy.
8. **Grounded generation in historical responses** so replies reflect the selected brand's historical support behaviour.
9. **Used semantic retrieval** to improve retrieval beyond lexical similarity.
10. **Deduplicated retrieved tweet IDs** to reduce repeated evidence from the same historical interaction.
11. **Excluded golden interactions from retrieval** to reduce direct evaluation contamination.
12. **Added a deterministic capability firewall** because historical support actions do not imply current system access.
13. **Sanitized PII and historical artifacts** before returning generated responses.
14. **Validated the LLM judge** instead of assuming its ratings were reliable.
15. **Reported failure modes and limitations honestly** rather than presenting only favourable metrics.

---

## Final Takeaway

The main result is not that the classifier is highly accurate.

The main result is that the project contains a **runnable end-to-end support agent and an evaluation process that exposes where it fails**:

```text
Classify → Retrieve → Generate → Sanitize → Escalate → Evaluate
```

This provides a concrete basis for deciding what would need to improve before trusting the system in production.
