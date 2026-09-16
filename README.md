# GloCare AI Support Agent

AI-powered customer-support agent built for the **Hiver SDE Intern Take-Home Assignment**.

The agent:
- classifies customer messages into support intents,
- retrieves relevant historical GloCare responses,
- generates a grounded reply,
- sanitizes unsupported claims and historical artifacts,
- decides whether to auto-handle or escalate to a human.

The project is evaluated as an **offline prototype**, with emphasis on measurable performance, failure analysis, and safe system boundaries.

---

## 1. Problem Framing — What Does “Good” Mean?

For this project, **GloCare** was selected from the Customer Support on Twitter dataset because it provides diverse, real-world support interactions across network, data, plans, recharge, VAS, SIM/line, and bonus-related issues.

### What “Good” Means

A good support agent should:

1. **Identify the customer's actual need**
   - Predict the correct support intent.
   - Primary classifier metric: **Macro F1**, with accuracy and per-class metrics as supporting measures.

2. **Produce a grounded and useful reply**
   - Address the customer's actual problem.
   - Use relevant historical GloCare responses as evidence.
   - Avoid copying customer-specific information from historical examples.
   - Avoid unsupported claims, policies, outcomes, or timelines.
   - Provide a useful next step when direct resolution is not possible.

3. **Escalate appropriately**
   - Sensitive or account-specific issues should be handled conservatively.
   - Low-confidence or insufficiently supported cases should be routed to a human.
   - The goal is not simply maximum automation: both unsafe false negatives and unnecessary false positives matter.

### What I Chose Not to Build

This is intentionally an **offline historical-data prototype**. It does not provide:

- customer account / CRM access,
- payment or recharge processing,
- subscription management,
- live network or service-status access,
- internal GloCare support tools,
- real-time verification of customer state.

Therefore, the system must not claim that it checked an account, verified a transaction, restored service, cancelled a subscription, or performed another unavailable operational action.

---

## 2. Approach

The end-to-end pipeline is:

```text
Customer Message
      |
      v
TF-IDF + Logistic Regression
      |
      v
Intent + Confidence
      |
      v
Persistent ChromaDB Semantic Retrieval
(top-k = 3)
      |
      v
Ollama Qwen3 4B
Grounded Reply Generation
      |
      v
Deterministic Reply Sanitizer
      |
      v
Escalation Policy
      |
      v
Final Support Response
```

### Components

**Intent classification**
- TF-IDF + Logistic Regression
- Model: `models/tfidf_logreg_baseline.joblib`

**Historical retrieval**
- Persistent ChromaDB
- Collection: `glocare_historical_responses`
- Embeddings: `all-MiniLM-L6-v2`
- Top-k: 3
- Retrieved examples are deduplicated by historical customer tweet ID.

**Reply generation**
- Local Ollama
- Model: `qwen3:4b-instruct`
- Generation is constrained by a capability firewall so historical support actions are not treated as current system capabilities.

**Sanitization**
- Removes Twitter usernames, URLs, signatures, and other historical artifacts.
- Rewrites unsupported operational claims where detected.
- Adds a deterministic safety layer after generation.

**Escalation**
- Current confidence threshold: `0.50`
- Sensitive/account-specific cases and insufficiently supported cases can be escalated.

---

## 3. Dataset

Source: **Customer Support on Twitter** (`thoughtvector/customer-support-on-twitter`).

Working GloCare dataset:

`data/glocare_pairs.csv`

- **6,300** customer → GloCare response pairs
- **6,161** unique customer messages
- **132** customer messages have multiple GloCare replies
- **139** duplicate customer tweet IDs
- **201** duplicate customer texts
- No missing values
- No full-row duplicates

The source is real, noisy, and multi-turn. Duplicate and correlated interactions remain an evaluation limitation.

### Final Intent Taxonomy

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

The taxonomy is action-oriented: separate intents should correspond to meaningfully different support actions and be consistently labelable.

---

## 4. Golden Evaluation Set

A locked **156-example golden set** was created for final evaluation.

- IDs: `G001`–`G156`
- Hand-labelled using the final intent definitions.
- Golden interactions are excluded from the retrieval corpus.
- The locked golden set is not used for post-finalization tuning.

Annotation guidance:

```text
docs/annotation_guidelines.md
```

The broader source dataset can still contain correlated or near-duplicate conversations, which is discussed in the limitations and headline-number analysis.

---

## 5. Tech Stack

- Python
- scikit-learn
- pandas
- TF-IDF
- Logistic Regression
- ChromaDB
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Ollama
- Qwen3 4B Instruct

Dependencies are pinned in:

```text
requirements.txt
```

---

## 6. Setup

### Windows PowerShell

Create and activate the virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### Ollama prerequisite

Make sure Ollama is installed and the required model is available:

```powershell
ollama pull qwen3:4b-instruct
```

The embedding model `all-MiniLM-L6-v2` is loaded locally by the retrieval code.

---

## 7. Run the Agent

Start the interactive demo:

```powershell
python src/demo_agent.py
```

Example input:

```text
my internet is not working
```

The agent returns:

- predicted intent,
- confidence,
- escalation decision,
- escalation reason,
- draft customer-facing reply.

Type `exit` or `quit` to stop.

---

## 8. Build / Rebuild the Vector Index

The project uses a persistent ChromaDB index.

To rebuild it:

```powershell
python src/build_vector_index.py
```

The index stores historical GloCare responses using `all-MiniLM-L6-v2` embeddings.

---

## 9. Reproduce Evaluation

### End-to-end golden evaluation

```powershell
python src/evaluate_end_to_end_golden.py
```

### Intent evaluation

```powershell
python src/evaluate_intent_classifier_final.py
```

### Vector retrieval evaluation

```powershell
python src/evaluate_vector_retrieval.py
```

### Reply-quality evaluation

```powershell
python src/evaluate_reply_quality.py
```

### Judge validation

```powershell
python src/evaluate_reply_quality_validation_40.py
```

Evaluation artifacts are stored under:

```text
results/
```

---

# 10. Results

## Intent Classification

Locked 156-example golden set:

| Metric | Result |
|---|---:|
| Accuracy | **38.46%** |
| Macro F1 | **30.55%** |
| Weighted F1 | **37.32%** |
| Correct | **60 / 156** |
| Incorrect | **96 / 156** |

**Macro F1 is the primary metric** because performance varies substantially across the 10 intents.

## Baselines

The two baseline results below were measured on the **same 20-example pilot validation set**, not the locked 156-example golden set.

| Model | Accuracy | Macro F1 | Evaluation Set |
|---|---:|---:|---|
| Majority-class baseline | **40.00%** | **5.71%** | 20-example pilot |
| TF-IDF + Logistic Regression | **60.00%** | **35.33%** | 20-example pilot |
| Final classifier | **38.46%** | **30.55%** | 156-example golden |

The baseline and final results are therefore **not apples-to-apples comparisons**. The baselines provide methodological reference points rather than a direct leaderboard comparison.

## Retrieval

### Semantic / Vector Retrieval

Human-reviewed retrieval results:

| Metric | Result |
|---|---:|
| Mean relevance | **1.6538 / 2** |
| Useful retrieval | **91.67%** |
| Clearly relevant | **73.72%** |

Relevance counts:

- 0: 13
- 1: 28
- 2: 115

Earlier TF-IDF retrieval baseline:

- Clearly relevant: **62.18%**
- Useful: **79.49%**
- Mean relevance: **1.417 / 2**

An important distinction is that **intent alignment is not the same as human retrieval relevance**.

## Reply Quality

156-row LLM-judge evaluation:

| Dimension | Mean |
|---|---:|
| Correctness | **2.79 / 5** |
| Groundedness | **1.28 / 5** |
| Helpfulness | **2.80 / 5** |
| Safety | **3.86 / 5** |
| Conciseness | **4.41 / 5** |
| Overall | **2.76 / 5** |
| Pass rate | **17.31%** |

These results exposed weaknesses in groundedness and practical usefulness despite strong conciseness.

## Escalation

On the 156-example golden set:

- Expected human escalation: **102 / 156 (65.38%)**
- Expected non-escalation: **54 / 156**
- Actual escalation: **156 / 156**
- False positives: **54 / 156 (34.62%)**
- False negatives: **0**

This is a **very conservative** policy: safer with respect to missed escalations, but operationally inefficient.

---

# 11. Failure Analysis — Top 5 Failure Modes

## 1. Intent Boundary Confusion

**Measured result:** 96/156 predictions were incorrect (**61.54%**).

Strong confusion pairs included:

- `data_issue` → `network_issue`: 6
- `data_issue` → `data_plan_issue`: 5
- `network_issue` → `data_plan_issue`: 5
- `unauthorized_charge_or_vas` → `data_plan_issue`: 5

Examples:

- **G021:** “cannot browse on Glo, same setting works on MTN” → gold `data_issue`, predicted `network_issue`
- **G048:** “poor data connectivity in specific location” → gold `network_issue`, predicted `data_plan_issue`

**Hypothesis:** TF-IDF depends heavily on lexical overlap, while short support messages often require context to distinguish closely related intents.

---

## 2. Escalation Over-Triggering

**Measured result:** 34.62% false positives and 0 false negatives.

**Hypothesis:** classifier confidence is very low, so the current `0.50` threshold causes almost every example to escalate.

**Implication:** escalation should be calibrated on separate validation data rather than the locked golden set.

---

## 3. Generic / Non-actionable Replies

A diagnostic check flagged **66/156 (42.31%)** replies as potentially generic or non-actionable.

This is a **diagnostic signal, not a confirmed human failure count**.

**Hypothesis:** historical support data contains many short verification-style replies, which can encourage similarly generic generated responses.

---

## 4. Unsupported Operational / Current-State Claims

A diagnostic check flagged **8/156 (5.13%)** examples for possible unsupported operational or current-state claims.

Examples included claims such as:

- “4G is active on the sim card.”
- location-specific coverage claims,
- “The issue is being resolved.”
- “Work is ongoing to resolve the data browsing issue.”

**Hypothesis:** historical support agents had operational access that this offline prototype does not have.

**Mitigation:** capability firewall + deterministic reply sanitizer.

---

## 5. LLM-as-a-Judge Over-rating

On the 40-example validation subset:

- LLM judge overall: **4.85 / 5**
- Independent human-oriented review: approximately **3.23 / 5**
- Overall exact agreement: **22.5%**
- Groundedness exact agreement: **0%**

**Hypothesis:** the judge rewards fluent and plausible replies even when grounding or practical usefulness is weaker.

This is a **calibration signal**, not formal external human validation.

---

# 12. What Is Misleading About My Headline Number?

The headline result — **30.55% Macro F1** — should not be interpreted as production performance.

### 1. The golden set is small

The final evaluation contains only **156 examples**, so the result has meaningful sampling uncertainty.

### 2. Accuracy hides class-level weaknesses

Accuracy is **38.46%**, but Macro F1 is **30.55%**. Macro F1 gives each intent equal weight and therefore exposes weak performance on individual classes.

### 3. The source data is correlated

The source contains duplicate customer texts, repeated tweet IDs, multiple replies, and multi-turn conversations. Golden interactions were excluded from retrieval, but broader correlation remains.

### 4. Confidence is poorly calibrated

Mean confidence was:

- Correct: **0.1391**
- Incorrect: **0.1300**
- Gap: only **0.0091**

So confidence alone is not a strong correctness signal.

### 5. Escalation is conservative

The system had **0 observed false negatives**, but **34.62% false positives**. The safety-oriented policy therefore trades automation efficiency for caution.

### 6. The LLM judge is too lenient

The 40-example judge validation averaged **4.85/5**, while independent human-oriented review was approximately **3.23/5**. This makes the raw judge score misleading without calibration evidence.

### 7. Offline evaluation is not live-support validation

The prototype has no live access to accounts, payments, subscriptions, network state, or internal support systems. A strong offline reply does not prove that the system can safely perform real operational actions.

### Bottom line

> On a 156-example golden set, the intent classifier achieved 30.55% macro F1 and 38.46% accuracy. The system showed conservative escalation with 0 observed false negatives but 34.62% false positives, while reply evaluation exposed an over-lenient LLM judge and occasional unsupported operational claims. These results demonstrate a working, measurable prototype rather than production-ready support automation.

---

# 13. What I Would Do With One More Week

1. Improve the `data_issue` / `network_issue` / `data_plan_issue` boundaries.
2. Use conversation/thread-level train-validation splits.
3. Calibrate the escalation threshold on a separate validation set.
4. Compare TF-IDF, semantic, and hybrid retrieval more systematically.
5. Collect stronger human ratings for reply quality.
6. Recalibrate or replace the LLM judge.
7. Add more structured evidence and response constraints.
8. Add production-style monitoring and human-override metrics.

---

# 14. Decision Log

1. **Selected GloCare** because its data provided diverse and manageable support interactions.
2. **Defined 10 action-based intents** instead of relying only on keyword categories.
3. **Used a hand-labelled golden set** because the assignment requires measurable proof.
4. **Used Macro F1 as the primary classifier metric** because class-level performance matters under imbalance.
5. **Included a majority-class baseline** to establish a trivial reference point.
6. **Included TF-IDF + Logistic Regression** as a simple ML baseline.
7. **Added semantic retrieval** to find historically similar support resolutions.
8. **Used top-3 retrieved examples** as grounding evidence for generation.
9. **Used a local Ollama model** so the prototype does not depend on a paid LLM API.
10. **Added a capability firewall** because historical actions do not imply current system access.
11. **Added deterministic sanitization** as a second layer against unsupported claims and historical artifacts.
12. **Made escalation conservative** for low-confidence and sensitive cases.
13. **Validated the LLM judge against human-oriented review** instead of assuming judge scores were reliable.
14. **Excluded golden interactions from retrieval** to reduce direct evaluation leakage.
15. **Reported failure modes and limitations** rather than presenting only favorable metrics.

---

## 15. Repository Structure

```text
hiver-sde-assignment/
├── data/
│   └── glocare_pairs.csv
│
├── docs/
│   └── annotation_guidelines.md
│
├── models/
│   └── tfidf_logreg_baseline.joblib
│
├── vector_db/
│   └── chroma/
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
│   └── evaluation scripts
│
├── requirements.txt
└── README.md
```

---

## Final Takeaway

The main result is not that the classifier is highly accurate.

The main result is that the project has a **runnable end-to-end support agent and an evaluation process that exposes where it fails**:

```text
Classify → Retrieve → Generate → Sanitize → Escalate → Evaluate
```

The prototype demonstrates measurable engineering work, explicit safety boundaries, and honest failure analysis. It should **not** be presented as production-ready customer-support automation.
