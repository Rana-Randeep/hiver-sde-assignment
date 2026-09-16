# Evaluation Harness

This document describes the executable evaluation harness used for the GloCare support-agent prototype.

The evaluation is intentionally separated from the main README/report narrative: the README explains the project, results, failure analysis, and decisions, while this document explains **how evaluation is executed, what is measured, and how the LLM judge was validated**.

---

## 1. Evaluation Overview

The harness evaluates four main areas:

1. **Intent classification**
2. **Historical-response retrieval**
3. **Generated reply quality**
4. **Escalation behavior**

The evaluation uses the locked 156-example golden set for final end-to-end assessment.

A separate 40-example validation subset is used to compare the LLM-as-judge with an independent human-oriented evaluation.

---

## 2. Golden Evaluation Set

Final golden set:

```text
156 examples
IDs: G001–G156
```

The golden interactions are excluded from the retrieval corpus.

The examples are hand-labelled using the project's intent taxonomy and annotation guidance.

Annotation guidance:

```text
docs/annotation_guidelines.md
```

---

## 3. Automated Evaluation

### 3.1 End-to-End Evaluation

Run:

```powershell
python src/evaluate_end_to_end_golden.py
```

This evaluates the complete pipeline:

```text
Customer message
    ↓
Intent + confidence
    ↓
Historical retrieval
    ↓
LLM reply generation
    ↓
Reply sanitization
    ↓
Escalation decision
```

The output contains the prediction and evaluation information needed for final analysis.

---

### 3.2 Intent Classifier Evaluation

Run:

```powershell
python src/evaluate_intent_classifier_final.py
```

Metrics include:

- Accuracy
- Macro F1
- Weighted F1
- Per-class F1
- Confusion matrix
- Confidence statistics

Final golden-set results:

| Metric | Result |
|---|---:|
| Accuracy | 38.46% |
| Macro F1 | 30.55% |
| Weighted F1 | 37.32% |
| Correct | 60 / 156 |
| Incorrect | 96 / 156 |

Macro F1 is treated as the primary classifier metric because the 10 intents do not perform uniformly.

---

### 3.3 Retrieval Evaluation

Run:

```powershell
python src/evaluate_vector_retrieval.py
```

The retrieval review measures human-rated relevance of retrieved historical responses.

Final vector-retrieval review:

| Metric | Result |
|---|---:|
| Mean relevance | 1.6538 / 2 |
| Useful retrieval | 91.67% |
| Clearly relevant | 73.72% |

Relevance distribution:

```text
0 = 13
1 = 28
2 = 115
```

Earlier TF-IDF retrieval baseline:

```text
Clearly relevant: 62.18%
Useful:           79.49%
Mean relevance:    1.417 / 2
```

Important: **intent alignment and human-rated retrieval relevance are different measures**.

---

## 4. LLM-as-a-Judge

### 4.1 Reply Quality Evaluation

Run:

```powershell
python src/evaluate_reply_quality.py
```

The script uses an LLM judge to score generated replies against the customer message and retrieved historical evidence.

The rubric evaluates:

| Dimension | Meaning |
|---|---|
| Correctness | Does the reply correctly address the customer's issue? |
| Groundedness | Are claims/actions supported by the available historical evidence? |
| Helpfulness | Does the reply provide a useful next step or resolution path? |
| Safety | Does it avoid unsupported, risky, or misleading claims? |
| Conciseness | Is it appropriately brief for customer support? |
| Overall | Overall quality of the response |

The judge also produces a pass/fail assessment.

### Final 156-example judge results

| Dimension | Mean |
|---|---:|
| Correctness | 2.79 / 5 |
| Groundedness | 1.28 / 5 |
| Helpfulness | 2.80 / 5 |
| Safety | 3.86 / 5 |
| Conciseness | 4.41 / 5 |
| Overall | 2.76 / 5 |
| Pass rate | 17.31% |

The particularly low groundedness score is an important signal: fluent responses were not necessarily well-supported by the retrieved evidence.

---

## 5. Judge–Human Validation

A judge should not be treated as ground truth without checking whether it agrees with human-oriented assessment.

Run:

```powershell
python src/evaluate_reply_quality_validation_40.py
```

This uses a 40-example validation subset.

The comparison produced:

| Measure | Result |
|---|---:|
| LLM judge overall | 4.85 / 5 |
| Independent human-oriented overall | ~3.23 / 5 |
| Overall exact agreement | 22.5% |
| Groundedness exact agreement | 0% |
| Cohen's kappa | Near zero |

The human-oriented evaluation was used as a **calibration check**, not as a claim of formal external human validation.

### Interpretation

The judge was substantially more positive than the independent human-oriented review.

This suggests that the LLM judge is **too lenient**, particularly around groundedness.

Therefore, the final report does not present the raw judge score as an unquestionable measure of quality. Instead, the disagreement itself is reported as an evaluation limitation and failure mode.

---

## 6. Escalation Evaluation

Escalation is evaluated as a separate safety/automation metric.

For the 156-example golden set:

```text
Expected escalation:      102 / 156  = 65.38%
Expected non-escalation:   54 / 156

Actual escalation:        156 / 156
False positives:           54 / 156  = 34.62%
False negatives:            0 / 156  = 0%
```

The result shows a conservative escalation policy: no observed false negatives, but substantial over-escalation.

---

## 7. Evaluation Artifacts

Evaluation outputs are stored under:

```text
results/
```

Important artifacts include the golden predictions, retrieval reviews/failure analysis, reply-quality judge outputs, and judge-validation outputs.

These artifacts provide the evidence behind the metrics reported in the README.

---

## 8. Baseline Evaluation

The project also evaluates two reference baselines on the same 20-example pilot validation split:

| Baseline | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 40.00% | 5.71% |
| TF-IDF + Logistic Regression | 60.00% | 35.33% |

The final 156-example golden-set results should **not** be treated as a direct apples-to-apples comparison with these pilot baselines because the evaluation sets differ.

---

## 9. Reproducibility

Required setup is documented in the root `README.md`.

The main evaluation commands are:

```powershell
python src/evaluate_end_to_end_golden.py
python src/evaluate_intent_classifier_final.py
python src/evaluate_vector_retrieval.py
python src/evaluate_reply_quality.py
python src/evaluate_reply_quality_validation_40.py
```

The evaluation scripts are designed to produce the artifacts used for the final analysis.

---

## 10. Evaluation Limitation

The evaluation harness provides measurable evidence, but it does not establish production readiness.

Important limitations include:

- small locked golden set,
- correlated source conversations,
- low classifier confidence,
- conservative escalation,
- limitations of offline historical retrieval,
- unsupported operational claims in some generated replies,
- and demonstrated over-leniency of the LLM judge.

The purpose of the harness is therefore not to manufacture a single favorable headline number. It is to make the system's **strengths, weaknesses, and evaluation uncertainty visible and reproducible**.
