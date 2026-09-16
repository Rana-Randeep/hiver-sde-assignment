# Golden Evaluation Set

This document describes the **golden evaluation set** used to evaluate the GloCare customer-support agent.

It is kept separate from the main README/report so that the evaluation dataset and its construction methodology are directly visible as **Deliverable #2**.

---

## 1. Purpose

The golden set provides a fixed, hand-labelled evaluation set for measuring whether the support agent correctly identifies customer intent and makes appropriate escalation decisions.

The assignment requires **150–250 hand-labelled examples with a short sampling/labelling note**. The final set contains **156 examples**, which is within that required range.

---

## 2. Source Data

The working dataset is:

```text
data/glocare_pairs.csv
```

It contains historical customer → GloCare response pairs from the Customer Support on Twitter dataset.

The golden set was created from the GloCare working data rather than from synthetic examples.

---

## 3. Final Golden Set

The locked evaluation set contains:

| Property | Value |
|---|---:|
| Examples | **156** |
| IDs | **G001–G156** |
| Selected examples | **156 / 156** |
| Final intents | **10** |
| Expected escalation: yes | **102** |
| Expected escalation: no | **54** |

The final set is stored as:

```text
results/golden_set.csv
```

The golden interactions are excluded from the retrieval corpus and are not used for model training or tuning after finalization.

---

## 4. Sampling Method

The goal was **not** to take a purely random sample.

Candidate examples were reviewed in batches and selected to provide coverage of different support situations.

The sampling process deliberately considered:

- common support intents,
- less frequent / rare intents,
- ambiguous cases,
- difficult intent-boundary cases,
- low-information messages,
- escalation-relevant cases,
- multi-turn or context-dependent situations,
- routine support requests.

The review sheets used sampling aids such as:

```text
review_priority
topic_signal_count
possible_escalation
short_message
low_information_signal
candidate_group
```

These fields were explicitly treated as **sampling aids only**, not as ground-truth labels.

The final intent and escalation labels were assigned during human annotation.

---

## 5. Human Labelling Process

Each reviewed example was labelled with:

```text
golden_selected
golden_intent
confidence
case_type
escalation_expected
annotation_notes
```

The review process was performed against the customer message and its historical GloCare response.

The annotator was instructed to assign **exactly one primary intent** representing the customer's main support need or required support action.

A keyword alone was not sufficient to determine the label.

---

## 6. Intent Taxonomy

The final golden set uses these 10 intents:

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

The detailed definitions and boundary rules are documented in:

```text
docs/annotation_guidelines.md
```

For example, the guidelines distinguish:

- an existing data-service problem → `data_issue`
- a plan/subscription question → `data_plan_support` during annotation guidance
- a general network/connectivity problem → `network_issue`

The final stored golden-set labels use the project's final `*_issue` taxonomy names.

---

## 7. Annotation Rules

### One primary intent

Each message receives exactly one primary intent.

### Do not classify from keywords alone

Words such as “network”, “internet”, or “data” do not automatically determine the class. The customer's actual support need is the deciding factor.

### Use `non_actionable_or_context_required` when necessary

This category is used when the available message does not contain enough actionable information to identify a meaningful support intent.

Short messages are not automatically assigned to this category.

### Explicit boundaries

The annotation guidelines contain explicit boundaries for the most easily confused categories, especially:

```text
network_issue
data_issue
data_plan_support
```

and other overlapping support categories.

---

## 8. Case-Type Coverage

The final 156 examples contain:

| Case type | Count |
|---|---:|
| Routine | **64** |
| Escalation-relevant | **54** |
| Ambiguous | **31** |
| Low-information | **5** |
| Rare | **2** |
| **Total** | **156** |

This gives the evaluation set coverage beyond only straightforward routine requests.

---

## 9. Intent Composition

| Intent | Count |
|---|---:|
| `network_issue` | 28 |
| `unauthorized_charge_or_vas` | 27 |
| `data_plan_issue` | 24 |
| `data_issue` | 18 |
| `recharge_or_airtime_issue` | 14 |
| `bonus_or_promotion_issue` | 12 |
| `sim_or_device_issue` | 12 |
| `voice_or_line_issue` | 11 |
| `non_actionable_or_context_required` | 6 |
| `account_or_general_support` | 4 |
| **Total** | **156** |

The distribution is intentionally not perfectly balanced. It reflects the selected evaluation examples while ensuring coverage of all final taxonomy classes.

---

## 10. Escalation Labels

Human annotation also recorded whether human escalation was expected.

Final distribution:

```text
Expected escalation:      102 / 156 = 65.38%
Expected non-escalation:   54 / 156 = 34.62%
```

These labels are later used to evaluate whether the implemented escalation policy is too conservative or too permissive.

---

## 11. Quality and Leakage Controls

### Golden set is locked

After finalization, the golden set is treated as an evaluation-only dataset.

It is not used for model training or tuning.

### Retrieval exclusion

Golden interactions are excluded from the retrieval corpus so that evaluation examples are not directly retrieved as historical evidence.

### Correlation limitation

The source dataset is multi-turn and contains duplicate/correlated interactions.

Known working-dataset characteristics include:

- 6,300 customer → GloCare response pairs
- 6,161 unique customer messages
- 132 messages with multiple GloCare replies
- 139 duplicate customer tweet IDs
- 201 duplicate customer texts
- no full-row duplicates

Therefore, retrieval leakage from the exact golden interaction is controlled, but broader near-duplicate/thread correlation remains a limitation.

### PII awareness

Historical customer-support data can contain phone numbers, names, locations, and other identifying details.

Public examples and reports should avoid unnecessarily exposing such information.

---

## 12. Annotation Guidelines

The complete annotation rules are maintained separately:

```text
docs/annotation_guidelines.md
```

The guidelines define:

- intent meanings,
- inclusion/exclusion rules,
- boundaries between overlapping intents,
- handling of incomplete messages,
- network vs data distinctions,
- use of `non_actionable_or_context_required`.

The supplied review sheets also make clear that sampling signals were not intended to become ground-truth labels.

---

## 13. Dataset Artifact

The final golden dataset is:

```text
results/golden_set.csv
```

Expected structure includes:

```text
golden_id
customer_tweet_id
customer_text
golden_intent
confidence
case_type
escalation_expected
annotation_notes
brand_response
golden_selected
...
```

The evaluation harness reads the selected golden examples from this artifact.

---

## 14. How It Connects to the Evaluation Harness

The golden set is the evaluation input for the automated evaluation scripts.

For example:

```powershell
python src/evaluate_end_to_end_golden.py
```

and:

```powershell
python src/evaluate_intent_classifier_final.py
```

use the finalized golden set to generate measurable evaluation results.

The full evaluation methodology is documented separately in:

```text
docs/evaluation.md
```

---

## 15. Final Position

The final golden set is a **156-example, hand-labelled, locked evaluation set** designed to cover both routine and difficult customer-support situations.

Its purpose is not to claim statistical representativeness of all GloCare traffic. Its purpose is to provide a **fixed, inspectable, diverse test set** for measuring the prototype and exposing its failure modes.

That distinction is important when interpreting the final 38.46% accuracy and 30.55% Macro F1 results.
