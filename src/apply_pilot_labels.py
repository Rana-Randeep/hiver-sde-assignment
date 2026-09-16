import pandas as pd

ORIGINAL_PATH = "results/pilot_candidates_for_labeling.csv"
LABELLED_PATH = "results/pilot_candidates_labeled_batch1_100.csv"

print("=" * 70)
print("APPLYING PILOT LABELS")
print("=" * 70)

original = pd.read_csv(ORIGINAL_PATH)
labelled = pd.read_csv(LABELLED_PATH)

print(f"Original rows: {len(original)}")
print(f"Labelled rows: {len(labelled)}")

# Safety checks
assert len(original) == 100, "Original pilot CSV should contain 100 rows."
assert len(labelled) == 100, "Labelled pilot CSV should contain 100 rows."

required_columns = [
    "pilot_id",
    "pilot_intent",
    "label_confidence",
    "label_notes",
]

for column in required_columns:
    assert column in labelled.columns, (
        f"Missing required column in labelled file: {column}"
    )

assert original["pilot_id"].is_unique, "Original pilot_id values are not unique."
assert labelled["pilot_id"].is_unique, "Labelled pilot_id values are not unique."

assert set(original["pilot_id"]) == set(labelled["pilot_id"]), (
    "Pilot IDs do not match between original and labelled files."
)

# Keep only the label information we need.
labels = labelled[
    [
        "pilot_id",
        "pilot_intent",
        "label_confidence",
        "label_notes",
    ]
].copy()

# Remove old label columns if they already exist.
for column in [
    "pilot_intent",
    "label_confidence",
    "label_notes",
]:
    if column in original.columns:
        original = original.drop(columns=[column])

# Merge labels onto the original pilot dataset.
merged = original.merge(
    labels,
    on="pilot_id",
    how="left",
    validate="one_to_one",
)

# Final safety checks.
assert len(merged) == 100, "Merge changed the number of rows."

assert merged["pilot_intent"].notna().all(), (
    "Some pilot_intent values are missing after merge."
)

assert merged["label_confidence"].notna().all(), (
    "Some label_confidence values are missing after merge."
)

assert merged["label_notes"].notna().all(), (
    "Some label_notes values are missing after merge."
)

# Preserve original row order.
merged = merged.sort_values("pilot_id").reset_index(drop=True)

merged.to_csv(
    ORIGINAL_PATH,
    index=False,
)

print()
print("=" * 70)
print("LABEL APPLICATION SUMMARY")
print("=" * 70)

print()
print("Intent distribution:")
print(merged["pilot_intent"].value_counts())

print()
print("Confidence distribution:")
print(merged["label_confidence"].value_counts())

print()
print(f"Saved updated file: {ORIGINAL_PATH}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)