from pathlib import Path

import joblib


MODEL_FILE = Path("models/tfidf_logreg_baseline.joblib")


model = joblib.load(MODEL_FILE)


test_message = "my internet is not working"

probabilities = model.predict_proba([test_message])[0]
classes = model.classes_

results = sorted(
    zip(classes, probabilities),
    key=lambda x: x[1],
    reverse=True,
)

print("Customer message:", test_message)
print()
print("Intent probabilities:")
print("-" * 45)

for intent, probability in results:
    print(f"{intent:<35} {probability:.4f}")