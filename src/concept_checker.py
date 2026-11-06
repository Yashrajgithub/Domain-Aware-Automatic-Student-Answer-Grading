
import re
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

print("Loading NLI model...")
nli_model = CrossEncoder(MODEL_NAME)

id2label = nli_model.model.config.id2label

LABELS = {
    str(label).lower(): int(idx)
    for idx, label in id2label.items()
}

ENTAILMENT_ID = next(
    idx for label, idx in LABELS.items()
    if "entailment" in label
)

CONTRADICTION_ID = next(
    idx for label, idx in LABELS.items()
    if "contradiction" in label
)

NEUTRAL_ID = next(
    idx for label, idx in LABELS.items()
    if "neutral" in label
)


def check_concepts(student_answer, required_claims):
    sentences = [
        s.strip()
        for s in re.split(r"[.!?;\n]+", student_answer)
        if s.strip()
    ]

    supported = []
    contradicted = []
    unresolved = []

    if not sentences:
        return {
            "supported": [],
            "contradicted": [],
            "unresolved": list(required_claims),
            "coverage_percent": 0.0,
        }

    threshold = 0.70

    for claim in required_claims:
        # Compare each student sentence with this individual claim.
        pairs = [(sentence, claim) for sentence in sentences]

        probabilities = nli_model.predict(
            pairs,
            apply_softmax=True
        )

        for sentence, row in zip(sentences, probabilities):
            print(f"\nSentence: {sentence}")
            print(f"Claim: {claim}")
            print(f"Entailment: {float(row[ENTAILMENT_ID]):.4f}")
            print(f"Contradiction: {float(row[CONTRADICTION_ID]):.4f}")
            print(f"Neutral: {float(row[NEUTRAL_ID]):.4f}")

        entailment_scores = [
            float(row[ENTAILMENT_ID]) for row in probabilities
        ]

        contradiction_scores = [
            float(row[CONTRADICTION_ID]) for row in probabilities
        ]

        best_entailment_idx = max(
            range(len(sentences)),
            key=lambda i: entailment_scores[i]
        )

        best_contradiction_idx = max(
            range(len(sentences)),
            key=lambda i: contradiction_scores[i]
        )

        best_entailment = entailment_scores[best_entailment_idx]
        best_contradiction = contradiction_scores[best_contradiction_idx]

        # Updated decision logic
        if best_entailment >= threshold:
            supported.append(claim)

        elif best_contradiction >= threshold:
            contradiction_sentence = sentences[
                best_contradiction_idx
            ].lower()

            claim_lower = claim.lower()

            # Extract the subject of the required claim.
            # Example: "Regression predicts continuous numerical values."
            # Subject: "Regression"
            subject = claim_lower.split()[0].strip(".,:;")

            # Require the subject to be mentioned explicitly.
            if re.search(
                r"\b" + re.escape(subject) + r"\b",
                contradiction_sentence
            ):
                contradicted.append(claim)
            else:
                unresolved.append(claim)

        else:
            unresolved.append(claim)

    # Calculate diagnostic coverage based on supported claims.
    coverage = (
        len(supported) / len(required_claims) * 100
        if required_claims else 100.0
    )

    return {
        "supported": supported,
        "contradicted": contradicted,
        "unresolved": unresolved,
        "coverage_percent": coverage,
    }


if __name__ == "__main__":

    test_cases = [
        {
            "name": "Correct ML answer",
            "answer": (
                "Classification assigns discrete labels to data, "
                "whereas regression predicts continuous numerical values."
            ),
            "claims": [
                "Classification predicts discrete labels.",
                "Regression predicts continuous numerical values.",
            ],
        },
        {
            "name": "Contradictory ML answer",
            "answer": (
                "Classification predicts continuous numerical values, "
                "while regression predicts discrete class labels."
            ),
            "claims": [
                "Classification predicts discrete labels.",
                "Regression predicts continuous numerical values.",
            ],
        },
        {
            "name": "Partial ML answer",
            "answer": "Classification predicts discrete labels.",
            "claims": [
                "Classification predicts discrete labels.",
                "Regression predicts continuous numerical values.",
            ],
        },
        {
            "name": "Correct general answer",
            "answer": (
                "Ice floats because it is less dense than liquid water."
            ),
            "claims": [
                "Ice is less dense than liquid water.",
            ],
        },
        {
            "name": "Contradictory general answer",
            "answer": (
                "Ice sinks because it is denser than liquid water."
            ),
            "claims": [
                "Ice is less dense than liquid water.",
            ],
        },
    ]

    for case in test_cases:
        result = check_concepts(
            case["answer"],
            case["claims"]
        )

        print("\n" + "=" * 60)
        print(case["name"])
        print("=" * 60)

        print("Supported:", result["supported"])
        print("Contradicted:", result["contradicted"])
        print("Unresolved:", result["unresolved"])

        print(
            "Diagnostic coverage:",
            result["coverage_percent"],
            "%"
        )
