
import numpy as np
from sentence_transformers import CrossEncoder

from predict import grade_answer


# ============================================================
# CONTRADICTION-AWARE GRADING EXPERIMENT
# ============================================================

print("Loading Natural Language Inference model...")

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

# Read label names from the model configuration.
id2label = nli_model.model.config.id2label

label_ids = {
    str(label).lower(): int(idx)
    for idx, label in id2label.items()
}

print("NLI labels:", id2label)


def get_contradiction_probability(reference, student):
    """Estimate whether the student answer contradicts the reference."""

    logits = nli_model.predict(
        [(reference, student)],
        apply_softmax=True
    )

    probabilities = np.asarray(logits).reshape(-1)

    contradiction_idx = next(
        (
            idx for label, idx in label_ids.items()
            if "contradiction" in label
        ),
        None
    )

    if contradiction_idx is None:
        raise ValueError(
            f"Could not identify contradiction label: {id2label}"
        )

    return float(probabilities[contradiction_idx])


def grade_answer_improved(
    domain,
    question,
    reference_answer,
    student_answer
):
    # Existing model prediction
    original = grade_answer(
        domain,
        question,
        reference_answer,
        student_answer
    )

    base_score = original["score"]

    # New correctness signal
    contradiction_probability = get_contradiction_probability(
        reference_answer,
        student_answer
    )

    # Conservative experimental penalty:
    # no penalty when contradiction probability <= 0.60;
    # penalty increases above 0.60, up to 50%.
    penalty_strength = max(
        0.0,
        (contradiction_probability - 0.60) / 0.40
    )

    penalty_strength = min(penalty_strength, 0.50)

    improved_score = base_score * (1.0 - penalty_strength)
    improved_score = float(np.clip(improved_score, 0, 10))

    return {
        "domain": original["domain"],
        "baseline_score": round(base_score, 2),
        "contradiction_probability": round(
            contradiction_probability, 4
        ),
        "penalty_applied_percent": round(
            penalty_strength * 100, 2
        ),
        "improved_score": round(improved_score, 2),
        "feedback": original["feedback"]
    }


if __name__ == "__main__":

    tests = [
        {
            "name": "Correct ML answer",
            "domain": "ml",
            "question": (
                "What is the difference between classification "
                "and regression?"
            ),
            "reference": (
                "Classification predicts discrete labels, while "
                "regression predicts continuous numerical values."
            ),
            "student": (
                "Classification assigns discrete labels to data, "
                "whereas regression predicts continuous values."
            )
        },
        {
            "name": "Contradictory ML answer",
            "domain": "ml",
            "question": (
                "What is the difference between classification "
                "and regression?"
            ),
            "reference": (
                "Classification predicts discrete labels, while "
                "regression predicts continuous numerical values."
            ),
            "student": (
                "Classification predicts continuous numerical values, "
                "while regression predicts discrete class labels."
            )
        },
        {
            "name": "Correct DE answer",
            "domain": "de",
            "question": (
                "Why can duplicate packets be a problem in a network?"
            ),
            "reference": (
                "The receiver may process duplicated packets as valid "
                "packets and perform the same action twice."
            ),
            "student": (
                "Duplicate packets may cause the receiver to process "
                "the same information twice, potentially repeating "
                "a transaction."
            )
        }
    ]

    for test in tests:
        print("\n" + "=" * 65)
        print(test["name"])
        print("=" * 65)

        result = grade_answer_improved(
            test["domain"],
            test["question"],
            test["reference"],
            test["student"]
        )

        print("Domain:", result["domain"])
        print("Baseline score:", result["baseline_score"], "/10")
        print(
            "Contradiction probability:",
            result["contradiction_probability"]
        )
        print(
            "Penalty applied:",
            result["penalty_applied_percent"], "%"
        )
        print("Experimental score:", result["improved_score"], "/10")
