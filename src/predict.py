import joblib
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

print("Loading Sentence Transformer...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# LOAD DOMAIN MODELS
# ============================================================

MODELS = {}
TFIDF_MODELS = {}

for domain in ["general", "ml", "de"]:

    model_path = MODEL_DIR / f"{domain}_model.pkl"
    tfidf_path = MODEL_DIR / f"{domain}_tfidf.pkl"

    MODELS[domain] = joblib.load(model_path)
    TFIDF_MODELS[domain] = joblib.load(tfidf_path)


# ============================================================
# DOMAIN NAMES
# ============================================================

DOMAIN_MAP = {
    "general": "General / Theory",
    "ml": "Machine Learning",
    "de": "Data Engineering"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(
    question,
    reference_answer,
    student_answer,
    tfidf
):
    """
    Generate the same 7 features used during training.
    """

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    reference_embedding = embedding_model.encode(
        [reference_answer]
    )

    student_embedding = embedding_model.encode(
        [student_answer]
    )

    semantic_similarity = cosine_similarity(
        reference_embedding,
        student_embedding
    )[0][0]

    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    reference_tfidf = tfidf.transform(
        [reference_answer]
    )

    student_tfidf = tfidf.transform(
        [student_answer]
    )

    tfidf_similarity = cosine_similarity(
        reference_tfidf,
        student_tfidf
    )[0][0]

    # --------------------------------------------------------
    # Lexical features
    # --------------------------------------------------------

    ref_words = set(
        reference_answer.lower().split()
    )

    student_words = set(
        student_answer.lower().split()
    )

    reference_length = len(
        reference_answer.split()
    )

    student_length = len(
        student_answer.split()
    )

    common_words = len(
        ref_words.intersection(student_words)
    )

    if len(ref_words) > 0:

        word_overlap = (
            common_words / len(ref_words)
        )

    else:

        word_overlap = 0

    if reference_length > 0:

        length_ratio = (
            student_length / reference_length
        )

    else:

        length_ratio = 0

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = np.array([[
        semantic_similarity,
        tfidf_similarity,
        reference_length,
        student_length,
        length_ratio,
        common_words,
        word_overlap
    ]])

    return features


# ============================================================
# GRADE FUNCTION
# ============================================================

def grade_answer(
    domain,
    question,
    reference_answer,
    student_answer
):

    domain = domain.lower().strip()

    if domain not in MODELS:

        raise ValueError(
            "Invalid domain. Choose: general, ml or de."
        )

    # --------------------------------------------------------
    # Get appropriate domain model
    # --------------------------------------------------------

    model = MODELS[domain]

    tfidf = TFIDF_MODELS[domain]

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    features = calculate_features(
        question,
        reference_answer,
        student_answer,
        tfidf
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predicted_score = model.predict(
        features
    )[0]

    # Keep score between 0 and 1

    predicted_score = np.clip(
        predicted_score,
        0,
        1
    )

    # Convert to 10-point scale

    score_out_of_10 = predicted_score * 10

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    if score_out_of_10 >= 8:

        feedback = "Excellent answer. The response is highly similar to the reference answer."

    elif score_out_of_10 >= 6:

        feedback = "Good answer. The response covers a substantial part of the expected content."

    elif score_out_of_10 >= 4:

        feedback = "Partially correct. Some relevant concepts are present, but important content may be missing."

    elif score_out_of_10 >= 2:

        feedback = "Limited answer. The response has some relevance but misses important concepts."

    else:

        feedback = "Poor answer. The response has very little similarity to the expected answer."

    return {
        "domain": DOMAIN_MAP[domain],
        "score": round(float(score_out_of_10), 2),
        "normalized_score": round(float(predicted_score), 4),
        "feedback": feedback
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("AUTOMATIC ANSWER GRADER")
    print("=" * 70)

    domain = "ml"

    question = (
        "Explain classification and regression; "
        "what is the difference?"
    )

    reference_answer = (
        "Classification is a task of mapping data to discrete "
        "labels while regression maps data to a continuous "
        "function or real values."
    )

    student_answer = (
        "Classification assigns discrete labels to data, "
        "while regression predicts continuous values."
    )

    result = grade_answer(
        domain,
        question,
        reference_answer,
        student_answer
    )

    print("\nDomain:")
    print(result["domain"])

    print("\nScore:")
    print(f"{result['score']} / 10")

    print("\nNormalized Score:")
    print(result["normalized_score"])

    print("\nFeedback:")
    print(result["feedback"])

    print("\n" + "=" * 70)