import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "dataset/train_clean.parquet"
TEST_PATH = "dataset/validation_clean.parquet"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ASAG2024 - TF-IDF BASELINE MODEL")
print("=" * 70)

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

print("\nTrain samples:", len(train))
print("Validation samples :", len(test))

# ============================================================
# COMBINE REFERENCE + STUDENT ANSWERS FOR TF-IDF VOCABULARY
# ============================================================

reference_answers = train["reference_answer"].tolist()
student_answers = train["provided_answer"].tolist()

all_text = reference_answers + student_answers


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

vectorizer.fit(all_text)

print("Vocabulary size:", len(vectorizer.vocabulary_))


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

print("\nCalculating cosine similarity...")

reference_vectors = vectorizer.transform(
    test["reference_answer"]
)

student_vectors = vectorizer.transform(
    test["provided_answer"]
)

similarities = cosine_similarity(
    reference_vectors,
    student_vectors
).diagonal()


# ============================================================
# PREDICT SCORE
# ============================================================

# TF-IDF similarity is already between 0 and 1.
predicted_scores = similarities

actual_scores = test["normalized_grade"].values


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    actual_scores,
    predicted_scores
)

rmse = np.sqrt(
    mean_squared_error(
        actual_scores,
        predicted_scores
    )
)

r2 = r2_score(
    actual_scores,
    predicted_scores
)


print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(10, len(test))):

    print("\n------------------------------------------------------------")

    print("Question:")
    print(test.iloc[i]["question"][:300])

    print("\nReference:")
    print(test.iloc[i]["reference_answer"][:300])

    print("\nStudent:")
    print(test.iloc[i]["provided_answer"][:300])

    print(f"\nActual score     : {actual_scores[i]:.3f}")
    print(f"Similarity score : {predicted_scores[i]:.3f}")
    print(f"Predicted /10    : {predicted_scores[i] * 10:.2f}")


print("\n" + "=" * 70)
print("BASELINE COMPLETE")
print("=" * 70)