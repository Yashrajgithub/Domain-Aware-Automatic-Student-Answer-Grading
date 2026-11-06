import argparse
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "ASAG-Dataset-master" / "asag_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MACHINE LEARNING DOMAIN MODEL")
print("=" * 70)

print("\nLoading DigiKlausur dataset...")

df = pd.read_csv(DATA_FILE)

print("Original dataset shape:", df.shape)


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    df["student_answer"].notna() &
    df["ref_answer"].notna()
].copy()

df["student_answer"] = df["student_answer"].astype(str)
df["ref_answer"] = df["ref_answer"].astype(str)
df["question"] = df["question"].astype(str)

print("After removing missing answers:", df.shape)

print("\nGrade distribution:")
print(df["grades_round"].value_counts().sort_index())


# ============================================================
# TARGET
# ============================================================

# DigiKlausur grades are 0, 1 and 2.
# Convert them to 0-1 scale.

df["normalized_grade"] = df["grades_round"] / 2.0

print("\nNormalized grade range:")
print(df["normalized_grade"].min(),
      "to",
      df["normalized_grade"].max())


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

# Use question_id to avoid putting answers from the same
# question into both train and validation sets.

question_ids = sorted(df["question_id"].unique())

rng = np.random.RandomState(42)
rng.shuffle(question_ids)

split_index = int(len(question_ids) * 0.8)

train_questions = question_ids[:split_index]
validation_questions = question_ids[split_index:]

train_df = df[df["question_id"].isin(train_questions)].copy()
val_df = df[df["question_id"].isin(validation_questions)].copy()

print("\nTrain size:", len(train_df))
print("Validation size:", len(val_df))

print("\nTrain questions:", len(train_questions))
print("Validation questions:", len(validation_questions))


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# TEXT DATA
# ============================================================

train_ref = train_df["ref_answer"].tolist()
train_student = train_df["student_answer"].tolist()

val_ref = val_df["ref_answer"].tolist()
val_student = val_df["student_answer"].tolist()


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

print("\nCalculating semantic similarity...")

train_ref_emb = embedding_model.encode(
    train_ref,
    show_progress_bar=True
)

train_student_emb = embedding_model.encode(
    train_student,
    show_progress_bar=True
)

val_ref_emb = embedding_model.encode(
    val_ref,
    show_progress_bar=True
)

val_student_emb = embedding_model.encode(
    val_student,
    show_progress_bar=True
)


train_semantic = np.diag(
    cosine_similarity(train_ref_emb, train_student_emb)
)

val_semantic = np.diag(
    cosine_similarity(val_ref_emb, val_student_emb)
)


# ============================================================
# TF-IDF
# ============================================================

print("\nCalculating TF-IDF similarity...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

combined_train = train_ref + train_student

tfidf.fit(combined_train)

train_ref_tfidf = tfidf.transform(train_ref)
train_student_tfidf = tfidf.transform(train_student)

val_ref_tfidf = tfidf.transform(val_ref)
val_student_tfidf = tfidf.transform(val_student)

train_tfidf_similarity = np.diag(
    cosine_similarity(train_ref_tfidf, train_student_tfidf)
)

val_tfidf_similarity = np.diag(
    cosine_similarity(val_ref_tfidf, val_student_tfidf)
)


# ============================================================
# LEXICAL FEATURES
# ============================================================

print("\nCalculating lexical features...")


def lexical_features(reference_answers, student_answers):

    features = []

    for ref, student in zip(reference_answers, student_answers):

        ref_words = set(ref.lower().split())
        student_words = set(student.lower().split())

        ref_len = len(ref.split())
        student_len = len(student.split())

        common_words = len(ref_words.intersection(student_words))

        if len(ref_words) > 0:
            word_overlap = (
                common_words / len(ref_words)
            )
        else:
            word_overlap = 0

        if ref_len > 0:
            length_ratio = student_len / ref_len
        else:
            length_ratio = 0

        features.append([
            ref_len,
            student_len,
            length_ratio,
            common_words,
            word_overlap
        ])

    return np.array(features)


train_lexical = lexical_features(
    train_ref,
    train_student
)

val_lexical = lexical_features(
    val_ref,
    val_student
)


# ============================================================
# FEATURE MATRIX
# ============================================================

X_train = np.column_stack([
    train_semantic,
    train_tfidf_similarity,
    train_lexical
])

X_val = np.column_stack([
    val_semantic,
    val_tfidf_similarity,
    val_lexical
])

y_train = train_df["normalized_grade"].values
y_val = val_df["normalized_grade"].values

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# PREDICTION
# ============================================================

print("\nPredicting validation scores...")

predictions = model.predict(X_val)

predictions = np.clip(
    predictions,
    0,
    1
)


# ============================================================
# EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_val,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        predictions
    )
)

r2 = r2_score(
    y_val,
    predictions
)


print("\n" + "=" * 70)
print("ML MODEL RESULTS")
print("=" * 70)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = [
    "semantic_similarity",
    "tfidf_similarity",
    "reference_length",
    "student_length",
    "length_ratio",
    "common_words",
    "word_overlap"
]

print("\nFeature Importance:")
print("-" * 50)

for name, importance in zip(
    feature_names,
    model.feature_importances_
):
    print(
        f"{name:<24}: {importance:.4f}"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(5, len(val_df))):

    print("\n" + "-" * 60)

    print("Question:")
    print(val_df.iloc[i]["question"][:500])

    print("\nReference:")
    print(val_df.iloc[i]["ref_answer"][:500])

    print("\nStudent:")
    print(val_df.iloc[i]["student_answer"][:500])

    print(
        f"\nActual      : {y_val[i]:.3f}"
    )

    print(
        f"Predicted   : {predictions[i]:.3f}"
    )

    print(
        f"Predicted/10: {predictions[i] * 10:.2f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "ml_model.pkl"
tfidf_path = MODEL_DIR / "ml_tfidf.pkl"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    tfidf,
    tfidf_path
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("Model :", model_path)
print("TF-IDF:", tfidf_path)

print("\nML DOMAIN TRAINING COMPLETE")
print("=" * 70)