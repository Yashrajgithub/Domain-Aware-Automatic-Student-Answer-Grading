import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "dataset/train_clean.parquet"
VAL_PATH = "dataset/validation_clean.parquet"

MODEL_DIR = "models"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DOMAIN_SOURCES = {
    "general": ["SciEntsBank", "Beetle"],
    "de": ["SAF"]
}


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--domain",
    required=True,
    choices=["general", "de"],
    help="Domain to train"
)

args = parser.parse_args()

domain = args.domain

sources = DOMAIN_SOURCES[domain]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(f"TRAINING DOMAIN MODEL: {domain.upper()}")
print("=" * 70)

print("Sources:", sources)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_parquet(TRAIN_PATH)
val = pd.read_parquet(VAL_PATH)


# ============================================================
# SELECT DOMAIN
# ============================================================

train = train[
    train["data_source"].isin(sources)
].copy()

val = val[
    val["data_source"].isin(sources)
].copy()


# ============================================================
# REMOVE MISSING DATA
# ============================================================

required_columns = [
    "question",
    "provided_answer",
    "reference_answer",
    "normalized_grade"
]

train = train.dropna(
    subset=required_columns
).reset_index(drop=True)

val = val.dropna(
    subset=required_columns
).reset_index(drop=True)


print("\nDataset sizes:")
print("Train:", len(train))
print("Validation:", len(val))


if len(train) == 0:
    raise ValueError(
        f"No training samples found for domain: {domain}"
    )


# ============================================================
# TEXT DATA
# ============================================================

train_reference = train["reference_answer"].astype(str).tolist()
train_student = train["provided_answer"].astype(str).tolist()

val_reference = val["reference_answer"].astype(str).tolist()
val_student = val["provided_answer"].astype(str).tolist()


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating training reference embeddings...")

train_ref_embeddings = embedding_model.encode(
    train_reference,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating training student embeddings...")

train_student_embeddings = embedding_model.encode(
    train_student,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating validation reference embeddings...")

val_ref_embeddings = embedding_model.encode(
    val_reference,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating validation student embeddings...")

val_student_embeddings = embedding_model.encode(
    val_student,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

print("\nCalculating semantic similarity...")

train_semantic = np.sum(
    train_ref_embeddings * train_student_embeddings,
    axis=1
)

val_semantic = np.sum(
    val_ref_embeddings * val_student_embeddings,
    axis=1
)


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000
)

tfidf.fit(
    train_reference + train_student
)

print(
    "TF-IDF vocabulary:",
    len(tfidf.vocabulary_)
)


def calculate_tfidf_similarity(
    references,
    students
):

    ref_vectors = tfidf.transform(
        references
    )

    student_vectors = tfidf.transform(
        students
    )

    return cosine_similarity(
        ref_vectors,
        student_vectors
    ).diagonal()


print("\nCalculating TF-IDF similarity...")

train_tfidf = calculate_tfidf_similarity(
    train_reference,
    train_student
)

val_tfidf = calculate_tfidf_similarity(
    val_reference,
    val_student
)


# ============================================================
# LEXICAL FEATURES
# ============================================================

def lexical_features(
    references,
    students
):

    features = []

    for ref, student in zip(
        references,
        students
    ):

        ref_words = set(
            ref.lower().split()
        )

        student_words = set(
            student.lower().split()
        )

        common_words = (
            ref_words.intersection(
                student_words
            )
        )

        ref_length = len(ref_words)

        student_length = len(
            student_words
        )

        if ref_length == 0:

            overlap = 0
            length_ratio = 0

        else:

            overlap = (
                len(common_words)
                / ref_length
            )

            length_ratio = (
                student_length
                / ref_length
            )

        features.append([
            ref_length,
            student_length,
            length_ratio,
            len(common_words),
            overlap
        ])

    return np.array(features)


print("\nCalculating lexical features...")

train_lexical = lexical_features(
    train_reference,
    train_student
)

val_lexical = lexical_features(
    val_reference,
    val_student
)


# ============================================================
# COMBINE FEATURES
# ============================================================

train_features = np.column_stack([
    train_semantic,
    train_tfidf,
    train_lexical
])

val_features = np.column_stack([
    val_semantic,
    val_tfidf,
    val_lexical
])


print("\nFeature shapes:")
print("Train:", train_features.shape)
print("Validation:", val_features.shape)


# ============================================================
# TARGET
# ============================================================

y_train = train[
    "normalized_grade"
].values

y_val = val[
    "normalized_grade"
].values


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

model.fit(
    train_features,
    y_train
)

print("Training complete.")


# ============================================================
# VALIDATION
# ============================================================

print("\nPredicting validation scores...")

predictions = model.predict(
    val_features
)

predictions = np.clip(
    predictions,
    0,
    1
)


# ============================================================
# METRICS
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
print(f"{domain.upper()} MODEL RESULTS")
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
        f"{name:25s}: "
        f"{importance:.4f}"
    )


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(
    min(5, len(val))
):

    print("\n------------------------------------------------------------")

    print(
        "Question:",
        val.iloc[i]["question"][:200]
    )

    print(
        "Reference:",
        val.iloc[i]["reference_answer"][:200]
    )

    print(
        "Student:",
        val.iloc[i]["provided_answer"][:200]
    )

    print(
        f"Actual      : {y_val[i]:.3f}"
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

model_path = (
    f"{MODEL_DIR}/{domain}_model.pkl"
)

tfidf_path = (
    f"{MODEL_DIR}/{domain}_tfidf.pkl"
)

joblib.dump(
    model,
    model_path
)

joblib.dump(
    tfidf,
    tfidf_path
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "domain": domain,
    "sources": sources,
    "embedding_model": EMBEDDING_MODEL,
    "feature_names": feature_names
}

joblib.dump(
    config,
    f"{MODEL_DIR}/{domain}_config.pkl"
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("Model :", model_path)
print("TF-IDF:", tfidf_path)
print(
    "Config:",
    f"{MODEL_DIR}/{domain}_config.pkl"
)

print("\nDOMAIN TRAINING COMPLETE")
print("=" * 70)