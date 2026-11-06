import os
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "dataset/train_clean.parquet"
VAL_PATH = "dataset/validation_clean.parquet"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ASAG2024 - MODEL 2")
print("Semantic + Lexical Features + Random Forest")
print("=" * 70)

train = pd.read_parquet(TRAIN_PATH)
val = pd.read_parquet(VAL_PATH)

print("\nTrain samples:", len(train))
print("Validation samples:", len(val))


# ============================================================
# LIMIT DATA FOR INITIAL EXPERIMENT
# ============================================================

# Start with full data.
# If your PC becomes too slow, we can reduce this later.

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ============================================================
# CREATE COMBINED TEXT
# ============================================================

train_reference = train["reference_answer"].tolist()
train_student = train["provided_answer"].tolist()

val_reference = val["reference_answer"].tolist()
val_student = val["provided_answer"].tolist()


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating reference embeddings...")

train_ref_embeddings = embedding_model.encode(
    train_reference,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating student embeddings...")

train_student_embeddings = embedding_model.encode(
    train_student,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nGenerating validation embeddings...")

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

train_semantic_similarity = np.sum(
    train_ref_embeddings * train_student_embeddings,
    axis=1
)

val_semantic_similarity = np.sum(
    val_ref_embeddings * val_student_embeddings,
    axis=1
)


# ============================================================
# TF-IDF SIMILARITY
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

print("TF-IDF vocabulary:", len(tfidf.vocabulary_))


def calculate_tfidf_similarity(references, students):

    ref_vectors = tfidf.transform(references)
    student_vectors = tfidf.transform(students)

    return cosine_similarity(
        ref_vectors,
        student_vectors
    ).diagonal()


train_tfidf_similarity = calculate_tfidf_similarity(
    train_reference,
    train_student
)

val_tfidf_similarity = calculate_tfidf_similarity(
    val_reference,
    val_student
)


# ============================================================
# LEXICAL FEATURES
# ============================================================

def lexical_features(references, students):

    features = []

    for ref, student in zip(references, students):

        ref_words = set(ref.lower().split())
        student_words = set(student.lower().split())

        common_words = ref_words.intersection(student_words)

        ref_length = len(ref_words)
        student_length = len(student_words)

        if ref_length == 0:
            overlap = 0
        else:
            overlap = len(common_words) / ref_length

        if ref_length == 0:
            length_ratio = 0
        else:
            length_ratio = student_length / ref_length

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
    train_semantic_similarity,
    train_tfidf_similarity,
    train_lexical
])

val_features = np.column_stack([
    val_semantic_similarity,
    val_tfidf_similarity,
    val_lexical
])

print("\nFeature shape:")
print("Train:", train_features.shape)
print("Validation:", val_features.shape)


# ============================================================
# TARGET
# ============================================================

y_train = train["normalized_grade"].values
y_val = val["normalized_grade"].values


# ============================================================
# RANDOM FOREST REGRESSOR
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
# PREDICTION
# ============================================================

print("\nPredicting validation scores...")

predictions = model.predict(val_features)

# Keep scores between 0 and 1
predictions = np.clip(predictions, 0, 1)


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
print("MODEL 2 RESULTS")
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
    print(f"{name:25s}: {importance:.4f}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for i in range(min(10, len(val))):

    print("\n------------------------------------------------------------")

    print("Question:")
    print(val.iloc[i]["question"][:250])

    print("\nReference:")
    print(val.iloc[i]["reference_answer"][:250])

    print("\nStudent:")
    print(val.iloc[i]["provided_answer"][:250])

    print(f"\nActual score     : {y_val[i]:.3f}")
    print(f"Predicted score  : {predictions[i]:.3f}")
    print(f"Predicted /10    : {predictions[i] * 10:.2f}")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = f"{MODEL_DIR}/asag_random_forest.pkl"
tfidf_path = f"{MODEL_DIR}/tfidf_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(tfidf, tfidf_path)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print("Model    :", model_path)
print("TF-IDF   :", tfidf_path)

print("\nMODEL 2 COMPLETE")
print("=" * 70)