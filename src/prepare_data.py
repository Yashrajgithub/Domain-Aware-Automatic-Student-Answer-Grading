import pandas as pd

DATASET_PATH = "dataset"

print("=" * 70)
print("PREPARING ASAG2024 DATASET")
print("=" * 70)

# Load datasets
train = pd.read_parquet(f"{DATASET_PATH}/train.parquet")
validation = pd.read_parquet(f"{DATASET_PATH}/validation.parquet")
test = pd.read_parquet(f"{DATASET_PATH}/test.parquet")

print("\nOriginal sizes:")
print("Train      :", train.shape)
print("Validation :", validation.shape)
print("Test       :", test.shape)

# Remove rows missing required text
required_columns = [
    "question",
    "provided_answer",
    "reference_answer",
    "normalized_grade"
]

train = train.dropna(subset=required_columns).copy()
validation = validation.dropna(subset=required_columns).copy()
test = test.dropna(subset=required_columns).copy()

# Convert text columns to string
for df in [train, validation, test]:
    df["question"] = df["question"].astype(str)
    df["provided_answer"] = df["provided_answer"].astype(str)
    df["reference_answer"] = df["reference_answer"].astype(str)

print("\nAfter removing missing values:")
print("Train      :", train.shape)
print("Validation :", validation.shape)
print("Test       :", test.shape)

# Save processed datasets
train.to_parquet("dataset/train_clean.parquet", index=False)
validation.to_parquet("dataset/validation_clean.parquet", index=False)
test.to_parquet("dataset/test_clean.parquet", index=False)

print("\nClean datasets saved successfully.")

print("\nTarget variable:")
print("normalized_grade")

print("\nTraining score distribution:")
print(train["normalized_grade"].value_counts().sort_index())

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)