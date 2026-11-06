import pandas as pd

DATASET_PATH = "dataset"

print("=" * 70)
print("ASAG2024 DATASET INSPECTION")
print("=" * 70)

# Load datasets
train = pd.read_parquet(f"{DATASET_PATH}/train.parquet")
validation = pd.read_parquet(f"{DATASET_PATH}/validation.parquet")
test = pd.read_parquet(f"{DATASET_PATH}/test.parquet")

# --------------------------------------------------
# 1. Dataset sizes
# --------------------------------------------------

print("\n1. DATASET SIZES")
print("-" * 70)

print("Train      :", train.shape)
print("Validation :", validation.shape)
print("Test       :", test.shape)

# --------------------------------------------------
# 2. Columns
# --------------------------------------------------

print("\n2. COLUMNS")
print("-" * 70)

print(train.columns.tolist())

# --------------------------------------------------
# 3. Data types
# --------------------------------------------------

print("\n3. DATA TYPES")
print("-" * 70)

print(train.dtypes)

# --------------------------------------------------
# 4. Missing values
# --------------------------------------------------

print("\n4. MISSING VALUES")
print("-" * 70)

print(train.isnull().sum())

# --------------------------------------------------
# 5. First 5 records
# --------------------------------------------------

print("\n5. SAMPLE RECORDS")
print("-" * 70)

print(train.head().to_string())

# --------------------------------------------------
# 6. Score distribution
# --------------------------------------------------

print("\n6. SCORE DISTRIBUTION")
print("-" * 70)

if "grade" in train.columns:
    print(train["grade"].describe())

# --------------------------------------------------
# 7. Normalized score
# --------------------------------------------------

if "normalized_grade" in train.columns:
    print("\n7. NORMALIZED GRADE")
    print("-" * 70)
    print(train["normalized_grade"].describe())

# --------------------------------------------------
# 8. Data sources / domains
# --------------------------------------------------

if "data_source" in train.columns:
    print("\n8. DATA SOURCES")
    print("-" * 70)
    print(train["data_source"].value_counts())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)