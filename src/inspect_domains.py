import pandas as pd

DATASET_PATH = "dataset/train_clean.parquet"

df = pd.read_parquet(DATASET_PATH)

print("=" * 70)
print("ASAG2024 DOMAIN / SOURCE INSPECTION")
print("=" * 70)

for source in sorted(df["data_source"].unique()):

    subset = df[df["data_source"] == source]

    print("\n" + "=" * 70)
    print(f"SOURCE: {source}")
    print(f"NUMBER OF ANSWERS: {len(subset)}")
    print("=" * 70)

    # Show 10 random questions
    samples = subset.sample(
        n=min(10, len(subset)),
        random_state=42
    )

    for i, (_, row) in enumerate(samples.iterrows(), 1):

        print(f"\n{i}. QUESTION:")
        print(row["question"][:500])

        print("\n   REFERENCE:")
        print(row["reference_answer"][:300])

        print("\n   SCORE:", row["normalized_grade"])

print("\n" + "=" * 70)
print("DOMAIN INSPECTION COMPLETE")
print("=" * 70)