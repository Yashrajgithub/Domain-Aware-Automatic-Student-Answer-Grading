import pandas as pd

df = pd.read_parquet("dataset/train_clean.parquet")

domain_sources = {
    "General": ["SciEntsBank", "Beetle"],
    "Machine Learning": ["DigiKlausur"],
    "Data Engineering": ["SAF"]
}

print("=" * 60)
print("DOMAIN DATASET COUNTS")
print("=" * 60)

for domain, sources in domain_sources.items():

    subset = df[df["data_source"].isin(sources)]

    print(f"\n{domain}")
    print("-" * 40)

    print("Sources :", ", ".join(sources))
    print("Samples :", len(subset))

    print("\nScore distribution:")
    print(subset["normalized_grade"].value_counts().sort_index())

print("\n" + "=" * 60)