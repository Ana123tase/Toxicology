import pandas as pd
df = pd.read_csv("data/Ames_clean.csv")  # or wherever it sits pre-featurize
print([c for c in df.columns if c.upper().startswith("TA")])
print(df["Overall"].value_counts(dropna=False))