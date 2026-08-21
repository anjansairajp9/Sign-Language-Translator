import pandas as pd
import numpy as np


DATASET_PATH = "data/landmarks_v2.csv"


df = pd.read_csv(DATASET_PATH)

features = df.drop(columns=["label"])
labels = df["label"]


print("=" * 60)
print("V2 DATASET ANALYSIS")
print("=" * 60)

print()
print("Dataset shape:")
print(df.shape)

print()
print("Samples per class:")
print(labels.value_counts().sort_index())

print()
print("Number of classes:")
print(labels.nunique())

print()
print("Number of features:")
print(features.shape[1])

print()
print("Missing values:")
print(df.isnull().sum().sum())

print()
print("Infinite values:")
print(
    np.isinf(features.to_numpy()).sum()
)

print()
print("Duplicate rows:")
print(df.duplicated().sum())

print()
print("Constant features:")
constant_features = [
    column
    for column in features.columns
    if features[column].nunique() <= 1
]

print(len(constant_features))

print()
print("Feature statistics:")
print(features.describe().loc[["mean", "std", "min", "max"]])

print()
print("=" * 60)
print("DATASET ANALYSIS COMPLETE")
print("=" * 60)