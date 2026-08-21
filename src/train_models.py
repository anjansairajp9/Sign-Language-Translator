import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


DATASET_PATH = "data/landmarks_v2.csv"
MODEL_DIR = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2


df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["label"])
y = df["label"]

labels = sorted(y.unique())

label_to_id = {
    label: index
    for index, label in enumerate(labels)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}

y_encoded = y.map(label_to_id)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)


models = {
    "Random Forest": Pipeline([
        (
            "variance",
            VarianceThreshold(threshold=0.0)
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]),

    "SVM": Pipeline([
        (
            "variance",
            VarianceThreshold(threshold=0.0)
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                C=10,
                kernel="rbf",
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        )
    ]),

    "XGBoost": Pipeline([
        (
            "variance",
            VarianceThreshold(threshold=0.0)
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])
}


os.makedirs(MODEL_DIR, exist_ok=True)

results = {}


print("=" * 70)
print("SIGN LANGUAGE MODEL TRAINING")
print("=" * 70)

print()
print(f"Samples          : {len(df)}")
print(f"Features          : {X.shape[1]}")
print(f"Classes           : {len(labels)}")
print(f"Training samples  : {len(X_train)}")
print(f"Testing samples   : {len(X_test)}")

print()
print("Classes:")

for index, label in id_to_label.items():
    print(f"{index}: {label}")


for model_name, model in models.items():

    print()
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[model_name] = accuracy

    print()
    print(f"Accuracy: {accuracy:.4f}")

    print()
    print("Classification Report")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=labels,
            zero_division=0
        )
    )

    print("Confusion Matrix")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print()
    print(f"Model saved: {model_path}")


best_model_name = max(
    results,
    key=results.get
)

best_model = models[best_model_name]

best_model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

joblib.dump(
    best_model,
    best_model_path
)

metadata = {
    "labels": labels,
    "label_to_id": label_to_id,
    "id_to_label": id_to_label,
    "feature_count": X.shape[1],
    "best_model": best_model_name
}

joblib.dump(
    metadata,
    os.path.join(
        MODEL_DIR,
        "metadata.joblib"
    )
)


print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for model_name, accuracy in results.items():
    print(
        f"{model_name:<20} "
        f"{accuracy * 100:.2f}%"
    )

print()
print(f"Best model: {best_model_name}")
print(f"Saved to: {best_model_path}")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
