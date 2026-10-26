"""Train & compare classifiers, save the best one by F1 on the fake class.

Usage:  python -m src.train            (auto-picks real dataset if present)
"""
import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocess import build_dataframe

FLAG_COLS = ["asks_money", "free_email", "urgency", "instant_selection",
             "whatsapp_telegram", "has_company_website", "exclamations", "text_length"]


def load_data():
    for path in ["data/fake_job_postings.csv", "data/sample_postings.csv"]:
        if os.path.exists(path):
            print(f"Using dataset: {path}")
            return pd.read_csv(path)
    raise SystemExit("No dataset found. Run: python -m src.generate_sample_data")


def main():
    os.makedirs("models", exist_ok=True)
    df = build_dataframe(load_data())
    print("Class balance:\n", df["fraudulent"].value_counts().to_string())

    train, test = train_test_split(df, test_size=0.2, stratify=df["fraudulent"], random_state=42)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    Xtr = hstack([tfidf.fit_transform(train["clean"]), csr_matrix(train[FLAG_COLS].values)]).tocsr()
    Xte = hstack([tfidf.transform(test["clean"]), csr_matrix(test[FLAG_COLS].values)]).tocsr()
    ytr, yte = train["fraudulent"], test["fraudulent"]

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
    }

    results, fitted = [], {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        fitted[name] = (model, pred)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
        })

    res = pd.DataFrame(results).sort_values("F1", ascending=False).round(4)
    print("\n=== Model comparison (fake class) ===")
    print(res.to_string(index=False))
    res.to_csv("models/results.csv", index=False)

    best_name = res.iloc[0]["Model"]
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model: {best_name}\n")
    print(classification_report(yte, best_pred, target_names=["Genuine", "Fake"]))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(yte, best_pred, display_labels=["Genuine", "Fake"], ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig("models/confusion_matrix.png", dpi=150)

    joblib.dump({"model": best_model, "tfidf": tfidf, "name": best_name, "flag_cols": FLAG_COLS}, "models/fake_internship_model.joblib")
    print("Saved models/fake_internship_model.joblib")


if __name__ == "__main__":
    main()
