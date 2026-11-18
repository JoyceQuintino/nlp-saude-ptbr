"""Baseline clássico: TF-IDF + Regressão Logística (scikit-learn)."""
import argparse

import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_PATH, load_splits
from src.metrics import evaluate, save_confusion_matrix


def main(args):
    train, _, test, label2id = load_splits(args.data)
    names = list(label2id)

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=args.C)),
        ]
    )
    mlflow.set_experiment("nlp-saude-ptbr")
    with mlflow.start_run(run_name="tfidf_logreg"):
        pipe.fit(train["text"], train["y"])
        preds = pipe.predict(test["text"])
        res = evaluate(test["y"], preds, names)
        mlflow.log_params({"model": "tfidf_logreg", "C": args.C})
        mlflow.log_metrics({"test_f1_macro": res["f1_macro"], "test_accuracy": res["accuracy"]})
        save_confusion_matrix(test["y"], preds, names, "reports/cm_tfidf.png")
        mlflow.log_artifact("reports/cm_tfidf.png")
        print(res["report"])
        print(f"F1 macro (teste): {res['f1_macro']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_PATH)
    p.add_argument("--C", type=float, default=1.0)
    main(p.parse_args())
