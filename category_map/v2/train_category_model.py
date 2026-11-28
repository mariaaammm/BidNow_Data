import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

MODEL_PATH = HERE / "category_model.joblib"
DEFAULT_DATA = PARENT / "data" / "category_data.csv"


def make_features(word_vec, char_vec, texts, fit=False):
    texts = [str(t) for t in texts]
    if fit:
        w = word_vec.fit_transform(texts)
        c = char_vec.fit_transform(texts)
    else:
        w = word_vec.transform(texts)
        c = char_vec.transform(texts)
    return hstack([w, c]).tocsr()


def train_model(data_path):
    df = pd.read_csv(data_path).dropna(subset=["title", "category"])

    if "split" in df.columns and {"train", "test"}.issubset(set(df["split"])):
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)

    X_train = make_features(word_vec, char_vec, train_df["title"], fit=True)
    X_test = make_features(word_vec, char_vec, test_df["title"], fit=False)

    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(X_train, train_df["category"])

    preds = clf.predict(X_test)
    exact_acc = (preds == test_df["category"]).mean()
    macro_f1 = f1_score(test_df["category"], preds, average="macro", zero_division=0)

    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Categories: {df['category'].nunique()}")
    print(f"ML Exact Accuracy: {exact_acc:.1%} | Macro F1: {macro_f1:.3f}")

    try:
        from category_preprocessing import classify
        rule_preds = test_df["title"].map(classify)
        rule_acc = (rule_preds == test_df["category"]).mean()
        print(f"Rule-based Exact Accuracy: {rule_acc:.1%}")
    except Exception:
        pass

    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def predict_titles(titles):
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run `train` first.")

    bundle = joblib.load(MODEL_PATH)
    X = make_features(bundle["word_vec"], bundle["char_vec"], titles, fit=False)
    probs = bundle["clf"].predict_proba(X)
    classes = bundle["clf"].classes_

    for title, row in zip(titles, probs):
        top_idx = row.argsort()[::-1][:3]
        print(f"\n{title!r}")
        for i in top_idx:
            print(f"  {row[i]:6.1%}  {classes[i]}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data", default=str(DEFAULT_DATA))

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("titles", nargs="+")

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(args.data)
    elif args.cmd == "predict":
        predict_titles(args.titles)


if __name__ == "__main__":
    main()