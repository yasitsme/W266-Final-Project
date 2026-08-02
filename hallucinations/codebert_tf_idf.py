import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import json
import random
import pickle
import numpy as np
import torch
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, classification_report,
)
from transformers import AutoTokenizer, AutoModel

DATASET_PATH = "juliet_hallucination_dataset_5k.json"
GOLD_SET_PATH = "gold_set.json"
MODEL_NAME = "microsoft/codebert-base"
SAVE_PATH = "codebert_hybrid_classifier.pkl"
MAX_LENGTH = 512
TEST_FRACTION = 0.15
RANDOM_SEED = 42
EMBED_BATCH_SIZE = 16
EXCLUDED_TYPES = {"DUPLICATE_OR_DEGENERATE", "EMPTY_EXTRACTION", "GENERATION_ERROR", "AST_PARSE_ERROR"}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def clean_code(text):
    if not text:
        return ""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*?(?:\n|$)', '\n', text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for row in raw:
        if row.get("hallucination_type", "NONE") in EXCLUDED_TYPES:
            continue
        code = row.get("extracted_code") or row.get("model_generated_output") or ""
        if not code.strip():
            continue
        case_id = row.get("case_identifier", "unknown")
        rows.append({
            "text": f"{clean_code(row.get('prompt_context', ''))}\n{clean_code(code)}",
            "label": int(bool(row.get("is_hallucinated", False))),
            "cwe_group": case_id.split("__")[0],
        })

    print(f"Loaded {len(raw)} rows, kept {len(rows)} after filtering.")
    return rows


def group_split(rows, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    groups = sorted(set(r["cwe_group"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(groups)

    target = int(len(rows) * test_fraction)
    test_groups, count = set(), 0
    for g in groups:
        if count >= target:
            break
        n = sum(1 for r in rows if r["cwe_group"] == g)
        test_groups.add(g)
        count += n

    train = [r for r in rows if r["cwe_group"] not in test_groups]
    test = [r for r in rows if r["cwe_group"] in test_groups]
    print(f"Split: {len(train)} train / {len(test)} eval ({len(test_groups)}/{len(groups)} CWE groups held out)")
    return train, test


def embed(texts, tokenizer, model, device, batch_size=EMBED_BATCH_SIZE):
    """Frozen [CLS] embeddings -- no gradients, no fine-tuning."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                                padding=True, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state[:, 0, :]
            out.append(hidden.cpu().numpy())
    return np.vstack(out)


def featurize(texts, vectorizer, tokenizer, embed_model, device, fit=False):
    tfidf = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    codebert = embed(texts, tokenizer, embed_model, device)
    return hstack([tfidf, csr_matrix(codebert)]).tocsr()


def report(name, labels, preds):
    print(f"\n{name}")
    print(f"  accuracy:  {accuracy_score(labels, preds):.3f}")
    print(f"  f1:        {f1_score(labels, preds, zero_division=0):.3f}")
    print(f"  precision: {precision_score(labels, preds, zero_division=0):.3f}")
    print(f"  recall:    {recall_score(labels, preds, zero_division=0):.3f}")
    print(f"  MCC:       {matthews_corrcoef(labels, preds):.3f}")
    print(f"  confusion matrix: {confusion_matrix(labels, preds, labels=[0, 1]).tolist()}")


def main():
    rows = load_rows(DATASET_PATH)
    train_rows, eval_rows = group_split(rows)
    train_texts = [r["text"] for r in train_rows]
    eval_texts = [r["text"] for r in eval_rows]
    train_labels = [r["label"] for r in train_rows]
    eval_labels = [r["label"] for r in eval_rows]

    print(f"Loading {MODEL_NAME} for frozen feature extraction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embed_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train = featurize(train_texts, vectorizer, tokenizer, embed_model, device, fit=True)
    X_eval = featurize(eval_texts, vectorizer, tokenizer, embed_model, device)
    print(f"Feature matrix: {X_train.shape[1]} dims")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train, train_labels)

    train_preds = clf.predict(X_train)
    report("Train split (sanity check)", train_labels, train_preds)
    eval_preds = clf.predict(X_eval)
    report("Eval split (auto-labeled)", eval_labels, eval_preds)
    print()
    print(classification_report(eval_labels, eval_preds,
                                 target_names=["not_hallucinated", "hallucinated"], zero_division=0))

    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "classifier": clf, "codebert_model_name": MODEL_NAME}, f)
    print(f"Saved to {SAVE_PATH}")

    if os.path.exists(GOLD_SET_PATH):
        with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
            gold_rows = [r for r in json.load(f) if r.get("human_label") is not None]

        if gold_rows:
            gold_texts = [f"{clean_code(r.get('prompt_context',''))}\n{clean_code(r.get('extracted_code',''))}"
                          for r in gold_rows]
            gold_labels = [int(bool(r["human_label"])) for r in gold_rows]
            X_gold = featurize(gold_texts, vectorizer, tokenizer, embed_model, device)
            gold_preds = clf.predict(X_gold)
            report("Gold set (human-labeled)", gold_labels, gold_preds)
        else:
            print(f"\n{GOLD_SET_PATH} has no labeled rows -- skipping gold-set eval.")
    else:
        print(f"\n{GOLD_SET_PATH} not found -- skipping gold-set eval.")


if __name__ == "__main__":
    main()