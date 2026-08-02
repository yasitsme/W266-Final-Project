import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import json
import random
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix,
)

DATASET_PATH = "juliet_hallucination_dataset_5k.json"
GOLD_SET_PATH = "gold_set.json"
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "./lora_hallucination_detector_codebert"
FINAL_SAVE_PATH = "./final_lora_detector_weights_codebert"
MAX_LENGTH = 512
TEST_FRACTION = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 8
NUM_EPOCHS = 5
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
        cwe_group = case_id.split("__")[0]
        rows.append({
            "prompt_context": clean_code(row.get("prompt_context", "")),
            "extracted_code": clean_code(code),
            "labels": int(bool(row.get("is_hallucinated", False))),
            "cwe_group": cwe_group,
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


def class_weights_from(labels, max_ratio=5.0):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])
    total = n_pos + n_neg
    w = [total / (2 * n_neg), total / (2 * n_pos)]
    if max(w) / min(w) > max_ratio:
        scale = max_ratio / (max(w) / min(w))
        w = [v * scale if v == max(w) else v for v in w]
    return torch.tensor(w, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device).to(logits.dtype)
        loss = nn.CrossEntropyLoss(weight=weights)(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def predict_all(model, tokenizer, prompts, codes, device):
    """One example at a time, no batching or Dataset object -- keeps
    memory/behavior predictable regardless of dataset size."""
    preds = []
    model.eval()
    with torch.no_grad():
        for p, c in zip(prompts, codes):
            inputs = tokenizer(p, c, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            preds.append(int(torch.argmax(logits, dim=-1).item()))
    return preds


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

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(batch["prompt_context"], batch["extracted_code"],
                          truncation=True, max_length=MAX_LENGTH)

    train_tok = train_ds.map(tokenize, batched=True)
    eval_tok = eval_ds.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"], modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    weights = class_weights_from(train_tok["labels"])
    print(f"Class weights: {weights.tolist()}")

    steps_per_epoch = -(-len(train_tok) // BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(0.1 * total_steps))

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.05,
        max_grad_norm=1.0,
        warmup_steps=warmup_steps,
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_tok,
        processing_class=tokenizer, data_collator=collator, class_weights=weights,
    )

    trainer.train()
    model.save_pretrained(FINAL_SAVE_PATH)
    tokenizer.save_pretrained(FINAL_SAVE_PATH)
    print(f"Adapter saved to {FINAL_SAVE_PATH}")

    device = next(model.parameters()).device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_prompts = [r["prompt_context"] for r in eval_rows]
    eval_codes = [r["extracted_code"] for r in eval_rows]
    eval_labels = [r["labels"] for r in eval_rows]
    eval_preds = predict_all(model, tokenizer, eval_prompts, eval_codes, device)
    report("Eval split (auto-labeled)", eval_labels, eval_preds)

    if os.path.exists(GOLD_SET_PATH):
        with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
            gold_rows = [r for r in json.load(f) if r.get("human_label") is not None]

        if gold_rows:
            gold_prompts = [clean_code(r.get("prompt_context", "")) for r in gold_rows]
            gold_codes = [clean_code(r.get("extracted_code", "")) for r in gold_rows]
            gold_labels = [int(bool(r["human_label"])) for r in gold_rows]
            gold_preds = predict_all(model, tokenizer, gold_prompts, gold_codes, device)
            report("Gold set (human-labeled)", gold_labels, gold_preds)
        else:
            print(f"\n{GOLD_SET_PATH} has no labeled rows -- skipping gold-set eval.")
    else:
        print(f"\n{GOLD_SET_PATH} not found -- skipping gold-set eval.")


if __name__ == "__main__":
    main()