"""
Step 3: Fine-tune mT5-small end-to-end (category + sentiment + summary
in ONE generation pass) — Colab-ready.

HOW TO USE:
  1. Open a new Colab notebook, set Runtime -> GPU.
  2. Upload train_dataset.csv and hard_test_set.csv (from Step 1).
  3. Paste each "# %% CELL n" block into its own Colab cell, run top to bottom.

WHY THIS FIXES THE ORIGINAL SUMMARIZATION PROBLEM:
  - mT5-small is an encoder-DECODER model, so unlike every BERT variant used
    in the old notebook, it can actually generate text - no more TF-IDF
    extractive fallbacks, no more separate AR/EN summarizer routing.
  - We frame the whole task as text-to-text: input = raw ticket, target =
    a single structured string "category: X | sentiment: Y | summary: Z".
    One fine-tune, one model, one forward pass covers everything Sys1 +
    Sys2 + Sys2.1 needed 3 separate pipelines to attempt.
  - The model naturally summarizes in whichever language/mix the ticket
    was written in, because it's not routed through a language-specific
    summarizer - it just learns from the summary_target examples directly.

PARSING NOTE:
  Since output is free-text, we parse the "category: X | sentiment: Y |
  summary: Z" pattern back out after generation. If the model ever produces
  a category/sentiment string that isn't an exact match to our label list
  (rare after fine-tuning, but possible on hard/novel inputs), we flag it as
  "UNCERTAIN" rather than silently guessing - the plan's confidence flag.
"""

# %% CELL 1 — Install & imports
# !pip install -q transformers accelerate sentencepiece sacrebleu rouge-score pandas scikit-learn

import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "google/mt5-small"
CATEGORY_LABELS = ["Billing", "Network", "Technical Support", "Sales/Plans", "Other"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


# %% CELL 2 — Load data & build the text-to-text target string
train_df = pd.read_csv("train_dataset.csv")
hard_df = pd.read_csv("hard_test_set.csv")


def build_target(row):
    return f"category: {row['category']} | sentiment: {row['sentiment']} | summary: {row['summary_target']}"


train_df["target_text"] = train_df.apply(build_target, axis=1)
hard_df["target_text"] = hard_df.apply(build_target, axis=1)

# Prefix task instruction — helps mT5 (trained on span-corruption, not
# instructions) learn faster that this is a structured extraction task
PREFIX = "classify and summarize ticket: "
train_df["input_text"] = PREFIX + train_df["text"]
hard_df["input_text"] = PREFIX + hard_df["text"]

train_split, val_split = train_test_split(train_df, test_size=0.15, random_state=42)
print(f"Train: {len(train_split)} | Val: {len(val_split)} | Hard test: {len(hard_df)}")
print("\nExample target string:\n", train_df.iloc[0]["target_text"])


# %% CELL 3 — Dataset & tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

MAX_INPUT_LEN = 96
MAX_TARGET_LEN = 80


class TicketSeq2SeqDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = df["input_text"].tolist()
        self.targets = df["target_text"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        model_inputs = self.tokenizer(
            self.inputs[idx], truncation=True, padding="max_length",
            max_length=MAX_INPUT_LEN, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                self.targets[idx], truncation=True, padding="max_length",
                max_length=MAX_TARGET_LEN, return_tensors="pt"
            )
        item = {k: v.squeeze(0) for k, v in model_inputs.items()}
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100  # ignore pad in loss
        item["labels"] = label_ids
        return item


train_ds = TicketSeq2SeqDataset(train_split, tokenizer)
val_ds = TicketSeq2SeqDataset(val_split, tokenizer)


# %% CELL 4 — Train
training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_ticket_ckpt",
    num_train_epochs=3,                 # was 15 — 3k rows is enough signal that 15 epochs will just overfit/memorize
    per_device_train_batch_size=16,     # T4 has 16GB, mt5-small is small enough to raise this from 8
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,      # effective batch size 32, smoother gradients on more data
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=3e-4,
    warmup_ratio=0.06,                  # short warmup helps stability now that step count is much lower (3k rows vs 300)
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    logging_steps=50,                   # 20 was tuned for tiny datasets; 50 avoids log spam over more steps
    fp16=True,                          # T4 supports fp16 — meaningful speedup, MPS/CPU can't do this but T4 can
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()


# %% CELL 5 — Parsing helper (structured string -> dict, with UNCERTAIN fallback)
PARSE_RE = re.compile(
    r"category:\s*(?P<category>.*?)\s*\|\s*sentiment:\s*(?P<sentiment>.*?)\s*\|\s*summary:\s*(?P<summary>.*)",
    re.IGNORECASE | re.DOTALL
)


def parse_output(raw_text):
    m = PARSE_RE.search(raw_text)
    if not m:
        return {"category": "UNCERTAIN", "sentiment": "UNCERTAIN", "summary": raw_text.strip()}

    cat = m.group("category").strip()
    sent = m.group("sentiment").strip()
    summary = m.group("summary").strip()

    cat_match = next((c for c in CATEGORY_LABELS if c.lower() == cat.lower()), "UNCERTAIN")
    sent_match = next((s for s in SENTIMENT_LABELS if s.lower() == sent.lower()), "UNCERTAIN")

    return {"category": cat_match, "sentiment": sent_match, "summary": summary}


def generate_for_text(text):
    inputs = tokenizer(PREFIX + text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=MAX_TARGET_LEN, num_beams=4, no_repeat_ngram_size=3
        )
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return parse_output(raw), raw


# %% CELL 6 — Evaluate on the HARD test set
hard_preds = []
for _, row in hard_df.iterrows():
    parsed, raw = generate_for_text(row["text"])
    hard_preds.append(parsed)

pred_cat = [p["category"] for p in hard_preds]
pred_sent = [p["sentiment"] for p in hard_preds]
true_cat = hard_df["category"].tolist()
true_sent = hard_df["sentiment"].tolist()

print("=== HARD TEST SET — Category (mT5-small generation) ===")
print(classification_report(true_cat, pred_cat, zero_division=0))
print("=== HARD TEST SET — Sentiment (mT5-small generation) ===")
print(classification_report(true_sent, pred_sent, zero_division=0))

uncertain_count = sum(1 for p in hard_preds if "UNCERTAIN" in (p["category"], p["sentiment"]))
print(f"\nUNCERTAIN (unparseable) outputs: {uncertain_count}/{len(hard_preds)}")

results_df = hard_df[["text", "category", "sentiment", "summary_target"]].copy()
results_df["predicted_category"] = pred_cat
results_df["predicted_sentiment"] = pred_sent
results_df["generated_summary"] = [p["summary"] for p in hard_preds]
results_df.to_csv("mt5_hard_test_predictions.csv", index=False)
print(results_df[["text", "category", "predicted_category", "sentiment", "predicted_sentiment", "generated_summary"]])


# %% CELL 7 — Save the fine-tuned model (for the Streamlit app later)
model.save_pretrained("mt5_ticket_model")
tokenizer.save_pretrained("mt5_ticket_model")
print("Saved to mt5_ticket_model/ — download this folder before your Colab session ends.")
