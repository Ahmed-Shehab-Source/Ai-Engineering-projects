"""
Step 2: Fine-tune MARBERTv2 (multi-task: category + sentiment) — Colab-ready.

HOW TO USE:
  1. Open a new Colab notebook, set Runtime -> GPU.
  2. Upload train_dataset.csv and hard_test_set.csv (from Step 1) to the
     Colab working directory (or mount Drive and adjust the paths below).
  3. Paste each "# %% CELL n" block into its own Colab cell, in order, and
     run top to bottom.

WHY MARBERTv2 (vs the old mBERT/AraBERT split):
  - Pretrained on Arabic dialect + social-media text (Twitter-scale corpus),
    so it already "speaks" Egyptian Arabic and code-switched text natively -
    no more separate EN/AR routing logic needed (Sys1/Sys2 language split
    is gone; this is one model for everything).

WHY ONE MODEL, TWO HEADS (multi-task):
  - category (5 classes) and sentiment (3 classes) share the same underlying
    understanding of the ticket, so a shared encoder + two small classifier
    heads trains faster and more robustly than two fully separate models,
    while still giving you two independent predictions to evaluate/report.

NOTE: MARBERTv2 has no decoder, so it still cannot produce the `summary_target`
column - that's what the mT5-small fine-tune (Step 3) is for. This script is
the "stronger BERT baseline" side of the comparison from the plan.
"""

# %% CELL 1 — Install & imports
# !pip install -q transformers accelerate scikit-learn pandas

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_NAME = "UBC-NLP/MARBERTv2"
CATEGORY_LABELS = ["Billing", "Network", "Technical Support", "Sales/Plans", "Other"]
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

cat2id = {c: i for i, c in enumerate(CATEGORY_LABELS)}
sent2id = {s: i for i, s in enumerate(SENTIMENT_LABELS)}


# %% CELL 2 — Load data
train_df = pd.read_csv("train_dataset.csv")
hard_df = pd.read_csv("hard_test_set.csv")

train_df["cat_id"] = train_df["category"].map(cat2id)
train_df["sent_id"] = train_df["sentiment"].map(sent2id)
hard_df["cat_id"] = hard_df["category"].map(cat2id)
hard_df["sent_id"] = hard_df["sentiment"].map(sent2id)

train_split, val_split = train_test_split(
    train_df, test_size=0.15, random_state=42, stratify=train_df["cat_id"]
)
print(f"Train: {len(train_split)} | Val: {len(val_split)} | Hard test: {len(hard_df)}")


# %% CELL 3 — Dataset & multi-task model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TicketDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=96):
        self.texts = df["text"].tolist()
        self.cat_ids = df["cat_id"].tolist()
        self.sent_ids = df["sent_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["cat_label"] = torch.tensor(self.cat_ids[idx], dtype=torch.long)
        item["sent_label"] = torch.tensor(self.sent_ids[idx], dtype=torch.long)
        return item


class MultiTaskMARBERT(nn.Module):
    def __init__(self, model_name, n_cat, n_sent, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cat_head = nn.Linear(hidden, n_cat)
        self.sent_head = nn.Linear(hidden, n_sent)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.cat_head(pooled), self.sent_head(pooled)


model = MultiTaskMARBERT(MODEL_NAME, len(CATEGORY_LABELS), len(SENTIMENT_LABELS)).to(device)

train_ds = TicketDataset(train_split, tokenizer)
val_ds = TicketDataset(val_split, tokenizer)
hard_ds = TicketDataset(hard_df, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
hard_loader = DataLoader(hard_ds, batch_size=16)


# %% CELL 4 — Training loop
EPOCHS = 8
LR = 2e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = nn.CrossEntropyLoss()

def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, cat_preds, cat_true, sent_preds, sent_true = 0, [], [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        cat_label = batch["cat_label"].to(device)
        sent_label = batch["sent_label"].to(device)

        with torch.set_grad_enabled(train_mode):
            cat_logits, sent_logits = model(input_ids, attn)
            loss = loss_fn(cat_logits, cat_label) + loss_fn(sent_logits, sent_label)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        total_loss += loss.item()
        cat_preds += cat_logits.argmax(dim=1).cpu().tolist()
        cat_true += cat_label.cpu().tolist()
        sent_preds += sent_logits.argmax(dim=1).cpu().tolist()
        sent_true += sent_label.cpu().tolist()

    return {
        "loss": total_loss / len(loader),
        "cat_acc": accuracy_score(cat_true, cat_preds),
        "cat_f1": f1_score(cat_true, cat_preds, average="macro"),
        "sent_acc": accuracy_score(sent_true, sent_preds),
        "sent_f1": f1_score(sent_true, sent_preds, average="macro"),
    }

for epoch in range(EPOCHS):
    tr = run_epoch(train_loader, train_mode=True)
    va = run_epoch(val_loader, train_mode=False)
    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr['loss']:.3f} "
          f"| val cat_acc={va['cat_acc']:.2f} cat_f1={va['cat_f1']:.2f} "
          f"| val sent_acc={va['sent_acc']:.2f} sent_f1={va['sent_f1']:.2f}")


# %% CELL 5 — Evaluate on the HARD test set (the real signal)
def evaluate_hard(loader, df):
    model.eval()
    cat_preds, sent_preds = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            cat_logits, sent_logits = model(input_ids, attn)
            cat_preds += cat_logits.argmax(dim=1).cpu().tolist()
            sent_preds += sent_logits.argmax(dim=1).cpu().tolist()

    cat_true = df["cat_id"].tolist()
    sent_true = df["sent_id"].tolist()

    print("=== HARD TEST SET — Category ===")
    print(classification_report(cat_true, cat_preds, target_names=CATEGORY_LABELS, zero_division=0))
    print("=== HARD TEST SET — Sentiment ===")
    print(classification_report(sent_true, sent_preds, target_names=SENTIMENT_LABELS, zero_division=0))

    return cat_preds, sent_preds

cat_preds, sent_preds = evaluate_hard(hard_loader, hard_df)

# Show per-row predictions for a sanity-check / writeup screenshot
results_df = hard_df[["text", "category", "sentiment"]].copy()
results_df["predicted_category"] = [CATEGORY_LABELS[i] for i in cat_preds]
results_df["predicted_sentiment"] = [SENTIMENT_LABELS[i] for i in sent_preds]
results_df.to_csv("marbert_hard_test_predictions.csv", index=False)
print(results_df)


# %% CELL 6 — Save the fine-tuned model (for the Streamlit app later)
torch.save(model.state_dict(), "marbertv2_multitask.pt")
tokenizer.save_pretrained("marbertv2_tokenizer")
print("Saved marbertv2_multitask.pt + marbertv2_tokenizer/ — download these before your Colab session ends.")
