#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
import csv
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

# === CONFIG ===
model_name = "emilyalsentzer/Bio_ClinicalBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
max_length = 256
epochs = 5
top_K = 5  # top symptoms to select from subset

# === STEP 1: Full Symptom Lexicon ===
symptom_lexicon_full = [
    "fever", "cough", "headache", "nausea", "vomiting", "fatigue", "chest pain", "shortness of breath",
    "abdominal pain", "dizziness", "diarrhea", "constipation", "joint pain", "back pain", "depression", "anxiety",
    "rash", "itching", "seizure", "confusion", "palpitations", "insomnia", "loss of appetite", "urinary frequency",
    "chills", "syncope", "sore throat", "swelling", "pain", "malaise", "cramps", "numbness", "tingling",
    "blurry vision", "weakness", "edema", "hallucinations", "bleeding", "difficulty breathing", "burning"
]

# === STEP 2: Load Notes & Sample 10% ===
notes_full = pd.read_csv(
    "NOTEEVENTS_random.csv",
    usecols=["TEXT"],
    quoting=csv.QUOTE_NONE,
    on_bad_lines="skip"
).dropna()

notes = notes_full.sample(frac=0.1, random_state=1000).copy()
notes["TEXT"] = notes["TEXT"].str.lower().str.slice(0, 1000)

# === STEP 2b: Extract symptoms ===
def extract_label(text):
    """Return first matching symptom (single-label)."""
    matches = [s for s in symptom_lexicon_full if re.search(rf"\b{re.escape(s)}\b", text)]
    return matches[0] if matches else None

notes["symptom"] = notes["TEXT"].apply(extract_label)
notes = notes.dropna(subset=["symptom"])

# === STEP 3: Select Top-K Symptoms from 10% subset ===
top_symptoms = [s for s, _ in Counter(notes["symptom"]).most_common(top_K)]
if not top_symptoms:
    raise ValueError("❌ No symptoms matched the lexicon in the dataset.")
symptom_lexicon = top_symptoms
print("✅ Top symptoms from 10% subset:", symptom_lexicon)

# Optional: show counts of top symptoms
top_counts = Counter(notes["symptom"])
for s in symptom_lexicon:
    print(f"  {s}: {top_counts[s]} occurrences")

notes = notes[notes["symptom"].isin(symptom_lexicon)]
symptom2id = {s: i for i, s in enumerate(symptom_lexicon)}
id2symptom = {i: s for s, i in symptom2id.items()}
notes["label"] = notes["symptom"].map(symptom2id)

# === STEP 4: Balance Dataset ===
balanced = []
counts = [sum(notes["label"] == i) for i in range(len(symptom_lexicon))]
min_count = min(counts)
print("Class distribution before balancing:", counts)

for i in range(len(symptom_lexicon)):
    subset = notes[notes["label"] == i].sample(n=min_count, random_state=42)
    balanced.append(subset)

balanced_df = pd.concat(balanced).reset_index(drop=True)
balanced_df = balanced_df.rename(columns={"TEXT": "text"})
print(f"✅ Balanced dataset with {min_count} samples per class, total = {len(balanced_df)}")

# === STEP 5: Prepare HuggingFace Datasets ===
train_df, test_df = train_test_split(balanced_df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

# === STEP 6: Evaluation Function ===
def evaluate_model(model, loader, method_name=None, epoch=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            labels = batch["label"].cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(np.eye(len(symptom_lexicon))[all_labels], np.array(all_probs), average="macro", multi_class="ovr")
    except:
        auroc = 0.0

    if method_name and epoch is not None:
        csv_file = f"{method_name}_metrics.csv"
        write_header = not os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Epoch", "Accuracy", "Precision", "Recall", "F1", "AUROC"])
            writer.writerow([epoch, acc, precision, recall, f1, auroc])

        per_class_metrics = precision_recall_fscore_support(all_labels, all_preds, labels=list(range(len(symptom_lexicon))), zero_division=0)
        try:
            per_class_auroc = [roc_auc_score((np.array(all_labels) == i).astype(int), np.array(all_probs)[:, i]) for i in range(len(symptom_lexicon))]
        except:
            per_class_auroc = [0.0] * len(symptom_lexicon)

        per_symptom_file = f"{method_name}_per_symptom_metrics.csv"
        write_header_ps = not os.path.exists(per_symptom_file)
        with open(per_symptom_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if write_header_ps:
                writer.writerow(["Epoch", "Symptom", "Precision", "Recall", "F1", "AUROC"])
            for i in range(len(symptom_lexicon)):
                writer.writerow([
                    epoch,
                    id2symptom[i],
                    per_class_metrics[0][i],
                    per_class_metrics[1][i],
                    per_class_metrics[2][i],
                    per_class_auroc[i]
                ])
    return acc, precision, recall, f1, auroc

# === STEP 7: Distillation Training with Activation Switching ===
class GReLU(nn.Module):
    """Convex Generalized ReLU"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return torch.maximum(x, self.a * x)

def replace_activation(model, activation="relu"):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            if activation == "relu":
                setattr(model, name, nn.ReLU())
            elif activation == "grelu":
                setattr(model, name, GReLU())
        else:
            replace_activation(module, activation)
    return model

def distillation_train(strategy="convex"):
    teacher = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(symptom_lexicon)).to(device)
    teacher.eval()
    student = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(symptom_lexicon)).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))

    T = 4.0
    for epoch in range(1, epochs + 1):
        if strategy == "convex":
            student = replace_activation(student, "grelu")
            act_stage = "Convex (GReLU)"
        elif strategy == "nonconvex":
            student = replace_activation(student, "relu")
            act_stage = "Nonconvex (ReLU)"
        elif strategy == "multistage":
            if epoch <= 2:
                student = replace_activation(student, "grelu")
                act_stage = "Convex (GReLU)"
            else:
                student = replace_activation(student, "relu")
                act_stage = "Nonconvex (ReLU)"
                optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
                scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=epochs * len(train_loader))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        student.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            ce_loss = F.cross_entropy(student_logits, batch["label"].long())
            kd_loss = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                               F.softmax(teacher_logits / T, dim=-1),
                               reduction="batchmean") * (T*T)

            loss = 0.5 * ce_loss + 0.5 * kd_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        acc, prec, rec, f1, auroc = evaluate_model(student, eval_loader, strategy, epoch)
        print(f"{strategy} | Epoch {epoch}/{epochs} [{act_stage}] - "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"- Acc: {acc:.4f} P: {prec:.4f} R: {rec:.4f} F1: {f1:.4f} AUROC: {auroc:.4f}")




# === STEP 7: Run All Three Strategies ===
def run_all():
    distillation_train("convex")
    distillation_train("nonconvex")
    distillation_train("multistage")



if __name__ == "__main__":
    run_all()


# 

# In[ ]:





# 

# In[ ]:




