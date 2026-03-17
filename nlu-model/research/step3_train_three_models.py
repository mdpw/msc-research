"""
STEP 3: Train 3 MobileBERT model variants

  Model A (baseline)  — trained on clean text only
  Model B (vosk-only) — trained on Vosk-transcribed text only
  Model C (mixed)     — trained on clean + Vosk-transcribed text

Run:
  cd nlu-model
  nlu-model-env/Scripts/python.exe research/step3_train_three_models.py

Output:
  models/model_a/                    — saved model, tokenizer, label_map.json
  models/model_b/                    — saved model, tokenizer, label_map.json
  models/model_c/                    — saved model, tokenizer, label_map.json
  models/test_set.csv                — shared held-out test set (step4 uses this)
  models/logs/model_X_history.csv    — epoch-by-epoch loss & accuracy curves
  models/logs/model_X_per_intent.csv — per-intent precision / recall / F1
  models/logs/model_X_confusion.csv  — confusion matrix
  training_results.json              — summary metrics for all 3 models
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import (
    MobileBertTokenizer,
    MobileBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LENGTH          = 32
BATCH_SIZE          = 16      # CPU-friendly
EPOCHS              = 5       # Early stopping cuts this short if needed
LEARNING_RATE       = 3e-5
WEIGHT_DECAY        = 0.01    # L2 regularisation — prevents overfitting
WARMUP_RATIO        = 0.1     # Gradual LR warmup — transformer best practice
SEED                = 42
EARLY_STOP_PATIENCE = 2       # Stop if val F1 doesn't improve for 2 epochs

# Model A — clean dataset (new_hotel_dataset.csv, 10,080 records)
# Model B — Vosk-noisy text only
# Model C — clean + Vosk mixed
DATASETS = {
    'model_a': 'new_hotel_dataset.csv',
    'model_b': 'vosk_only_dataset.csv',
    'model_c': 'paired_dataset.csv',
}

os.makedirs('models/logs', exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class HotelDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'f1_macro': float(f1_score(labels, preds, average='macro',    zero_division=0)),
        'f1_weighted': float(f1_score(labels, preds, average='weighted', zero_division=0)),
    }


def save_training_history(trainer, name: str):
    """Save epoch-by-epoch loss and accuracy curves to CSV."""
    history = trainer.state.log_history

    rows = []
    epoch_data = {}
    for entry in history:
        epoch = entry.get('epoch')
        if epoch is None:
            continue
        epoch = round(epoch)
        if epoch not in epoch_data:
            epoch_data[epoch] = {'epoch': epoch}
        epoch_data[epoch].update(entry)

    for epoch in sorted(epoch_data):
        e = epoch_data[epoch]
        rows.append({
            'epoch':        epoch,
            'train_loss':   e.get('loss',          ''),
            'val_loss':     e.get('eval_loss',      ''),
            'val_accuracy': e.get('eval_accuracy',  ''),
            'val_f1_macro': e.get('eval_f1_macro',  ''),
        })

    history_df = pd.DataFrame(rows)
    path = f'models/logs/{name}_history.csv'
    history_df.to_csv(path, index=False)
    print(f"  Training history saved → {path}")
    return history_df


def save_per_intent_report(preds, labels, intent_names, name: str):
    """Save per-intent precision, recall, F1, support to CSV."""
    report = classification_report(
        labels, preds,
        target_names=intent_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for intent in intent_names:
        r = report.get(intent, {})
        rows.append({
            'intent':    intent,
            'precision': round(r.get('precision', 0), 4),
            'recall':    round(r.get('recall',    0), 4),
            'f1_score':  round(r.get('f1-score',  0), 4),
            'support':   int(r.get('support',     0)),
        })
    per_intent_df = pd.DataFrame(rows)
    path = f'models/logs/{name}_per_intent.csv'
    per_intent_df.to_csv(path, index=False)

    # Print table
    print(f"\n  Per-intent results ({name.upper()}):")
    print(f"  {'Intent':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'-'*63}")
    for _, row in per_intent_df.iterrows():
        marker = ' ✗' if row['f1_score'] < 0.80 else ''
        print(f"  {row['intent']:<25} {row['precision']:>10.4f} {row['recall']:>8.4f} "
              f"{row['f1_score']:>8.4f} {row['support']:>8}{marker}")
    print(f"  Saved → {path}")
    return per_intent_df


def save_confusion_matrix(preds, labels, intent_names, name: str):
    """Save confusion matrix to CSV."""
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(cm, index=intent_names, columns=intent_names)
    path = f'models/logs/{name}_confusion.csv'
    cm_df.to_csv(path)
    print(f"  Confusion matrix saved → {path}")
    return cm_df


def get_predictions(trainer, dataset):
    """Run inference and return (predictions, true_labels)."""
    output = trainer.predict(dataset)
    preds  = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    return preds, labels


def train_model(name: str, csv_path: str, label_encoder: LabelEncoder,
                tokenizer, test_texts: list, test_labels: list) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training {name.upper()}")
    print(f"  Dataset : {csv_path}")
    print(f"{'='*60}")

    # ── Load and validate ─────────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"  model_a needs new_hotel_dataset.csv\n"
            f"  model_b and model_c need step2_generate_vosk_noise.py to be run first."
        )

    df = pd.read_csv(csv_path)
    df = df[['text', 'intent']].dropna()
    df['text']  = df['text'].str.lower().str.strip()
    df['label'] = label_encoder.transform(df['intent'])
    print(f"  Records : {len(df)} | Intents: {df['intent'].nunique()}")

    # ── Stratified train / validation split ───────────────────────────────────
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.15,
        random_state=SEED,
        stratify=df['label'].tolist(),
    )
    print(f"  Train   : {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # ── Tokenise (padding=max_length for consistent tensor sizes → TFLite) ────
    train_enc = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH)
    val_enc   = tokenizer(val_texts,   truncation=True, padding='max_length', max_length=MAX_LENGTH)
    test_enc  = tokenizer(test_texts,  truncation=True, padding='max_length', max_length=MAX_LENGTH)

    train_ds = HotelDataset(train_enc, train_labels)
    val_ds   = HotelDataset(val_enc,   val_labels)
    test_ds  = HotelDataset(test_enc,  test_labels)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MobileBertForSequenceClassification.from_pretrained(
        'google/mobilebert-uncased',
        num_labels=len(label_encoder.classes_),
    )

    # ── Training arguments ────────────────────────────────────────────────────
    output_dir = f'models/{name}'
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',           # epoch-level logs for training curves
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        save_total_limit=1,
        seed=SEED,
        report_to='none',
        dataloader_num_workers=0,           # required on Windows
        dataloader_pin_memory=False,        # CPU training
        fp16=False,                         # CPU does not support fp16
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save training curves ──────────────────────────────────────────────────
    save_training_history(trainer, name)

    # ── Evaluate on held-out TEST set ─────────────────────────────────────────
    val_metrics  = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    print(f"\n  {name.upper()} — Validation:")
    print(f"     Accuracy : {val_metrics['eval_accuracy']*100:.2f}%  "
          f"F1 Macro: {val_metrics['eval_f1_macro']:.4f}")
    print(f"  {name.upper()} — Test (thesis number):")
    print(f"     Accuracy : {test_metrics['eval_accuracy']*100:.2f}%  "
          f"F1 Macro: {test_metrics['eval_f1_macro']:.4f}")

    # ── Per-intent breakdown and confusion matrix ─────────────────────────────
    test_preds, test_true = get_predictions(trainer, test_ds)
    intent_names = list(label_encoder.classes_)
    save_per_intent_report(test_preds, test_true, intent_names, name)
    save_confusion_matrix(test_preds, test_true, intent_names, name)

    # ── Save model + tokenizer + label map ───────────────────────────────────
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f'{output_dir}/label_map.json', 'w') as f:
        json.dump({str(i): l for i, l in enumerate(label_encoder.classes_)}, f, indent=2)
    print(f"  Model saved → {output_dir}/")

    return {
        'model':           name,
        'dataset':         csv_path,
        'train_size':      len(train_texts),
        'val_size':        len(val_texts),
        'test_size':       len(test_texts),
        'val_accuracy':    val_metrics['eval_accuracy'],
        'val_f1_macro':    val_metrics['eval_f1_macro'],
        'val_f1_weighted': val_metrics['eval_f1_weighted'],
        'test_accuracy':   test_metrics['eval_accuracy'],
        'test_f1_macro':   test_metrics['eval_f1_macro'],
        'test_f1_weighted':test_metrics['eval_f1_weighted'],
    }


# ── Build shared label encoder from all 18 intents ───────────────────────────
print("Building label encoder from all intents...")
all_intents = set()
for csv_path in DATASETS.values():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        all_intents.update(df['intent'].unique())

if len(all_intents) == 0:
    raise RuntimeError("No datasets found. Run step2_generate_vosk_noise.py first.")

label_encoder = LabelEncoder()
label_encoder.fit(sorted(all_intents))
print(f"  {len(label_encoder.classes_)} intents: {list(label_encoder.classes_)}")

tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# ── Create shared held-out test set from vosk_transcriptions.csv ─────────────
# Saves both clean_text and vosk_text so step4_evaluate.py can test both.
# All 3 models are evaluated on identical sentences — fair comparison.
print("\nPreparing shared test set from vosk_transcriptions.csv...")
trans_df = pd.read_csv('vosk_transcriptions.csv')
trans_df = trans_df[['clean_text', 'vosk_text', 'intent']].dropna()
trans_df['clean_text'] = trans_df['clean_text'].str.lower().str.strip()
trans_df['vosk_text']  = trans_df['vosk_text'].str.lower().str.strip()
trans_df['label']      = label_encoder.transform(trans_df['intent'])

_, test_df = train_test_split(
    trans_df,
    test_size=0.20,
    random_state=SEED,
    stratify=trans_df['label'],
)

# Training and evaluation use vosk_text as test input (real-world STT output)
test_texts  = test_df['vosk_text'].tolist()
test_labels = test_df['label'].tolist()
test_df.to_csv('models/test_set.csv', index=False)
print(f"  Shared test set: {len(test_df)} sentences (clean + vosk columns)")
print(f"  Saved → models/test_set.csv")

# ── Train all 3 models ────────────────────────────────────────────────────────
all_results = []
for name, csv_path in DATASETS.items():
    result = train_model(name, csv_path, label_encoder, tokenizer, test_texts, test_labels)
    all_results.append(result)

# ── Final comparison table ────────────────────────────────────────────────────
print("\n" + "="*75)
print("  FINAL RESULTS — all models evaluated on shared Vosk test set")
print("="*75)
print(f"{'Model':<12} {'Train':>8} {'Val Acc':>10} {'Test Acc':>10} "
      f"{'Test F1 Mac':>12} {'Test F1 Wgt':>12}")
print("-"*75)
for r in all_results:
    print(f"{r['model']:<12} {r['train_size']:>8} "
          f"{r['val_accuracy']*100:>9.2f}% "
          f"{r['test_accuracy']*100:>9.2f}% "
          f"{r['test_f1_macro']:>12.4f} "
          f"{r['test_f1_weighted']:>12.4f}")

best = max(all_results, key=lambda x: x['test_f1_macro'])
print(f"\n  Best model : {best['model'].upper()}  "
      f"(Test Accuracy: {best['test_accuracy']*100:.2f}%  "
      f"F1 Macro: {best['test_f1_macro']:.4f})")
print(f"  Deploy     : models/{best['model']}/ → use in step5_convert_best_model.py")

# ── Accuracy gap — core research finding ─────────────────────────────────────
results_map = {r['model']: r for r in all_results}
if 'model_a' in results_map and 'model_c' in results_map:
    acc_gap = (results_map['model_c']['test_accuracy'] -
               results_map['model_a']['test_accuracy']) * 100
    f1_gap  = (results_map['model_c']['test_f1_macro'] -
               results_map['model_a']['test_f1_macro'])
    print(f"\n  ── Research Finding ──────────────────────────────────────")
    print(f"  Accuracy gap  (Model C vs A): {acc_gap:+.2f} percentage points")
    print(f"  F1 Macro gap  (Model C vs A): {f1_gap:+.4f}")
    print(f"  Noise-aware training (Model C) vs clean baseline (Model A)")
    print(f"  ─────────────────────────────────────────────────────────")

# ── Save summary results ──────────────────────────────────────────────────────
with open('training_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll outputs saved:")
print(f"  training_results.json            — summary metrics")
print(f"  models/test_set.csv              — shared test set")
print(f"  models/logs/model_X_history.csv  — training curves (loss/accuracy per epoch)")
print(f"  models/logs/model_X_per_intent.csv — per-intent precision/recall/F1")
print(f"  models/logs/model_X_confusion.csv  — confusion matrix")
print(f"\nNext: run step4_evaluate.py")
