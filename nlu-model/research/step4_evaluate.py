"""
STEP 4: Evaluate all 3 models and produce thesis-ready results

Produces the core research finding:
  Test 1 — Model A on clean text     → standard benchmark (what papers report)
  Test 2 — Model A on Vosk output    → real-world accuracy (THE GAP)
  Test 3 — Model B on Vosk output    → vosk-only training performance
  Test 4 — Model C on Vosk output    → noise-aware training (your fix)

Run:
  cd nlu-model
  nlu-model-env/Scripts/python.exe research/step4_evaluate.py

Output:
  evaluation_results.json        — all metrics
  confusion_matrices/            — PNG heatmaps for thesis figures
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

os.makedirs('confusion_matrices', exist_ok=True)

# ── Load the shared test set saved by step3 ──────────────────────────────────
# IMPORTANT: must use the same test set step3 used — not a new split
print("Loading shared test set from models/test_set.csv...")
if not os.path.exists('models/test_set.csv'):
    raise FileNotFoundError(
        "models/test_set.csv not found.\n"
        "Run step3_train_three_models.py first."
    )

test_df     = pd.read_csv('models/test_set.csv')
clean_texts = test_df['clean_text'].tolist()
vosk_texts  = test_df['vosk_text'].tolist()
true_labels = test_df['intent'].tolist()
print(f"  Test set: {len(test_df)} sentences | {test_df['intent'].nunique()} intents")

# ── Load label map ────────────────────────────────────────────────────────────
with open('models/model_a/label_map.json') as f:
    label_map = json.load(f)
label_map   = {int(k): v for k, v in label_map.items()}
intent_names = [label_map[i] for i in range(len(label_map))]

le = LabelEncoder()
le.fit(intent_names)
true_encoded = le.transform(true_labels)

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model_dir: str, texts: list) -> np.ndarray:
    """Batch inference — returns integer prediction array."""
    tokenizer = MobileBertTokenizer.from_pretrained(model_dir)
    model     = MobileBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    all_preds = []
    batch_size = 16  # consistent with training batch size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding='max_length',   # matches training tokenisation
            max_length=32,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())

    return np.array(all_preds)


def evaluate(model_dir: str, texts: list, model_name: str, test_type: str) -> dict:
    print(f"  Evaluating {model_name} on {test_type}...")
    preds = predict(model_dir, texts)

    acc          = accuracy_score(true_encoded, preds)
    f1_mac       = f1_score(true_encoded, preds, average='macro',    zero_division=0)
    f1_wgt       = f1_score(true_encoded, preds, average='weighted', zero_division=0)
    pred_labels  = le.inverse_transform(preds).tolist()

    return {
        'model':        model_name,
        'test_type':    test_type,
        'accuracy':     round(acc,    4),
        'f1_macro':     round(f1_mac, 4),
        'f1_weighted':  round(f1_wgt, 4),
        'predictions':  pred_labels,
    }


# ── Run 4 evaluations ─────────────────────────────────────────────────────────
print("\nRunning evaluations...")
results = [
    evaluate('models/model_a', clean_texts, 'Model A (clean-trained)',  'Clean Text'),
    evaluate('models/model_a', vosk_texts,  'Model A (clean-trained)',  'Vosk Output'),
    evaluate('models/model_b', vosk_texts,  'Model B (vosk-trained)',   'Vosk Output'),
    evaluate('models/model_c', vosk_texts,  'Model C (mixed-trained)',  'Vosk Output'),
]

# ── Thesis results table ──────────────────────────────────────────────────────
print("\n" + "="*80)
print("  CORE RESEARCH FINDING")
print("="*80)
print(f"{'Model':<30} {'Test Input':<15} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Wtd':>10}")
print("-"*80)

flags = [
    "← baseline (what papers report)",
    "← THE GAP  (real-world drop)",
    "",
    "← your fix (noise-aware training)",
]
for r, flag in zip(results, flags):
    print(f"{r['model']:<30} {r['test_type']:<15} "
          f"{r['accuracy']*100:>9.2f}% "
          f"{r['f1_macro']:>10.4f} "
          f"{r['f1_weighted']:>10.4f}  {flag}")

# ── Key gap numbers ───────────────────────────────────────────────────────────
baseline = results[0]['accuracy']   # Model A on clean
gap_acc  = results[1]['accuracy']   # Model A on vosk
fixed    = results[3]['accuracy']   # Model C on vosk

drop        = baseline - gap_acc
improvement = fixed    - gap_acc
recovery    = (improvement / drop * 100) if drop > 0 else 0.0

print(f"\n  Accuracy drop  (clean → Vosk pipeline):      -{drop*100:.2f}%")
print(f"  Accuracy gain  (noise-aware training fix):   +{improvement*100:.2f}%")
print(f"  Gap recovery   (how much of the drop fixed):  {recovery:.1f}%")

# ── Per-intent breakdown: Model A clean vs Model C vosk ──────────────────────
print("\n\n  PER-INTENT BREAKDOWN  (Model A clean text  vs  Model C Vosk output)")
print(f"  {'Intent':<25} {'A-Clean':>10} {'C-Vosk':>10} {'Diff':>8}")
print("  " + "-"*57)

def per_intent_acc(predictions: list) -> dict:
    df = pd.DataFrame({'pred': predictions, 'true': true_labels})
    return {
        intent: (df[df['true'] == intent]['pred'] == intent).mean()
        for intent in df['true'].unique()
    }

a_clean_acc = per_intent_acc(results[0]['predictions'])
c_vosk_acc  = per_intent_acc(results[3]['predictions'])

for intent in sorted(a_clean_acc):
    a    = a_clean_acc.get(intent, 0)
    c    = c_vosk_acc.get(intent, 0)
    diff = c - a
    arrow = "↑" if diff > 0.005 else ("↓" if diff < -0.005 else "=")
    print(f"  {intent:<25} {a*100:>9.1f}% {c*100:>9.1f}% {diff*100:>+7.1f}%  {arrow}")

# ── Per-intent classification report for thesis appendix ─────────────────────
print("\n\nDetailed classification report — Model C (Vosk output):")
c_preds_enc = le.transform(results[3]['predictions'])
print(classification_report(
    true_encoded, c_preds_enc,
    target_names=intent_names,
    zero_division=0,
))

# ── Confusion matrix plots ────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    def save_confusion_matrix_plot(predictions: list, title: str, filename: str):
        pred_enc = le.transform(predictions)
        cm       = confusion_matrix(true_encoded, pred_enc)
        cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=intent_names, yticklabels=intent_names,
        )
        plt.title(title, fontsize=13)
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = f'confusion_matrices/{filename}.png'
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    print("\nGenerating confusion matrix plots...")
    save_confusion_matrix_plot(
        results[0]['predictions'],
        'Model A — Clean-Trained, Tested on Clean Text (Baseline)',
        'model_a_clean',
    )
    save_confusion_matrix_plot(
        results[1]['predictions'],
        'Model A — Clean-Trained, Tested on Vosk Output (THE GAP)',
        'model_a_vosk_gap',
    )
    save_confusion_matrix_plot(
        results[3]['predictions'],
        'Model C — Mixed-Trained, Tested on Vosk Output (Noise-Aware Fix)',
        'model_c_mixed',
    )

except ImportError:
    print("\nmatplotlib/seaborn not installed — skipping confusion matrix plots.")
    print("Install with: pip install matplotlib seaborn")

# ── Save all results ──────────────────────────────────────────────────────────
output = {
    'test_set_size': len(test_df),
    'summary': [
        {k: v for k, v in r.items() if k != 'predictions'}
        for r in results
    ],
    'research_finding': {
        'accuracy_drop_clean_to_vosk':        round(drop,        4),
        'accuracy_gain_noise_aware_training':  round(improvement, 4),
        'gap_recovery_percent':                round(recovery,    1),
    },
}
with open('evaluation_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved evaluation_results.json")
print(f"Next: run step5_convert_best_model.py")
