"""
Optimized MobileBERT Training for Production-Ready Dataset (4,972 examples)
Fixed for: lowercase consistency, proper batch size, optimal hyperparameters
"""

import pandas as pd
import numpy as np
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
import json
import os

print("=" * 60)
print("MOBILEBERT TRAINING - PRODUCTION READY DATASET")
print("=" * 60)

# Configuration
MAX_LENGTH = 32  # Optimal for hotel voice commands (most are < 10 words)
BATCH_SIZE = 32  # Increased for larger dataset
EPOCHS = 8       # Optimal for 4,972 examples (not too many)
LEARNING_RATE = 3e-5

# Load dataset
print("\n📥 Loading production-ready dataset...")
df = pd.read_csv('hotel_intents_production_ready.csv')

print(f"Total samples: {len(df)}")
print(f"Number of intents: {df['intent'].nunique()}")

# Verify dataset is lowercase
uppercase_count = df['text'].str.contains(r'[A-Z]').sum()
if uppercase_count > 0:
    print(f"\n⚠️  WARNING: Found {uppercase_count} examples with uppercase!")
    print("   Converting all to lowercase...")
    df['text'] = df['text'].str.lower()
else:
    print("✅ All text is lowercase")

# Check for empty texts
empty_texts = df['text'].isna().sum() or (df['text'].str.strip() == '').sum()
if empty_texts > 0:
    print(f"⚠️  Removing {empty_texts} empty texts...")
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]

# Display intent distribution
print("\n📊 Intent Distribution:")
intent_counts = df['intent'].value_counts()
print(intent_counts.to_string())
print(f"\nMin examples per intent: {intent_counts.min()}")
print(f"Max examples per intent: {intent_counts.max()}")
print(f"Average per intent: {intent_counts.mean():.1f}")

# Encode labels
print("\n🏷️  Encoding labels...")
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['intent'])

# Save label mapping (for Android)
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"✅ Created label mapping for {len(label_encoder.classes_)} intents")

# Show label mapping
print("\n📋 Label Mapping:")
for idx, intent in label_map.items():
    count = len(df[df['intent'] == intent])
    print(f"  {idx}: {intent} ({count} examples)")

# Split data with stratification (maintains balance)
print("\n✂️  Splitting dataset...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.15,  # 15% validation (optimal for 4972 examples)
    random_state=42,
    stratify=df['label']  # Keeps same distribution in train/val
)

print(f"Training samples: {len(train_texts)} ({len(train_texts)/len(df)*100:.1f}%)")
print(f"Validation samples: {len(val_texts)} ({len(val_texts)/len(df)*100:.1f}%)")

# Verify no data leakage
train_set = set(train_texts)
val_set = set(val_texts)
overlap = train_set.intersection(val_set)
if overlap:
    print(f"⚠️  WARNING: {len(overlap)} texts appear in both train and val!")
else:
    print("✅ No data leakage between train/val sets")

# Initialize tokenizer (UNCASED for voice commands)
print("\n🔧 Initializing MobileBERT tokenizer...")
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

print(f"✅ Using UNCASED tokenizer (correct for voice)")
print(f"📏 Max length: {MAX_LENGTH} tokens")

# Sample tokenization check
print("\n🔍 Sample Tokenizations:")
sample_texts = [
    "could you arrange transportation",
    "bring me pillows",
    "emergency help",
    "turn off lights"
]
for text in sample_texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, max_length=MAX_LENGTH, truncation=True)
    print(f"  '{text}'")
    print(f"    → {tokens} ({len(token_ids)} tokens)")

# Tokenize datasets
print(f"\n📤 Tokenizing {len(train_texts)} training examples...")
train_encodings = tokenizer(
    train_texts, 
    truncation=True, 
    padding='max_length',
    max_length=MAX_LENGTH,
    return_attention_mask=True,
    return_tensors='pt'
)

print(f"📤 Tokenizing {len(val_texts)} validation examples...")
val_encodings = tokenizer(
    val_texts, 
    truncation=True, 
    padding='max_length',
    max_length=MAX_LENGTH,
    return_attention_mask=True,
    return_tensors='pt'
)

print("✅ Tokenization complete")

# Check for truncation
truncated_count = sum(1 for text in train_texts if len(tokenizer.tokenize(text)) > MAX_LENGTH)
if truncated_count > 0:
    print(f"⚠️  {truncated_count} training texts truncated (>{MAX_LENGTH} tokens)")
    print("   This is OK for hotel commands (most are short)")

# Create PyTorch Dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

print(f"\n✅ Created PyTorch datasets:")
print(f"   Training: {len(train_dataset)} examples")
print(f"   Validation: {len(val_dataset)} examples")

# Load pre-trained model
print(f"\n🤖 Loading MobileBERT base model...")
model = MobileBertForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    num_labels=len(label_encoder.classes_),
    ignore_mismatched_sizes=True
)

print(f"✅ Model loaded with {len(label_encoder.classes_)} output labels")

# Training configuration
print("\n⚙️  Training Configuration:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Max length: {MAX_LENGTH}")
print(f"   Total training steps: {len(train_dataset) // BATCH_SIZE * EPOCHS}")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_ratio=0.1,  # 10% warmup
    weight_decay=0.01,
    learning_rate=LEARNING_RATE,
    logging_dir='./logs',
    logging_steps=20,  # Log every 20 steps
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    save_total_limit=2,  # Only keep best 2 checkpoints
    fp16=False,  # Set to True if you have GPU with fp16 support
    seed=42,
)

# Define comprehensive metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = (predictions == labels).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for intent_id, intent_name in label_map.items():
        mask = labels == intent_id
        if mask.sum() > 0:
            class_acc = (predictions[mask] == labels[mask]).mean()
            per_class_acc[intent_name] = float(class_acc)
    
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class_acc
    }

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)
print(f"\nDataset: 4,972 examples (production-ready)")
print(f"Training on: {len(train_texts)} examples")
print(f"Validating on: {len(val_texts)} examples")
print(f"Expected duration: ~{EPOCHS * 2}-{EPOCHS * 3} minutes")
print("\n" + "=" * 60 + "\n")

# Train
train_result = trainer.train()

# Final evaluation
print("\n" + "=" * 60)
print("📊 FINAL EVALUATION")
print("=" * 60)

eval_results = trainer.evaluate()

print(f"\n✅ Overall Validation Accuracy: {eval_results['eval_accuracy']:.2%}")
print(f"✅ Validation Loss: {eval_results['eval_loss']:.4f}")

# Detailed per-class metrics
print("\n📊 Per-Intent Accuracy:")
per_class = eval_results.get('eval_per_class_accuracy', {})
if per_class:
    for intent, acc in sorted(per_class.items(), key=lambda x: x[1]):
        status = "✅" if acc > 0.90 else "⚠️" if acc > 0.75 else "❌"
        print(f"  {status} {intent}: {acc:.1%}")

# Get predictions for confusion matrix
print("\n🔍 Generating detailed classification report...")
predictions = trainer.predict(val_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(
    val_labels, 
    pred_labels, 
    target_names=label_encoder.classes_,
    digits=3
)
print(report)

# Save detailed report
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Confusion matrix (save for analysis)
conf_matrix = confusion_matrix(val_labels, pred_labels)
np.save('confusion_matrix.npy', conf_matrix)

# Save model
print("\n💾 Saving trained model...")
model.save_pretrained('./mobilebert_hotel_model')
tokenizer.save_pretrained('./mobilebert_hotel_model')

print("✅ Model saved to: ./mobilebert_hotel_model")

# Save comprehensive training info
training_info = {
    'dataset': 'hotel_intents_production_ready.csv',
    'total_examples': len(df),
    'train_examples': len(train_texts),
    'val_examples': len(val_texts),
    'num_intents': len(label_encoder.classes_),
    'max_length': MAX_LENGTH,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'final_accuracy': float(eval_results['eval_accuracy']),
    'final_loss': float(eval_results['eval_loss']),
    'model_type': 'mobilebert-uncased',
    'tokenizer': 'google/mobilebert-uncased',
    'per_intent_accuracy': per_class,
    'intent_distribution': intent_counts.to_dict()
}

with open('training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("✅ Training info saved to: training_info.json")

# Final summary
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

print(f"\n📊 Final Results:")
print(f"   Validation Accuracy: {eval_results['eval_accuracy']:.2%}")
print(f"   Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"   Model Size: ~25-30 MB (optimized for mobile)")

print(f"\n📁 Generated Files:")
print(f"   ✅ ./mobilebert_hotel_model/ (PyTorch model)")
print(f"   ✅ label_map.json (intent labels)")
print(f"   ✅ training_info.json (training metadata)")
print(f"   ✅ classification_report.txt (detailed metrics)")
print(f"   ✅ confusion_matrix.npy (for analysis)")

print("\n📋 Next Steps:")
print("1. Run: python convert_mobilebert_to_tflite_production.py")
print("2. Copy hotel_mobilebert.tflite to Android assets")
print("3. Copy vocab.json and label_map.json to Android assets")
print("4. Update Android app to use new model")
print("5. Test with both clean text AND Vosk output")

print("\n🎯 Expected Production Performance:")
print("   With clean text: 95-98% accuracy")
print("   With Vosk STT: 75-85% accuracy")
print("   With Whisper STT: 90-95% accuracy")

print("\n✨ Dataset Improvements:")
print(f"   Previous: 951 examples → misc_request defaults ❌")
print(f"   Current: 4,972 examples → accurate classification ✅")
print(f"   Includes: STT error variations (Vosk mistakes)")
print(f"   Balanced: 192-508 examples per intent")

if eval_results['eval_accuracy'] > 0.95:
    print("\n🎉 EXCELLENT! Ready for production deployment!")
elif eval_results['eval_accuracy'] > 0.90:
    print("\n✅ GOOD! Should work well in production")
else:
    print("\n⚠️  WARNING: Accuracy below 90% - check data quality")

print("\n" + "=" * 60)