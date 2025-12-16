"""
Quick Fix: Retrain MobileBERT with MAX_LENGTH = 32
This will make training and inference match perfectly
"""

import pandas as pd
import numpy as np
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
import json

print("=" * 50)
print("MobileBERT Retraining with MAX_LENGTH = 32")
print("=" * 50)

# Load dataset
print("\n📥 Loading dataset...")
df = pd.read_csv('hotel_guest_intents_dataset.csv')
print(f"Total samples: {len(df)}")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['intent'])

# Save label mapping
label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"\n✅ Found {len(label_encoder.classes_)} intents")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Initialize tokenizer
print("\n🔧 Initializing MobileBERT tokenizer...")
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# ✅ CRITICAL: Use MAX_LENGTH = 32 to match TFLite conversion
MAX_LENGTH = 32  # CHANGED FROM 48

print(f"📤 Tokenizing texts (max length: {MAX_LENGTH})...")

train_encodings = tokenizer(
    train_texts, 
    truncation=True, 
    padding='max_length',
    max_length=MAX_LENGTH,  # ✅ FIXED: 32 instead of 48
    return_attention_mask=True
)

val_encodings = tokenizer(
    val_texts, 
    truncation=True, 
    padding='max_length',
    max_length=MAX_LENGTH,  # ✅ FIXED: 32 instead of 48
    return_attention_mask=True
)

# Create dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# Load model
print(f"\n🤖 Loading MobileBERT model...")
model = MobileBertForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    num_labels=len(label_encoder.classes_),
    ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=3e-5,
    logging_dir='./logs',
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    save_total_limit=3,
    fp16=False,
)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n🚀 Starting training with MAX_LENGTH = 32...")
print("This ensures perfect match with TFLite model!")
print("-" * 50)

# Train
train_result = trainer.train()

# Evaluate
print("\n📊 Final evaluation...")
eval_results = trainer.evaluate()

print(f"\n✅ Final Validation Accuracy: {eval_results['eval_accuracy']:.2%}")
print(f"✅ Final Validation Loss: {eval_results['eval_loss']:.4f}")

# Save model
print("\n💾 Saving model...")
model.save_pretrained('./mobilebert_hotel_model')
tokenizer.save_pretrained('./mobilebert_hotel_model')

# Save training info
training_info = {
    'final_accuracy': float(eval_results['eval_accuracy']),
    'final_loss': float(eval_results['eval_loss']),
    'num_epochs': 15,
    'max_length': MAX_LENGTH,  # ✅ Now correctly 32
    'num_labels': len(label_encoder.classes_),
    'num_train_samples': len(train_texts),
    'num_val_samples': len(val_texts)
}

with open('training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\n" + "=" * 50)
print("✅ Retraining Complete!")
print("=" * 50)
print(f"\nModel saved to: ./mobilebert_hotel_model")
print(f"Final Validation Accuracy: {eval_results['eval_accuracy']:.2%}")
print(f"MAX_LENGTH: {MAX_LENGTH} (now matches TFLite!)")

print("\n📋 Next steps:")
print("1. Run: python convert_mobilebert_to_tflite.py")
print("2. Replace hotel_mobilebert.tflite in Android assets")
print("3. Test - should now work correctly!")