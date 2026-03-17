"""
STEP 5: Convert best model (Model C) to TFLite for Android deployment

Pipeline: PyTorch checkpoint → TF (via HuggingFace from_pt=True) → TFLite
No ONNX required. Uses TensorFlow's built-in converter directly.

Prerequisites (already installed from training):
  tensorflow, transformers

Run:
  cd nlu-model
  nlu-model-env/Scripts/python.exe research/step5_convert_best_model.py

Output:
  hotel_mobilebert_v2.tflite   — copy to android/app/src/main/assets/
  label_map_v2.json            — copy to android/app/src/main/assets/

After running:
  1. Copy hotel_mobilebert_v2.tflite →
       android/app/src/main/assets/hotel_mobilebert.tflite  (replace existing)
  2. Copy label_map_v2.json →
       android/app/src/main/assets/label_map.json           (replace existing)
  3. Rebuild Android app in Android Studio
"""

import os
import json
import sys
import numpy as np

BEST_MODEL_DIR = 'models/model_c'
OUTPUT_TFLITE  = 'hotel_mobilebert_v2.tflite'
OUTPUT_LABEL   = 'label_map_v2.json'
MAX_LENGTH     = 32

# ── Prerequisites check ───────────────────────────────────────────────────────
for pkg in ['tensorflow', 'transformers']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"ERROR: Missing package: {pkg}")
        print(f"Install: nlu-model-env/Scripts/pip.exe install {pkg}")
        sys.exit(1)

if not os.path.exists(BEST_MODEL_DIR):
    print(f"ERROR: {BEST_MODEL_DIR} not found. Run step3_train_three_models.py first.")
    sys.exit(1)

print("=" * 60)
print("  STEP 5: PyTorch -> TensorFlow -> TFLite")
print("=" * 60)

# ── Step 1: Load tokenizer and label map ─────────────────────────────────────
print(f"\nLoading tokenizer and label map from {BEST_MODEL_DIR}...")
from transformers import MobileBertTokenizer
tokenizer = MobileBertTokenizer.from_pretrained(BEST_MODEL_DIR)

with open(f'{BEST_MODEL_DIR}/label_map.json') as f:
    label_map = json.load(f)
num_labels = len(label_map)
print(f"  Intents: {num_labels}")

# ── Step 2: Load PyTorch model into TF using from_pt=True ────────────────────
# HuggingFace automatically converts PyTorch weights to TF format
print("\nLoading PyTorch weights into TF model (from_pt=True)...")
import tensorflow as tf
from transformers import TFMobileBertForSequenceClassification

tf_model = TFMobileBertForSequenceClassification.from_pretrained(
    BEST_MODEL_DIR,
    from_pt=True,
    num_labels=num_labels,
)
tf_model.trainable = False
print("  Loaded successfully")

# ── Step 3: Define concrete function with fixed input shapes ──────────────────
# TFLite requires concrete (fixed) input shapes — batch=1 for on-device inference
print("\nCreating concrete function for TFLite export...")

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, MAX_LENGTH], dtype=tf.int32, name='input_ids'),
    tf.TensorSpec(shape=[1, MAX_LENGTH], dtype=tf.int32, name='attention_mask'),
    tf.TensorSpec(shape=[1, MAX_LENGTH], dtype=tf.int32, name='token_type_ids'),
])
def serving_fn(input_ids, attention_mask, token_type_ids):
    outputs = tf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        training=False,
    )
    return outputs.logits

concrete_fn = serving_fn.get_concrete_function()
print("  Concrete function created")

# ── Step 4: Convert to TFLite with dynamic range quantisation ─────────────────
print("\nConverting to TFLite (dynamic range quantisation)...")
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_fn], tf_model
)
# Dynamic range quantisation: weights → INT8, activations → float32
# ~4× smaller model, minimal accuracy loss, no calibration data needed
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,   # needed for some MobileBERT ops
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()
with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

size_mb = os.path.getsize(OUTPUT_TFLITE) / (1024 * 1024)
print(f"  Saved {OUTPUT_TFLITE} ({size_mb:.1f} MB)")

# ── Step 5: Save label map ────────────────────────────────────────────────────
with open(OUTPUT_LABEL, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f"  Saved {OUTPUT_LABEL}")

# ── Step 6: Verify TFLite model with quick inference test ────────────────────
print("\nVerifying TFLite model with quick inference test...")
interpreter = tf.lite.Interpreter(model_path=OUTPUT_TFLITE)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map tensor names to indices safely
input_index = {}
for d in input_details:
    name = d['name']
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        if key in name:
            input_index[key] = d['index']
            break

print(f"  Input tensors mapped: {list(input_index.keys())}")

label_list = [label_map[str(i)] for i in range(num_labels)]

test_utterances = [
    ("i need extra towels please",          "towel_request"),
    ("can someone clean my room",           "room_cleaning"),
    ("the air conditioning is not working", "maintenance"),
    ("i would like to order some food",     "food_order"),
    ("please do not disturb",               "do_not_disturb"),
    ("can i get an extra blanket",          "blanket_request"),
    ("i need a wake up call at seven",      "wake_up_call"),
    ("the tap in the bathroom is leaking",  "maintenance"),
]

print(f"\n  {'Utterance':<45} {'Expected':<22} {'Predicted':<22} Match")
print("  " + "-" * 100)

correct = 0
for text, expected in test_utterances:
    enc = tokenizer(
        text,
        return_tensors='np',
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
    )
    for key in ['input_ids', 'attention_mask', 'token_type_ids']:
        if key in input_index:
            interpreter.set_tensor(input_index[key], enc[key].astype(np.int32))

    interpreter.invoke()
    logits    = interpreter.get_tensor(output_details[0]['index'])
    predicted = label_list[np.argmax(logits)]
    match     = "OK" if predicted == expected else "FAIL"
    if predicted == expected:
        correct += 1
    print(f"  {text:<45} {expected:<22} {predicted:<22} {match}")

print(f"\n  Quick test: {correct}/{len(test_utterances)} correct")

# ── Deployment instructions ───────────────────────────────────────────────────
print(f"""
{'=' * 60}
  DEPLOYMENT INSTRUCTIONS
{'=' * 60}

1. Copy TFLite model to Android:
   FROM: nlu-model/{OUTPUT_TFLITE}
   TO:   android/app/src/main/assets/hotel_mobilebert.tflite

2. Copy label map to Android:
   FROM: nlu-model/{OUTPUT_LABEL}
   TO:   android/app/src/main/assets/label_map.json

3. Rebuild the Android app in Android Studio.

No code changes needed — NLUService.kt loads from assets automatically.

Model size : {size_mb:.1f} MB
Intents    : {num_labels}
{'=' * 60}
""")
