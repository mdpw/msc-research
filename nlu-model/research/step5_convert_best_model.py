"""
STEP 5: Convert best model (Model C) to TFLite for Android deployment

Pipeline: PyTorch → ONNX → TensorFlow SavedModel → TFLite

Prerequisites:
  nlu-model-env/Scripts/pip.exe install tf2onnx tensorflow onnx

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
import shutil
import subprocess
import sys
import numpy as np
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

BEST_MODEL_DIR  = 'models/model_c'
ONNX_PATH       = 'model_c.onnx'
TF_SAVED_MODEL  = 'model_c_tf'
OUTPUT_TFLITE   = 'hotel_mobilebert_v2.tflite'
OUTPUT_LABEL    = 'label_map_v2.json'
MAX_LENGTH      = 32


def check_prerequisites():
    missing = []
    for pkg in ['tf2onnx', 'tensorflow', 'onnx']:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}")
        print(f"Install with:")
        print(f"  nlu-model-env/Scripts/pip.exe install {' '.join(missing)}")
        sys.exit(1)


def run_command(cmd: list, step_name: str):
    """Run a shell command with error checking."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed.")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    if result.stdout:
        print(result.stdout)


def cleanup_temp_files():
    """Remove intermediate conversion files."""
    if os.path.exists(ONNX_PATH):
        os.remove(ONNX_PATH)
    if os.path.exists(TF_SAVED_MODEL):
        shutil.rmtree(TF_SAVED_MODEL)


# ── Prerequisites ─────────────────────────────────────────────────────────────
check_prerequisites()

if not os.path.exists(BEST_MODEL_DIR):
    print(f"ERROR: {BEST_MODEL_DIR} not found. Run step3_train_three_models.py first.")
    sys.exit(1)

# ── Step 1: Load PyTorch model ────────────────────────────────────────────────
print("="*60)
print("  STEP 5: PyTorch → ONNX → TFLite conversion")
print("="*60)
print(f"\nLoading model from {BEST_MODEL_DIR}...")

tokenizer = MobileBertTokenizer.from_pretrained(BEST_MODEL_DIR)
pt_model  = MobileBertForSequenceClassification.from_pretrained(BEST_MODEL_DIR)
pt_model.eval()

with open(f'{BEST_MODEL_DIR}/label_map.json') as f:
    label_map = json.load(f)

num_labels = len(label_map)
print(f"  Loaded: {num_labels} intents")

# ── Step 2: Export PyTorch → ONNX ────────────────────────────────────────────
print("\nExporting to ONNX...")
dummy = tokenizer(
    "i need some extra towels please",
    return_tensors='pt',
    max_length=MAX_LENGTH,
    padding='max_length',
    truncation=True,
)

torch.onnx.export(
    pt_model,
    (dummy['input_ids'], dummy['attention_mask'], dummy['token_type_ids']),
    ONNX_PATH,
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids':       {0: 'batch'},
        'attention_mask':  {0: 'batch'},
        'token_type_ids':  {0: 'batch'},
        'logits':          {0: 'batch'},
    },
    opset_version=13,
)
print(f"  Saved {ONNX_PATH}")

# ── Step 3: ONNX → TensorFlow SavedModel ─────────────────────────────────────
print("\nConverting ONNX → TensorFlow SavedModel...")
run_command(
    [sys.executable, '-m', 'tf2onnx.convert',
     '--onnx', ONNX_PATH,
     '--output', TF_SAVED_MODEL,
     '--saved-model'],
    'ONNX → TensorFlow SavedModel',
)
print(f"  Saved {TF_SAVED_MODEL}/")

# ── Step 4: TensorFlow SavedModel → TFLite ───────────────────────────────────
print("\nConverting TensorFlow SavedModel → TFLite...")
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL)

# Dynamic range quantisation: weights stored as INT8, activations as float32
# Reduces model size ~4× with minimal accuracy loss — no calibration dataset needed
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,    # needed for some MobileBERT ops
]

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

# Build index map by tensor name — do not assume fixed order
input_index = {d['name'].split('/')[-1].split(':')[0]: d['index'] for d in input_details}

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
print("  " + "-"*100)

correct = 0
for text, expected in test_utterances:
    enc = tokenizer(
        text,
        return_tensors='np',
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
    )

    # Set inputs by name to avoid index ordering issues
    for name in ['input_ids', 'attention_mask', 'token_type_ids']:
        # Try exact name match first, then partial match
        idx = input_index.get(name)
        if idx is None:
            for k, v in input_index.items():
                if name in k:
                    idx = v
                    break
        if idx is not None:
            interpreter.set_tensor(idx, enc[name].astype(np.int32))

    interpreter.invoke()
    logits    = interpreter.get_tensor(output_details[0]['index'])
    predicted = label_list[np.argmax(logits)]
    match     = "✓" if predicted == expected else "✗"
    if predicted == expected:
        correct += 1
    print(f"  {text:<45} {expected:<22} {predicted:<22} {match}")

print(f"\n  Quick test: {correct}/{len(test_utterances)} correct")

# ── Step 7: Cleanup intermediate files ───────────────────────────────────────
print("\nCleaning up intermediate files...")
cleanup_temp_files()
print("  Removed model_c.onnx and model_c_tf/")

# ── Deployment instructions ───────────────────────────────────────────────────
print(f"""
{'='*60}
  DEPLOYMENT INSTRUCTIONS
{'='*60}

1. Copy TFLite model to Android:
   FROM: nlu-model/{OUTPUT_TFLITE}
   TO:   android/app/src/main/assets/hotel_mobilebert.tflite

2. Copy label map to Android:
   FROM: nlu-model/{OUTPUT_LABEL}
   TO:   android/app/src/main/assets/label_map.json

3. Rebuild the Android app in Android Studio.

No code changes needed in Android — NLUService.kt
loads the model from assets automatically.

Model size : {size_mb:.1f} MB
Intents    : {num_labels}
{'='*60}
""")
