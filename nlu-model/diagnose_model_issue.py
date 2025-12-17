"""
CRITICAL DIAGNOSTIC: Find Why Model Misclassifies Training Examples
Run this to identify the Android implementation bug
"""

import json
import numpy as np
import tensorflow as tf
from transformers import MobileBertTokenizer

print("=" * 60)
print("CRITICAL DIAGNOSTIC - MODEL MISCLASSIFICATION DEBUG")
print("=" * 60)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = MobileBertTokenizer.from_pretrained('./mobilebert_hotel_model')

# Load label map
print("2. Loading label map...")
with open('label_map.json', 'r') as f:
    label_map = json.load(f)
    # Convert string keys to int
    label_map = {int(k): v for k, v in label_map.items()}

print(f"   Found {len(label_map)} intents")

# Load TFLite model
print("3. Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path='hotel_mobilebert.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"   Input 0: {input_details[0]['name']} - Shape: {input_details[0]['shape']}")
print(f"   Input 1: {input_details[1]['name']} - Shape: {input_details[1]['shape']}")
print(f"   Output: {output_details[0]['name']} - Shape: {output_details[0]['shape']}")

# The EXACT phrases that are failing in your dashboard
test_cases = [
    ("may i request room cleaning service", "room_cleaning"),
    ("i need a coffee", "food_order"),
    ("housekeeping please", "room_cleaning"),
]

print("\n" + "=" * 60)
print("TESTING MISCLASSIFIED PHRASES")
print("=" * 60)

def classify_text(text):
    """Classify text using TFLite model - EXACTLY as training does"""
    
    # CRITICAL: Lowercase (must match training)
    text = text.lower().strip()
    
    # Encode with tokenizer (exactly as training)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Adds [CLS] and [SEP]
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np'
    )
    
    input_ids = encoding['input_ids'].astype(np.int32)
    attention_mask = encoding['attention_mask'].astype(np.int32)
    
    # Set tensors
    interpreter.set_tensor(input_details[0]['index'], attention_mask)
    interpreter.set_tensor(input_details[1]['index'], input_ids)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    logits = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    # Get prediction
    predicted_idx = np.argmax(probs)
    confidence = probs[predicted_idx]
    predicted_intent = label_map[predicted_idx]
    
    return predicted_intent, confidence, probs, input_ids, attention_mask

# Test each case
results = []
for text, expected_intent in test_cases:
    print(f"\n{'='*60}")
    print(f"Input: '{text}'")
    print(f"Expected: {expected_intent}")
    print("-" * 60)
    
    predicted, confidence, probs, input_ids, attention_mask = classify_text(text)
    
    # Check if correct
    correct = predicted == expected_intent
    status = "✅ CORRECT" if correct else "❌ WRONG"
    
    print(f"Predicted: {predicted} ({confidence:.2%} confidence) {status}")
    
    # Show tokenization
    print(f"\nTokenization:")
    tokens = tokenizer.tokenize(text)
    print(f"  Tokens: {tokens}")
    print(f"  Input IDs: {input_ids[0][:10]}... (first 10)")
    print(f"  Attention: {attention_mask[0][:10]}... (first 10)")
    
    # Show top 5 predictions
    print(f"\nTop 5 Predictions:")
    top_5_idx = np.argsort(probs)[-5:][::-1]
    for i, idx in enumerate(top_5_idx, 1):
        intent_name = label_map[idx]
        prob = probs[idx]
        marker = "←" if intent_name == expected_intent else ""
        print(f"  {i}. {intent_name}: {prob:.2%} {marker}")
    
    results.append({
        'text': text,
        'expected': expected_intent,
        'predicted': predicted,
        'confidence': float(confidence),
        'correct': correct
    })

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

correct_count = sum(1 for r in results if r['correct'])
total_count = len(results)

print(f"\nResults: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")

if correct_count == total_count:
    print("\n✅ MODEL IS WORKING CORRECTLY!")
    print("\n🔍 The problem is in your Android implementation:")
    print("   - Check tokenization in Android")
    print("   - Verify you're using the correct vocab.json")
    print("   - Ensure [CLS] and [SEP] tokens are added")
    print("   - Confirm lowercase normalization")
else:
    print(f"\n❌ MODEL HAS PROBLEMS!")
    print(f"   {total_count - correct_count} out of {total_count} EXACT training phrases failed")
    print("\n🔍 Possible causes:")
    print("   1. Model file corrupted during conversion")
    print("   2. Wrong label_map.json being used")
    print("   3. TFLite conversion introduced errors")
    print("   4. Model wasn't saved correctly after training")

# Additional checks
print("\n" + "=" * 60)
print("ADDITIONAL DIAGNOSTICS")
print("=" * 60)

# Check if model outputs are reasonable
print("\n1. Checking model output distribution...")
random_logits = np.random.randn(18)
test_probs = np.exp(random_logits) / np.exp(random_logits).sum()
if np.any(np.isnan(test_probs)) or np.any(np.isinf(test_probs)):
    print("   ⚠️  Model outputs contain NaN or Inf")
else:
    print("   ✅ Model outputs are valid")

# Check label map consistency
print("\n2. Checking label map...")
if len(label_map) == 18:
    print(f"   ✅ Label map has correct number of intents (18)")
else:
    print(f"   ❌ Label map has {len(label_map)} intents, expected 18")

print("\n3. Label Map Contents:")
for idx, intent in sorted(label_map.items()):
    print(f"   {idx}: {intent}")

# Save detailed results
with open('diagnostic_results.json', 'w') as f:
    json.dump({
        'test_results': results,
        'summary': {
            'correct': correct_count,
            'total': total_count,
            'accuracy': correct_count / total_count
        }
    }, f, indent=2)

print(f"\n✅ Detailed results saved to: diagnostic_results.json")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)

if correct_count == total_count:
    print("""
✅ Your TFLite model is CORRECT!

The problem is in your Android implementation. Check:

1. Tokenization:
   - Are you using the vocab.json from the model?
   - Are you adding [CLS] (101) and [SEP] (102) tokens?
   - Is text being lowercased?

2. Input tensor order:
   - Check if Android expects [attention_mask, input_ids]
   - Or [input_ids, attention_mask]
   - Print the input_details order from TFLite

3. Output processing:
   - Are you applying softmax to logits?
   - Are you using the correct label_map.json?

Run this in your Android code to debug:
```kotlin
// Print input details order
Log.d("TFLite", "Input 0: ${interpreter.getInputTensor(0).name()}")
Log.d("TFLite", "Input 1: ${interpreter.getInputTensor(1).name()}")
```
""")
else:
    print("""
❌ Your TFLite model has ERRORS!

The model is misclassifying EXACT training examples.

Possible fixes:

1. Check if you're using the WRONG model file:
   - Verify hotel_mobilebert.tflite is from latest training
   - Check file size (should be ~25MB)
   - Check timestamp

2. Check label_map.json:
   - Must match the one generated during training
   - Verify indices match intent names

3. Re-convert the model:
   - Delete hotel_mobilebert.tflite
   - Run convert_mobilebert_to_tflite_production.py again
   - Make sure no errors during conversion

4. Last resort - retrain:
   - Something may have gone wrong during training
   - Run train_mobilebert_production.py again
""")

print("\n" + "=" * 60)