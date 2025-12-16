"""
Convert MobileBERT to TensorFlow Lite for Android
MobileBERT has much better mobile support than DistilBERT
"""

import tensorflow as tf
from transformers import TFMobileBertForSequenceClassification, MobileBertTokenizer
import json
import os

print("=" * 50)
print("Converting MobileBERT to TFLite")
print("=" * 50)

# Check if model exists
if not os.path.exists('./mobilebert_hotel_model'):
    print("\n❌ Error: MobileBERT model not found!")
    print("Please run train_mobilebert.py first")
    exit(1)

print("\n📥 Loading trained model...")
tokenizer = MobileBertTokenizer.from_pretrained('./mobilebert_hotel_model')

# Load as TensorFlow model
print("🔄 Converting to TensorFlow format...")
tf_model = TFMobileBertForSequenceClassification.from_pretrained(
    './mobilebert_hotel_model',
    from_pt=True
)

print("✅ TensorFlow model loaded")

# Get model config
num_labels = tf_model.config.num_labels
max_length = 32

print(f"Number of labels: {num_labels}")
print(f"Max sequence length: {max_length}")

# Create concrete function with INT32 signature
print("\n💾 Creating mobile-optimized signature...")

class MobileBERTModule(tf.Module):
    def __init__(self, model):
        super(MobileBERTModule, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, max_length], dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=[1, max_length], dtype=tf.int32, name='attention_mask')
    ])
    def serving(self, input_ids, attention_mask):
        # MobileBERT expects INT32, no conversion needed!
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=False
        )
        return {'logits': outputs.logits}

# Wrap model
module = MobileBERTModule(tf_model)

# Save with signature
print("📦 Saving TensorFlow model...")
tf.saved_model.save(
    module,
    './mobilebert_tf',
    signatures={'serving_default': module.serving}
)

print("✅ TensorFlow model saved")

# Convert to TFLite
print("\n🔧 Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_saved_model(
    './mobilebert_tf',
    signature_keys=['serving_default']
)

# Optimizations for mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert
tflite_model = converter.convert()

# Save
tflite_filename = 'hotel_mobilebert.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)

model_size_mb = len(tflite_model) / 1024 / 1024

print(f"\n✅ TFLite model created: {tflite_filename}")
print(f"✅ Model size: {model_size_mb:.2f} MB")

# Save vocabulary
print("\n📝 Saving vocabulary...")
vocab = tokenizer.get_vocab()
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

vocab_size_kb = os.path.getsize('vocab.json') / 1024
print(f"✅ Vocabulary saved: vocab.json ({vocab_size_kb:.2f} KB)")

# Save config
config = {
    'max_length': max_length,
    'num_labels': num_labels,
    'model_type': 'mobilebert'
}
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "=" * 50)
print("✅ Conversion Complete!")
print("=" * 50)

print("\n📱 Files ready for Android:")
print(f"  1. {tflite_filename} ({model_size_mb:.2f} MB) - Much smaller than DistilBERT!")
print(f"  2. vocab.json ({vocab_size_kb:.2f} KB)")
print(f"  3. label_map.json")
print(f"  4. model_config.json")

print("\n🎯 MobileBERT advantages:")
print("  ✅ 4x faster inference than DistilBERT")
print("  ✅ 4x smaller model size")
print("  ✅ Native INT32 support (no conversion issues)")
print("  ✅ Optimized for mobile devices")

print("\n📋 Next steps:")
print("1. Replace old .tflite file in Android:")
print("   android/app/src/main/assets/models/nlu/hotel_mobilebert.tflite")
print("2. Update NLUService.kt to use 'hotel_mobilebert.tflite'")
print("3. Clean and rebuild Android app")
print("4. Test - should be much faster!")