#!/usr/bin/env python3
"""
Wake Word Model Training Script
Trains a TensorFlow Lite model for offline wake word detection
Target: "Hello Hotel" and "Hi Hotel"
100% offline - no external API dependencies
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import librosa
import soundfile as sf

print("=" * 70)
print("🎤 Wake Word Model Training")
print("=" * 70)

# Configuration
SAMPLE_RATE = 16000
DURATION = 2.0  # seconds
N_FEATURES = 40  # Number of features to extract
EPOCHS = 100
BATCH_SIZE = 32

def extract_features(audio_path, max_len=SAMPLE_RATE * 2):
    """
    Extract audio features for wake word detection
    Simplified MFCC-like features for efficiency
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or trim to exact length
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]
        
        # Extract features - divide into frames and compute energy
        features = []
        frame_size = len(audio) // N_FEATURES
        
        for i in range(N_FEATURES):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            
            # Compute frame energy (root mean square)
            energy = np.sqrt(np.mean(frame ** 2))
            features.append(energy)
        
        features = np.array(features)
        
        # Normalize features
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        return features
        
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}")
        return None

def load_dataset(data_dir):
    """
    Load dataset from directory structure:
    data/
    ├── hello_hotel/  (recordings of "Hello Hotel")
    ├── hi_hotel/     (recordings of "Hi Hotel")
    └── background/   (background noise, other speech)
    """
    X = []
    y = []
    
    classes = {
        'background': 0,
        'hello_hotel': 1,
        'hi_hotel': 2
    }
    
    print(f"\n📂 Loading dataset from: {data_dir}")
    print("-" * 70)
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️  {class_name}: Directory not found, skipping")
            continue
        
        audio_files = glob.glob(os.path.join(class_dir, "*.wav"))
        
        print(f"📁 {class_name}: Found {len(audio_files)} files")
        
        for audio_file in audio_files:
            features = extract_features(audio_file)
            if features is not None:
                X.append(features)
                y.append(label)
    
    if len(X) == 0:
        raise ValueError("No data loaded! Check your data directory structure.")
    
    X = np.array(X)
    y = np.array(y)
    
    print("-" * 70)
    print(f"✅ Total samples loaded: {len(X)}")
    print(f"   Features shape: {X.shape}")
    print(f"   Class distribution:")
    for class_name, label in classes.items():
        count = np.sum(y == label)
        percentage = (count / len(y)) * 100
        print(f"      {class_name}: {count} ({percentage:.1f}%)")
    
    return X, y

def build_model(input_shape=(N_FEATURES,), num_classes=3):
    """
    Build a lightweight model for wake word detection
    Optimized for on-device inference
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Hidden layers - kept small for efficiency
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_audio(audio, sample_rate=SAMPLE_RATE):
    """
    Apply data augmentation to increase training data variety
    """
    augmented = []
    
    # Original
    augmented.append(audio)
    
    # Add noise
    noise = np.random.normal(0, 0.005, len(audio))
    augmented.append(audio + noise)
    
    # Time shift
    shift = int(sample_rate * 0.1)  # 100ms shift
    augmented.append(np.roll(audio, shift))
    augmented.append(np.roll(audio, -shift))
    
    # Pitch shift (simplified)
    augmented.append(audio * 1.1)
    augmented.append(audio * 0.9)
    
    return augmented

def convert_to_tflite(model, output_path):
    """
    Convert Keras model to TensorFlow Lite format
    Optimized for mobile deployment
    """
    print("\n📦 Converting to TensorFlow Lite...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ TFLite model saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    return tflite_model

def main():
    # Check if data directory exists
    data_dir = "wake_word_data"
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Data directory not found: {data_dir}")
        print("\nPlease create the following structure:")
        print("wake_word_data/")
        print("├── hello_hotel/")
        print("│   ├── sample_001.wav")
        print("│   ├── sample_002.wav")
        print("│   └── ...")
        print("├── hi_hotel/")
        print("│   ├── sample_001.wav")
        print("│   ├── sample_002.wav")
        print("│   └── ...")
        print("└── background/")
        print("    ├── noise_001.wav")
        print("    ├── speech_001.wav")
        print("    └── ...")
        print("\nRecommended:")
        print("- 100-200 samples of 'Hello Hotel'")
        print("- 100-200 samples of 'Hi Hotel'")
        print("- 200-300 background/noise samples")
        print("\nAll audio files should be:")
        print("- WAV format")
        print("- 16kHz sample rate")
        print("- 1-2 seconds duration")
        return
    
    # Load dataset
    X, y = load_dataset(data_dir)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print(f"\n⚖️  Class weights (to handle imbalance):")
    for class_idx, weight in class_weight_dict.items():
        print(f"   Class {class_idx}: {weight:.2f}")
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_model()
    
    print(model.summary())
    
    # Train model
    print(f"\n🎯 Training model for {EPOCHS} epochs...")
    print("-" * 70)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Final Evaluation:")
    print("-" * 70)
    
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Testing accuracy: {test_acc:.2%}")
    
    # Detailed predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(f"\n🎯 Per-class accuracy:")
    for class_idx in range(3):
        class_mask = y_test == class_idx
        class_acc = np.mean(y_pred_classes[class_mask] == y_test[class_mask])
        class_name = ['background', 'hello_hotel', 'hi_hotel'][class_idx]
        print(f"   {class_name}: {class_acc:.2%}")
    
    # Save Keras model
    keras_model_path = "wake_word_model.h5"
    model.save(keras_model_path)
    print(f"\n💾 Keras model saved: {keras_model_path}")
    
    # Convert to TFLite
    tflite_model_path = "wake_word_model.tflite"
    convert_to_tflite(model, tflite_model_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📱 To use in Android app:")
    print(f"1. Copy {tflite_model_path} to:")
    print(f"   app/src/main/assets/models/wake_word/wake_word_model.tflite")
    print(f"2. Rebuild and run the app")
    print(f"3. The app will automatically use the trained model")
    print(f"\n🎯 Expected accuracy: {test_acc:.1%}")
    print("=" * 70)

if __name__ == "__main__":
    main()

# Additional utility: Record training samples
def record_samples():
    """
    Utility function to record training samples
    """
    import sounddevice as sd
    
    print("\n🎤 Recording Training Samples")
    print("=" * 70)
    
    class_name = input("Class name (hello_hotel/hi_hotel/background): ").strip()
    num_samples = int(input("Number of samples to record: "))
    
    output_dir = os.path.join("wake_word_data", class_name)
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        input(f"\nPress Enter to record sample {i+1}/{num_samples}...")
        
        print(f"🔴 Recording for 2 seconds...")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Save
        filename = os.path.join(output_dir, f"sample_{i+1:03d}.wav")
        sf.write(filename, audio, SAMPLE_RATE)
        print(f"✅ Saved: {filename}")
    
    print(f"\n✅ Recorded {num_samples} samples for {class_name}")

# Uncomment to use recording utility:
# record_samples()