#!/usr/bin/env python3
"""
Wake Word Data Collection Tool
Records audio samples for training wake word detection model
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import time

# Configuration
SAMPLE_RATE = 16000
DURATION = 2.0  # seconds
CHANNELS = 1

def create_directories():
    """Create directory structure for training data"""
    dirs = [
        "wake_word_data/hello_hotel",
        "wake_word_data/hi_hotel",
        "wake_word_data/background"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def record_sample(filename):
    """Record a single audio sample"""
    print(f"\n🔴 Recording in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("🎙️ RECORDING NOW! (2 seconds)")
    
    # Record audio
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()  # Wait for recording to complete
    
    # Save audio
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved: {filename}")
    
    # Play back (optional)
    play = input("Play back? (y/n): ").lower()
    if play == 'y':
        print("🔊 Playing back...")
        sd.play(audio, SAMPLE_RATE)
        sd.wait()

def collect_class_samples(class_name, target_count):
    """Collect samples for a specific class"""
    print("\n" + "=" * 70)
    print(f"📁 Collecting samples for: {class_name}")
    print("=" * 70)
    
    output_dir = f"wake_word_data/{class_name}"
    
    # Count existing samples
    existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    print(f"📊 Existing samples: {existing_files}")
    print(f"🎯 Target: {target_count} samples")
    
    if existing_files >= target_count:
        print(f"✅ Already have enough samples!")
        cont = input("Continue collecting more? (y/n): ").lower()
        if cont != 'y':
            return
    
    # Instructions based on class
    if class_name == "hello_hotel":
        print("\n📝 Instructions:")
        print("   - Say 'Hello Hotel' clearly")
        print("   - Try different tones and speeds")
        print("   - Vary your distance from microphone")
        print("   - Get different people to say it")
    elif class_name == "hi_hotel":
        print("\n📝 Instructions:")
        print("   - Say 'Hi Hotel' clearly")
        print("   - Try different tones and speeds")
        print("   - Vary your distance from microphone")
        print("   - Get different people to say it")
    else:  # background
        print("\n📝 Instructions:")
        print("   - Record silence/room noise")
        print("   - Record other conversations")
        print("   - Record TV/music in background")
        print("   - Record typing, doors closing, etc.")
        print("   - Say OTHER phrases (not the wake words)")
    
    # Start recording
    sample_num = existing_files + 1
    
    while sample_num <= target_count:
        print(f"\n--- Sample {sample_num}/{target_count} ---")
        
        action = input("Press ENTER to record, 's' to skip, 'q' to quit: ").lower()
        
        if action == 'q':
            print(f"✅ Collected {sample_num - 1} samples")
            break
        elif action == 's':
            continue
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"sample_{sample_num:03d}_{timestamp}.wav")
        
        # Record
        record_sample(filename)
        
        sample_num += 1
    
    print(f"\n✅ Completed! Total samples in {class_name}: {len(os.listdir(output_dir))}")

def batch_record_mode():
    """Quick batch recording mode - no playback"""
    print("\n" + "=" * 70)
    print("⚡ BATCH RECORDING MODE")
    print("=" * 70)
    
    class_name = input("Class (hello_hotel/hi_hotel/background): ").strip()
    count = int(input("How many samples? "))
    
    output_dir = f"wake_word_data/{class_name}"
    if not os.path.exists(output_dir):
        print(f"❌ Directory doesn't exist: {output_dir}")
        return
    
    existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    sample_num = existing_files + 1
    
    print(f"\n🚀 Recording {count} samples...")
    print("Get ready! Recording will start automatically with countdown.")
    
    for i in range(count):
        print(f"\n📍 Sample {i+1}/{count}")
        input("Press ENTER when ready...")
        
        # Countdown
        for j in range(3, 0, -1):
            print(f"{j}...")
            time.sleep(0.8)
        
        print("🎙️ RECORDING!")
        
        # Record
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32'
        )
        sd.wait()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"sample_{sample_num:03d}_{timestamp}.wav")
        sf.write(filename, audio, SAMPLE_RATE)
        
        print(f"✅ Saved: {filename}")
        sample_num += 1
        
        time.sleep(0.5)  # Brief pause between recordings
    
    print(f"\n🎉 Batch complete! Recorded {count} samples")

def verify_audio_quality():
    """Check audio quality of collected samples"""
    print("\n" + "=" * 70)
    print("🔍 Audio Quality Check")
    print("=" * 70)
    
    for class_name in ["hello_hotel", "hi_hotel", "background"]:
        dir_path = f"wake_word_data/{class_name}"
        if not os.path.exists(dir_path):
            continue
        
        files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
        
        if len(files) == 0:
            print(f"\n{class_name}: No files")
            continue
        
        print(f"\n{class_name}: {len(files)} files")
        
        # Check a few samples
        issues = []
        for file in files[:5]:  # Check first 5
            filepath = os.path.join(dir_path, file)
            audio, sr = sf.read(filepath)
            
            # Check sample rate
            if sr != SAMPLE_RATE:
                issues.append(f"  ❌ {file}: Wrong sample rate ({sr} Hz)")
            
            # Check duration
            duration = len(audio) / sr
            if duration < 1.0 or duration > 3.0:
                issues.append(f"  ⚠️  {file}: Unusual duration ({duration:.1f}s)")
            
            # Check if too quiet
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.01:
                issues.append(f"  ⚠️  {file}: Very quiet (RMS: {rms:.4f})")
        
        if issues:
            print("  Issues found:")
            for issue in issues:
                print(issue)
        else:
            print("  ✅ Quality looks good!")

def generate_synthetic_background():
    """Generate synthetic background noise samples"""
    print("\n" + "=" * 70)
    print("🔧 Generate Synthetic Background Noise")
    print("=" * 70)
    
    count = int(input("How many samples? "))
    output_dir = "wake_word_data/background"
    
    existing_files = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    
    print(f"\n🔄 Generating {count} synthetic noise samples...")
    
    for i in range(count):
        # Generate different types of noise
        noise_type = i % 3
        
        if noise_type == 0:
            # White noise
            audio = np.random.normal(0, 0.05, int(DURATION * SAMPLE_RATE))
        elif noise_type == 1:
            # Pink noise (1/f)
            audio = np.random.normal(0, 0.03, int(DURATION * SAMPLE_RATE))
            audio = np.cumsum(audio)  # Simple pink noise
            audio = audio / np.max(np.abs(audio)) * 0.05
        else:
            # Low amplitude noise (room tone)
            audio = np.random.normal(0, 0.01, int(DURATION * SAMPLE_RATE))
        
        # Save
        filename = os.path.join(output_dir, f"synthetic_{existing_files + i + 1:03d}.wav")
        sf.write(filename, audio.astype('float32'), SAMPLE_RATE)
    
    print(f"✅ Generated {count} synthetic background samples")

def main_menu():
    """Main menu"""
    while True:
        print("\n" + "=" * 70)
        print("🎤 Wake Word Data Collection Tool")
        print("=" * 70)
        print("\nOptions:")
        print("1. Collect 'Hello Hotel' samples")
        print("2. Collect 'Hi Hotel' samples")
        print("3. Collect Background samples")
        print("4. Batch recording mode (fast)")
        print("5. Generate synthetic background noise")
        print("6. Verify audio quality")
        print("7. Show statistics")
        print("0. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            target = int(input("Target number of samples (recommended 100-200): "))
            collect_class_samples("hello_hotel", target)
        elif choice == '2':
            target = int(input("Target number of samples (recommended 100-200): "))
            collect_class_samples("hi_hotel", target)
        elif choice == '3':
            target = int(input("Target number of samples (recommended 200-300): "))
            collect_class_samples("background", target)
        elif choice == '4':
            batch_record_mode()
        elif choice == '5':
            generate_synthetic_background()
        elif choice == '6':
            verify_audio_quality()
        elif choice == '7':
            show_statistics()
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

def show_statistics():
    """Show collection statistics"""
    print("\n" + "=" * 70)
    print("📊 Dataset Statistics")
    print("=" * 70)
    
    total = 0
    
    for class_name in ["hello_hotel", "hi_hotel", "background"]:
        dir_path = f"wake_word_data/{class_name}"
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) if f.endswith('.wav')])
            total += count
            
            # Recommendations
            if class_name == "background":
                recommended = "200-300"
                status = "✅" if count >= 200 else "⚠️"
            else:
                recommended = "100-200"
                status = "✅" if count >= 100 else "⚠️"
            
            print(f"\n{class_name}:")
            print(f"  {status} Collected: {count} samples")
            print(f"     Recommended: {recommended}")
        else:
            print(f"\n{class_name}: Directory not found")
    
    print(f"\n📈 Total: {total} samples")
    
    if total >= 400:
        print("\n🎉 You have enough data to train a good model!")
    elif total >= 200:
        print("\n👍 Good start! Collect more for better accuracy.")
    else:
        print("\n📝 Collect more samples for better results.")

if __name__ == "__main__":
    print("🎤 Wake Word Data Collection Tool")
    print("=" * 70)
    print("This tool helps you collect audio samples for training")
    print("your offline wake word detection model.")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Show instructions
    print("\n📋 TIPS FOR BEST RESULTS:")
    print("   1. Record in a quiet environment")
    print("   2. Use a decent microphone")
    print("   3. Get multiple people to record")
    print("   4. Vary tone, speed, and distance")
    print("   5. For background: record various non-wake-word sounds")
    
    input("\nPress ENTER to continue...")
    
    # Start main menu
    main_menu()