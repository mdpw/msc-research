"""
STEP 2: Generate Vosk noisy versions of clean dataset

Pipeline for each sentence:
  clean text → gTTS (spoken audio) → ffmpeg (WAV 16kHz) → Vosk → noisy transcript

Uses the SAME Vosk model as the Android app: vosk-model-small-en-in-0.4

Prerequisites:
  nlu-model-env/Scripts/pip.exe install vosk gtts
  Download vosk-model-small-en-in-0.4 from alphacephei.com/vosk/models
  Extract to: nlu-model/vosk-model-small-en-in-0.4/

Run:
  cd nlu-model
  nlu-model-env/Scripts/python.exe research/step2_generate_vosk_noise.py

Output:
  vosk_transcriptions.csv   — clean_text, vosk_text, intent, changed
  vosk_only_dataset.csv     — vosk_text, intent  (Model B training data)
  paired_dataset.csv        — clean_text + vosk_text combined (Model C training data)
"""

import os
import json
import wave
import subprocess
import tempfile
import pandas as pd
from gtts import gTTS
from vosk import Model, KaldiRecognizer

# ── Config ────────────────────────────────────────────────────────────────────
CLEAN_DATASET  = 'new_hotel_dataset.csv'
VOSK_MODEL_DIR = 'vosk-model-small-en-in-0.4'
OUTPUT_TRANS   = 'vosk_transcriptions.csv'
OUTPUT_VOSK    = 'vosk_only_dataset.csv'
OUTPUT_PAIRED  = 'paired_dataset.csv'
CHECKPOINT     = 'step2_checkpoint.csv'
SAMPLE_RATE    = 16000
# ─────────────────────────────────────────────────────────────────────────────

# ── Check model exists ────────────────────────────────────────────────────────
if not os.path.exists(VOSK_MODEL_DIR):
    print(f"ERROR: Vosk model not found at '{VOSK_MODEL_DIR}'")
    print("Download vosk-model-small-en-in-0.4 from alphacephei.com/vosk/models")
    print("Extract so that the folder contains am/ conf/ graph/ subfolders")
    exit(1)

print("Loading Vosk model (vosk-model-small-en-in-0.4)...")
model = Model(VOSK_MODEL_DIR)
print("Vosk model loaded")

# ── Load clean dataset ────────────────────────────────────────────────────────
clean_df = pd.read_csv(CLEAN_DATASET)
print(f"Loaded {len(clean_df)} clean records from {CLEAN_DATASET}")

# Resume from checkpoint if exists
if os.path.exists(CHECKPOINT):
    done_df   = pd.read_csv(CHECKPOINT)
    done_set  = set(done_df['clean_text'].tolist())
    results   = done_df.to_dict('records')
    remaining = clean_df[~clean_df['text'].isin(done_set)]
    print(f"Resuming: {len(done_df)} already done, {len(remaining)} remaining")
else:
    done_set  = set()
    results   = []
    remaining = clean_df


def text_to_vosk(text: str) -> str:
    """Convert text -> gTTS MP3 -> WAV 16kHz mono -> Vosk transcript."""
    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, 'speech.mp3')
        wav_path = os.path.join(tmp, 'speech.wav')

        # Step 1: text → MP3 via gTTS
        tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
        tts.save(mp3_path)

        # Step 2: MP3 → WAV 16kHz mono via ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', mp3_path,
            '-ar', str(SAMPLE_RATE), '-ac', '1',
            wav_path, '-loglevel', 'quiet'
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Step 3: WAV → Vosk transcript
        with wave.open(wav_path, 'rb') as wf:
            rec = KaldiRecognizer(model, wf.getframerate())
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
            result = json.loads(rec.FinalResult())
            return result.get('text', '').strip().lower()


# ── Main pipeline ─────────────────────────────────────────────────────────────
print(f"\nRunning TTS → Vosk pipeline on {len(remaining)} sentences...")
print("This will take a while — grab a coffee.\n")

changed_count = 0
error_count   = 0
total         = len(remaining)

for i, (_, row) in enumerate(remaining.iterrows()):
    clean = row['text']
    intent = row['intent']

    try:
        vosk_text = text_to_vosk(clean)
        vosk_text = vosk_text if vosk_text else clean  # fallback if Vosk returns empty
        changed   = vosk_text != clean                 # calculated AFTER fallback
        if changed:
            changed_count += 1
        results.append({
            'clean_text': clean,
            'vosk_text':  vosk_text,
            'intent':     intent,
            'changed':    changed,
        })
    except Exception as e:
        error_count += 1
        results.append({
            'clean_text': clean,
            'vosk_text':  clean,
            'intent':     intent,
            'changed':    False,
        })

    # Progress + checkpoint every 200 records
    done = i + 1
    if done % 200 == 0 or done == total:
        pct = done / total * 100
        print(f"  {done}/{total} ({pct:.1f}%)  changed={changed_count}  errors={error_count}")
        pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

# ── Save outputs ──────────────────────────────────────────────────────────────
trans_df = pd.DataFrame(results)
trans_df.to_csv(OUTPUT_TRANS, index=False)

changed_pct = changed_count / len(trans_df) * 100
print(f"\nVosk changed {changed_count}/{len(trans_df)} sentences ({changed_pct:.1f}%)")

# Sample errors
print("\nSample Vosk errors:")
print(f"{'CLEAN':<45} {'VOSK OUTPUT':<45}")
print("-" * 90)
for _, row in trans_df[trans_df['changed']].head(15).iterrows():
    print(f"{row['clean_text']:<45} {row['vosk_text']:<45}")

# Model B: vosk only
vosk_df = trans_df[['vosk_text', 'intent']].rename(columns={'vosk_text': 'text'})
vosk_df.to_csv(OUTPUT_VOSK, index=False)

# Model C: clean + vosk mixed
clean_rows = trans_df[['clean_text', 'intent']].rename(columns={'clean_text': 'text'})
vosk_rows  = trans_df[['vosk_text',  'intent']].rename(columns={'vosk_text':  'text'})
paired_df  = pd.concat([clean_rows, vosk_rows]).drop_duplicates(subset='text')
paired_df.to_csv(OUTPUT_PAIRED, index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_TRANS}  — {len(trans_df)} paired records (clean + vosk)")
print(f"  {OUTPUT_VOSK}   — {len(vosk_df)} records  (Model B training)")
print(f"  {OUTPUT_PAIRED} — {len(paired_df)} records  (Model C training)")
print(f"\nNext: run step3_train_three_models.py")
