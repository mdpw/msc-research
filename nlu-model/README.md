# NLU Model — STT Error-Aware Intent Classification

Part of the MSc research: *Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models.*

## Research Finding

Standard NLU benchmarks evaluate models on clean text, but real offline deployments use STT engines (e.g. Vosk) that introduce transcription noise. This pipeline quantifies that gap and shows noise-aware training closes it.

| Model | Training Data | Test Input | Accuracy | F1 Macro |
|-------|--------------|------------|----------|----------|
| A (baseline) | Clean text only | Clean text | 98.07% | 0.9805 | ← benchmark |
| A (baseline) | Clean text only | Vosk output | 89.34% | 0.8908 | ← **the gap** |
| B | Vosk text only | Vosk output | 96.38% | 0.9636 | |
| **C (final)** | **Clean + Vosk mixed** | **Vosk output** | **99.06%** | **0.9905** | ← **fix** |

- Accuracy drop (clean → Vosk pipeline): **−8.73 pp**
- Accuracy gain (noise-aware training): **+9.72 pp**
- Gap recovery: **111.3%**

## Folder Structure

```
nlu-model/
├── research/                        # 5-step pipeline scripts
│   ├── step1_create_dataset.py            # Step 1: generate clean dataset via Claude API
│   ├── step2_generate_vosk_noise.py # Step 2: transcribe with Vosk to add STT noise
│   ├── step3_train_three_models.py  # Step 3: train Model A, B, C
│   ├── step4_evaluate.py            # Step 4: evaluate and measure the gap
│   ├── step5_convert_best_model.py  # Step 5: convert Model C to TFLite for Android
│   └── step6_wer_analysis.py        # Step 6: measure Vosk WER on paired dataset
├── confusion_matrices/              # Thesis figures (PNG heatmaps)
├── new_hotel_dataset.csv            # Clean dataset (~10k sentences, 18 intents)
├── paired_dataset.csv               # Clean + Vosk paired records (14,863 rows)
├── vosk_transcriptions.csv          # Vosk-transcribed paired dataset (10,080 rows)
├── vosk_only_dataset.csv            # Vosk-only dataset (10,080 rows)
├── evaluation_results.json          # Final research metrics
├── hotel_mobilebert_v2.tflite       # Deployed TFLite model (Model C, 25.1 MB)
├── label_map_v2.json                # Intent label map (17 classes)
└── requirements.txt
```

> **Not in git (too large):** `models/` (trained PyTorch weights ~1.2 GB), `vosk-model-small-en-in-0.4/`, `nlu-model-env/`

## Setup

```bash
# Create virtual environment
python -m venv nlu-model-env
nlu-model-env\Scripts\activate        # Windows
# source nlu-model-env/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

Download the Vosk STT model (required for Step 2):
- Download `vosk-model-small-en-in-0.4` from https://alphacephei.com/vosk/models
- Extract into `nlu-model/vosk-model-small-en-in-0.4/`

Set Claude API key (required for Step 1 only):
```bash
set ANTHROPIC_API_KEY=your-key-here   # Windows
```

## Running the Pipeline

All commands run from the `nlu-model/` directory:

```bash
# Step 1 — Generate clean dataset (requires Claude API key)
nlu-model-env\Scripts\python.exe research/step1_create_dataset.py

# Step 2 — Add Vosk transcription noise (requires Vosk model)
nlu-model-env\Scripts\python.exe research/step2_generate_vosk_noise.py

# Step 3 — Train all three models (~2–4 hours on CPU)
nlu-model-env\Scripts\python.exe research/step3_train_three_models.py

# Step 4 — Evaluate and produce thesis results table
nlu-model-env\Scripts\python.exe research/step4_evaluate.py

# Step 5 — Convert best model (Model C) to TFLite for Android
nlu-model-env\Scripts\python.exe research/step5_convert_best_model.py
```

## Android Deployment

After Step 5, copy the outputs to the Android project:

```
hotel_mobilebert_v2.tflite  →  android/app/src/main/assets/models/nlu/hotel_mobilebert.tflite
label_map_v2.json           →  android/app/src/main/assets/models/nlu/label_map.json
```

Then rebuild the Android app in Android Studio.

## Intents (18 classes)

`food_order`, `room_cleaning`, `towel_request`, `toiletries_request`, `maintenance`,
`concierge_taxi`, `wake_up_call`, `checkout_billing`, `pillow_request`, `blanket_request`,
`laundry_service`, `noise_complaint`, `concierge_general`, `do_not_disturb`, `emergency`,
`lighting_control`, `temperature_control`, `misc_request`
