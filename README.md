# Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka

MSc Research Project — Using Small-Scale Neural Models for Privacy-Preserving, Edge-Based Hotel Service Automation.

## Research Summary

Standard NLU models are benchmarked on clean text, but real offline deployments use STT engines (Vosk) that introduce transcription noise. This research quantifies that accuracy gap and demonstrates that noise-aware training closes it — enabling a fully offline hotel voice assistant deployable on commodity Android hardware.

### Core Finding

| Model | Training Data | Test Input | Accuracy | F1 Macro |
|-------|--------------|------------|----------|----------|
| A — baseline | Clean text | Clean text | 98.07% | 0.9805 |
| A — baseline | Clean text | Vosk output | 89.34% | 0.8908 | ← **−8.73 pp gap** |
| B | Vosk only | Vosk output | 96.38% | 0.9636 |
| **C — deployed** | **Clean + Vosk** | **Vosk output** | **99.06%** | **0.9905** | ← **+9.72 pp recovery** |

Gap recovery: **111.3%** — noise-aware training more than closes the STT-induced accuracy drop.

## System Architecture

```
<img width="387" height="888" alt="image" src="https://github.com/mdpw/msc-research/blob/main/documents/images/3.4.1.png" />

```

Fully offline — no cloud, no external API calls, all processing on-device or within the local hotel network.

## Repository Structure

```
msc-research/
├── android/               # Android guest-room app (Kotlin + Jetpack Compose)
├── backend/               # FastAPI server + SQLite + staff dashboard
├── nlu-model/             # MobileBERT training pipeline (5-step research pipeline)
├── documents/             # Thesis chapters and diagrams
└── README.md
```

## Components

### Android App (`android/`)
- On-device STT via Vosk (`vosk-model-small-en-in-0.4`)
- On-device NLU via MobileBERT TFLite (Model C, 99.06% accuracy, 18 intents)
- Voice confirmation, cancellation, real-time request tracking via WebSocket
- See [android/README.md](android/README.md)

### Backend (`backend/`)
- FastAPI + SQLite, WebSocket hub for real-time updates
- Auto-routes requests to 5 hotel departments
- Staff dashboard served at `/dashboard`
- See [backend/README.md](backend/README.md)

### NLU Research Pipeline (`nlu-model/`)
- 5-step pipeline: dataset generation → Vosk noise → train A/B/C → evaluate → TFLite export
- 14,863 training samples (clean + Vosk-transcribed), 18 intent classes
- See [nlu-model/README.md](nlu-model/README.md)

## Quick Start

### Backend
```bash
cd backend
python -m venv hotel-backend-env && hotel-backend-env\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Android App
1. Open `android/` in Android Studio
2. Update server IP in `ServerConfig.kt`
3. Run ▶ on connected device

### NLU Pipeline (research reproduction)
```bash
cd nlu-model
python -m venv nlu-model-env && nlu-model-env\Scripts\activate
pip install -r requirements.txt
# Run steps 1-5 in research/ folder
```

## Key Tech Stack

| Layer | Technology |
|-------|-----------|
| STT | Vosk (`vosk-model-small-en-in-0.4`) |
| NLU | MobileBERT fine-tuned → TFLite (25.1 MB) |
| TTS | Android built-in TextToSpeech |
| Backend | FastAPI + SQLite + WebSocket |
| Android | Kotlin + Jetpack Compose + TFLite |

## Research Novelty

1. **Paired dataset** — 14,863 clean + Vosk-transcribed sentence pairs for the same utterances
2. **Gap measurement** — first quantification of STT noise impact on hotel NLU (−8.73 pp)
3. **Noise-aware training** — Model C trained on mixed data recovers +9.72 pp
4. **End-to-end offline system** — complete hospitality deployment without cloud dependency
