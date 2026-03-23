# CHAPTER 7: IMPLEMENTATION

## 7.1 Introduction

This chapter describes how the system design from Chapter 6 was actually built. It covers the development environment, the iterative build process across four cycles, and the key implementation decisions in each component — the Android guest application, the NLU training pipeline, the backend server, and the staff dashboard. Where things did not go according to plan, those issues and how they were resolved are described, since these cases often influenced the final design more than the parts that worked first time.

---

## 7.2 Development Environment

The system spans three technology stacks corresponding to its three-tier architecture. Table 7.1 lists the complete development environment.

**Table 7.1: Development Environment**

| Component | Technology | Version |
|-----------|-----------|---------|
| **Guest Application** | | |
| Language | Kotlin | 2.0.21 |
| UI Framework | Jetpack Compose | Latest stable (BOM 2024.09.00) |
| IDE | Android Studio | Hedgehog / Iguana |
| Min SDK | Android API 26 (Android 8.0) | |
| Compile/Target SDK | Android API 34 (Android 14) | |
| Build System | Gradle (Kotlin DSL) | |
| TFLite Runtime | TensorFlow Lite | 2.17.0 |
| WebSocket Client | OkHttp | 4.11.0 |
| STT Model | vosk-model-small-en-in-0.4 | ~36 MB |
| NLU Model | hotel_mobilebert_v2.tflite | 26 MB |
| Audio Format | 16 kHz, 16-bit PCM, mono | — |
| Audio Chunk Size | 4,096 bytes | — |
| TTS Engine | Android native TextToSpeech | Pre-installed |
| **NLU Training Pipeline** | | |
| Language | Python | 3.10+ |
| ML Framework | PyTorch + HuggingFace Transformers | 4.57.3 |
| Model Conversion | TensorFlow / tf-keras | 2.13+ |
| Data Processing | pandas, scikit-learn | |
| Dataset Generation | Claude Haiku API (anthropic) | |
| TTS for Vosk Pairing | gTTS | 2.3+ |
| Audio Conversion | ffmpeg / pydub | |
| STT for Pairing | Vosk Python SDK | 0.3.45+ |
| **Backend Server** | | |
| Language | Python | 3.10+ |
| Web Framework | FastAPI | 0.104.1 |
| ASGI Server | Uvicorn | 0.24.0 |
| Database | SQLite 3 | |
| WebSocket | Starlette (via FastAPI) | |
| Request Validation | Pydantic | 2.5.0 |
| **Staff Dashboard** | | |
| Languages | HTML5, CSS3, Vanilla JavaScript | |
| WebSocket Client | Native browser WebSocket API | |

A deliberate decision was made to keep external dependencies to a minimum throughout the project. The staff dashboard uses vanilla JavaScript with no frontend framework. The backend has no ORM layer — database queries are written in plain SQL. The Android app uses only the Vosk and TFLite SDKs beyond the standard Android libraries. This reduces deployment complexity and is consistent with the minimal IT expertise requirement (NFR-08).

---

## 7.3 Iterative Implementation Process

As described in the methodology (Chapter 3), the system was built across four iterations. Each produced a working prototype that was tested before moving on.

### 7.3.1 Iteration 1 — On-Device Speech Recognition

**What was built:**
- Android application skeleton with microphone permission handling
- Vosk model loaded from the app's assets directory
- Audio capture using Android's `AudioRecord` API at 16kHz, 16-bit mono PCM
- Energy-based voice activity detection (VAD)
- Real-time transcription display on screen

**Issues encountered:**

The first model tested was the larger Vosk English model (1.8GB). On a budget Android tablet it caused app startup times exceeding 15 seconds and occasional out-of-memory crashes — completely unusable in a hotel room setting. This drove the switch to `vosk-model-small-en-in-0.4` (~36MB), which loads in approximately 3 seconds with acceptable accuracy. The Indian English acoustic model was also a better fit for Sri Lankan English accents than a US English model of any size, which made the decision straightforward.

The VAD required careful calibration. The initial silence timeout was too short, cutting recordings off mid-sentence during natural pauses. Through iterative testing, a 1,500ms silence timeout with a 10,000ms maximum recording duration was found to work well — long enough to accommodate natural speech patterns, short enough to prevent the system hanging in a noisy room.

### 7.3.2 Iteration 2 — Intent Classification Pipeline

**What was built:**
- Three MobileBERT model variants (A, B, C) trained using `step3_train_three_models.py`
- TFLite conversion pipeline (`step5_convert_best_model.py`)
- TFLite model integrated into the Android app via `NLUService.kt`
- Custom BERT word-piece tokeniser implemented in Kotlin
- Rule-based keyword dictionary with pre-compiled regex patterns
- Hybrid classification logic with the 0.60 confidence threshold
- Label map and vocabulary loaded from app assets

**Issues encountered:**

The standard HuggingFace tokeniser is Python-only and cannot run on Android. A custom word-piece tokeniser was implemented in Kotlin that loads the BERT vocabulary (30,522 tokens) from `vocab.json` and handles subword tokenisation compatible with MobileBERT's expected input format. This includes special tokens ([CLS] ID 101, [SEP] ID 102, [UNK] ID 100), padding to 32 tokens, and truncation of longer inputs.

Early testing revealed that the neural model occasionally produced lower-than-expected confidence on simple requests when Vosk introduced minor transcription variations. For example, "I need towels" could come through as "i need tawels" and receive reduced confidence. This observation directly led to the rule-based keyword matching tier — intercepting clear, unambiguous requests before they reach the neural model.

**The three-model research design** was central to this iteration. Three separate training runs were performed, each with identical hyperparameters but different training data, to measure the effect of training data type on NLU performance in the offline pipeline:
- **Model A** — trained on clean text only (`new_hotel_dataset.csv`)
- **Model B** — trained on Vosk-transcribed text only (`vosk_only_dataset.csv`)
- **Model C** — trained on mixed clean + Vosk-paired data (`paired_dataset.csv`)

Model C was the best performer on the Vosk test set and was deployed as the production model.

### 7.3.3 Iteration 3 — Backend and Real-Time Communication

**What was built:**
- FastAPI application with all REST endpoints
- SQLite database with five-table schema
- WebSocket connection manager for guest and dashboard channels
- Department routing engine (DB-driven with keyword fallback)
- Staff dashboard as a single self-contained HTML file
- Full request submission flow from Android to server to dashboard

**Issues encountered:**

WebSocket connection stability on Android was a problem. When the device screen turned off or Wi-Fi briefly dropped, the client would silently disconnect without reconnecting. An exponential backoff reconnection mechanism was implemented in `WebSocketService.kt`:

```kotlin
private fun scheduleReconnect() {
    val delay = Math.min(2000L * (1 shl reconnectAttempts), MAX_RECONNECT_DELAY) // 30s cap
    reconnectHandler.postDelayed({
        reconnectAttempts++
        internalConnect()
    }, delay)
}
```

Reconnect delays: 2s → 4s → 8s → 16s → 30s (capped). This prevents flooding the server during extended outages while recovering quickly from brief drops.

The initial staff dashboard showed all requests from all departments. With even a small amount of test data this was visually overwhelming. Department filtering was added so each staff member sees only their department's queue after login.

### 7.3.4 Iteration 4 — Integration and Refinement

**What was built:**
- Voice confirmation flow (TTS readback → listen for yes/no → submit or cancel)
- Request cancellation by voice ("cancel order [number]")
- Guest rating system (1–5 stars)
- Staff-to-guest messaging with TTS announcement on the guest device
- Request department transfer from the dashboard
- Request history sorted by status priority
- Transcription cleaning (strip greetings like "Hi Sera")
- Network profile management for configuring different server addresses

**Issues encountered:**

The voice confirmation flow introduced a feedback loop — the TTS output was sometimes captured by the microphone and transcribed as the guest's response. This was resolved by adding a short delay between TTS completion and microphone reactivation, and by limiting the confirmation recording window to a shorter duration than a standard request.

Parsing spoken request IDs for the cancellation feature required handling both digit strings and spoken numbers. A regex pattern was implemented to match cancellation phrases and extract numeric identifiers, covering inputs like "cancel order 146" and "cancel order one four six".

---

## 7.4 Component Implementation Details

### 7.4.1 Android Guest Application

The guest app is structured around a `MainActivity` that initialises and coordinates six core classes:

**Table 7.2: Android Application Components**

| Class | File | Responsibility |
|-------|------|---------------|
| AudioRecorder | AudioRecorder.kt | Captures 16kHz, 16-bit PCM mono audio in 4,096-byte chunks; energy-based VAD |
| VoskService | VoskService.kt | On-device speech-to-text using `vosk-model-small-en-in-0.4` |
| NLUService | NLUService.kt | Hybrid intent classification (keyword matching → MobileBERT TFLite) |
| ApiService | ApiService.kt | HTTP REST client (OkHttp 4.11.0) for submitting and retrieving requests |
| WebSocketService | WebSocketService.kt | Persistent WebSocket connection with exponential backoff reconnection |
| TextToSpeechService | TextToSpeechService.kt | Android native TTS (speech rate 0.9x, pitch 1.0, Locale.US) |

The UI is built entirely in Jetpack Compose within `VoiceAssistantScreen`. The interface updates reactively as request states change — no manual refresh or polling is needed. The microphone button has an animated pulsing effect during recording, and a 20-bar audio level visualiser shows captured signal strength in real time.

The on-device ML assets are bundled in the APK under `assets/models/`:

**Table 7.3: On-Device ML Assets**

| Asset | Size | Purpose |
|-------|------|---------|
| `models/vosk-model-small-en-in-0.4/` | ~36MB | Offline speech recognition model (Indian English) |
| `models/nlu/hotel_mobilebert.tflite` | 26MB | Intent classification model (Model C — noise-aware) |
| `models/nlu/vocab.json` | ~623KB | BERT word-piece vocabulary (30,522 tokens) |
| `models/nlu/label_map.json` | <1KB | Intent index-to-name mapping (18 intents) |

The TFLite model takes three inputs — `input_ids`, `attention_mask`, and `token_type_ids` — each of shape `[1, 32]` (int32), and produces logits of shape `[1, 18]` (float32). The `token_type_ids` input is all zeros (single-sentence classification), but is required by the model architecture.

### 7.4.2 NLU Training Pipeline

The training and deployment pipeline is implemented as six sequential Python scripts in `nlu-model/research/`:

**Table 7.4: NLU Pipeline Scripts**

| Script | Purpose | Output |
|--------|---------|--------|
| `step1_create_dataset.py` | Generates 10,080 clean hotel utterances using Claude Haiku API | `new_hotel_dataset.csv` |
| `step2_generate_vosk_noise.py` | Runs each utterance through gTTS → ffmpeg → Vosk to produce transcribed pairs | `vosk_transcriptions.csv`, `vosk_only_dataset.csv`, `paired_dataset.csv` |
| `step3_train_three_models.py` | Fine-tunes three MobileBERT variants (A, B, C) with identical hyperparameters | Model A, B, C checkpoints |
| `step4_evaluate.py` | Evaluates all three models on clean and Vosk test sets | `evaluation_results.json`, confusion matrix PNGs |
| `step5_convert_best_model.py` | Converts Model C (best checkpoint) to TFLite | `hotel_mobilebert_v2.tflite` (copied to Android assets as `hotel_mobilebert.tflite`) |
| `step6_wer_analysis.py` | Computes WER and CER statistics on Vosk transcriptions | Per-intent and overall error rate analysis |

**Training configuration (all three models share identical settings):**

All three models were trained with the same hyperparameters — the full configuration with rationale for each choice is documented in Chapter 3 (Table 3.9). The only difference between models is training data. The training, validation, and test splits are:
- **Train/val split:** 85%/15% stratified split of each model's training dataset
- **Test set:** 20% held-out from `vosk_transcriptions.csv` (2,016 samples), shared across all three models for fair comparison

**Table 7.6: Training Dataset Summary**

| Model | Training Data | Records | Test Accuracy (Vosk) | F1 Macro (Vosk) |
|-------|--------------|---------|---------------------|-----------------|
| Model A | `new_hotel_dataset.csv` (clean text) | 10,080 | 89.34% | 0.8908 |
| Model B | `vosk_only_dataset.csv` (Vosk-transcribed) | 10,080 | 96.38% | 0.9636 |
| Model C | `paired_dataset.csv` (clean + Vosk mixed) | 14,864 | **99.06%** | **0.9905** |

Model C is the best performer and was selected for deployment. The accuracy drop from Model A on clean input (98.07%) to Model A on Vosk output (89.34%) — a gap of 8.73 percentage points — is the core research finding. Model C, trained on mixed data, closes this gap entirely, achieving 99.06% on the same Vosk test set.

**Model conversion (step5_convert_best_model.py):**

The best Model C checkpoint is converted to TFLite via a three-stage pipeline:

1. **PyTorch → TensorFlow** — The PyTorch checkpoint is loaded into `TFMobileBertForSequenceClassification` using `from_pt=True`.
2. **TensorFlow → SavedModel** — The model is wrapped in a `tf.Module` with a concrete serving function defining all three input signatures (shape `[1, 32]`).
3. **SavedModel → TFLite** — The TFLite converter applies `tf.lite.Optimize.DEFAULT` (dynamic range quantisation: weights → INT8, activations stay float32) with `SELECT_TF_OPS` support for BERT operations not covered by standard TFLite builtins.

This reduces the model from ~94MB (PyTorch) to 26MB (TFLite) — a 72% reduction. Full INT8 quantisation was deliberately avoided because BERT-family models are sensitive to aggressive quantisation and can lose accuracy.

The output `hotel_mobilebert_v2.tflite` is copied into the Android assets directory as `hotel_mobilebert.tflite`.

### 7.4.3 Backend Server

The backend is a single FastAPI application in `backend/app/` with three files:

**Table 7.7: Backend File Structure**

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application, all 17 route handlers, WebSocket endpoints, department routing engine |
| `database.py` | SQLite schema definition, table creation, and all query functions |
| `models.py` | Pydantic models for request/response validation |

The server is started with:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Binding to `0.0.0.0` makes the server accessible from any device on the hotel's local network.

The WebSocket connection manager maintains two registries in memory: a dictionary mapping room numbers to individual guest device connections, and a list of all active dashboard connections. When a request is submitted via `POST /api/submit-request`, the server:

1. Stores the request in SQLite
2. Looks up the department from the `intent_department_mapping` table
3. Falls back to keyword analysis if the intent is not in the mapping
4. Falls back to Front Desk if neither method produces a match
5. Broadcasts a `new_request` WebSocket event to all connected dashboards
6. Returns the assigned department and request ID to the guest device

When staff update a request status, the server pushes a `status_update` event to the specific room's WebSocket channel, which the guest device announces via TTS.

### 7.4.4 Staff Dashboard

The staff dashboard is a single self-contained `dashboard.html` file served by FastAPI at `/dashboard`. The decision to use a single file with embedded CSS and JavaScript was intentional — it eliminates build tools, bundlers, and dependency management, making it trivially deployable on any machine.

The dashboard uses the native browser WebSocket API to connect to `/ws/dashboard`. On connection, the server immediately sends an `initial` event containing all existing requests, so the dashboard has full state without needing a separate HTTP call. Subsequent events (`new_request`, `status_update`, `department_update`, `staff_message`, `rating_update`) trigger incremental DOM updates without page reloads.

Department colour coding is applied consistently between the guest app and the dashboard:

**Table 7.8: Department Colour Coding**

| Department | Colour | Hex Code |
|------------|--------|----------|
| Housekeeping | Emerald green | `#10b981` |
| Room Service | Amber | `#f59e0b` |
| Maintenance | Red | `#ef4444` |
| Front Desk | Blue | `#3b82f6` |
| Concierge | Purple | `#8b5cf6` |

---

## 7.5 Deployment

The system runs entirely within the hotel's existing local Wi-Fi infrastructure. No additional network hardware is needed.

**Table 7.9: Deployment Components**

| Component | Target | Requirements |
|-----------|--------|-------------|
| Backend server | Any PC/laptop on hotel Wi-Fi | Python 3.10+, FastAPI, Uvicorn |
| Staff dashboard | Browser on any device | Any modern web browser, no installation |
| Guest app | Android tablet per room | Android 8.0+, tablet with microphone |

**Deployment steps:**

1. Install Python 3.10+ and dependencies (`pip install -r requirements.txt`) on the hotel server machine
2. Start the FastAPI server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Install the Android APK on each guest room tablet
4. Open the network profile settings on each tablet and enter the server's local IP address
5. Place tablets in guest rooms

The network profile feature in the Android app allows multiple server addresses to be stored in `SharedPreferences` and switched without recompiling the application. This supports testing across different network environments.

---

## 7.6 Current Status

At the time of submission, the prototype is fully functional for its defined scope. Table 7.10 summarises the implementation status of each requirement.

**Table 7.10: Requirements Implementation Status**

| Requirement | Status | Notes |
|-------------|--------|-------|
| FR-01: Voice service requests | Implemented | Full voice input with VAD |
| FR-02: On-device STT without internet | Implemented | `vosk-model-small-en-in-0.4` |
| FR-03: Intent classification (18 categories) | Implemented | Hybrid pipeline (keyword + MobileBERT) |
| FR-04: Text confirmation before submission | Implemented | TTS readback + voice yes/no response |
| FR-05: Voice feedback on submission | Implemented | Android native TTS |
| FR-06: Real-time delivery to staff | Implemented | WebSocket push notification |
| FR-07: Staff status updates | Implemented | pending → in_progress → completed |
| FR-08: Bidirectional messaging | Implemented | Staff message → guest TTS announcement |
| FR-09: Automatic department routing | Implemented | 18 intents → 5 departments via DB mapping |
| FR-10: Request history | Implemented | Stored in SQLite, viewable on guest device |
| NFR-01: Fully offline operation | Implemented | All AI processing on-device |
| NFR-02: On-device voice processing | Implemented | No voice data transmitted externally |
| NFR-03: Response latency < 5 seconds | To be evaluated | Measured in Chapter 8 |
| NFR-04: Commodity Android tablets | Implemented | Tested on budget Android device |
| NFR-05: ≥ 90% intent classification accuracy | Achieved | 99.06% on Vosk test set (Model C) |
| NFR-06: Multi-room concurrent operation | Implemented | WebSocket supports multiple simultaneous connections |
| NFR-07: Browser-accessible dashboard | Implemented | Single HTML file, any modern browser |
| NFR-08: No specialised IT expertise | Partially met | Simple deployment, but initial setup requires basic command-line knowledge |

**Known limitations of the current prototype:**

- **Room number configuration is manual.** Each tablet requires the room number to be set through the app settings. A production deployment would use an MDM platform to provision this automatically.
- **No staff authentication.** The dashboard login only captures department and name — there is no password protection or role-based access control.
- **Single-room load testing only.** The system has been tested with one concurrent guest device. Multi-room load testing with many simultaneous connections is pending.
- **Hardcoded server address.** The server IP is stored manually in `SharedPreferences`. In production, this would be centrally managed.

---

## 7.7 Summary

The system was built across four iterative development cycles, each producing a testable prototype. The most consequential decisions came from what those iterations revealed: the switch from the 1.8GB Vosk model to the 36MB Indian English variant, the adoption of the hybrid NLU pipeline after observing neural-only classification failures, and the three-model training design that forms the core research contribution.

The implementation uses Kotlin 2.0.21 for the Android client, Python with HuggingFace Transformers and FastAPI for the training and server components, and vanilla HTML/CSS/JavaScript for the staff dashboard — with a deliberate emphasis on minimal dependencies throughout. All functional requirements have been implemented, and the system operates as a complete end-to-end prototype from voice input to staff notification. The following chapter presents the systematic evaluation of this implementation.
