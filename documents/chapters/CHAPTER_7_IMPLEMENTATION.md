# CHAPTER 7: IMPLEMENTATION

## 7.1 Introduction

This chapter explains how the system designed in Chapter 6 was actually built. It walks through the development environment, the four iterative build cycles, and the specific implementation decisions for each component — the Android guest app, the NLU training pipeline, the backend server, and the staff dashboard. Where things did not go as planned, those problems and how they were fixed are also described, because honestly, those issues shaped the final design more than the parts that worked smoothly from the start.

---

## 7.2 Development Environment

The system spans three technology stacks, one for each layer of the architecture. Table 7.1 gives a full breakdown of everything used during development.

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

One deliberate choice throughout the project was to keep external dependencies as light as possible. Since this is a research prototype, keeping things simple and transparent mattered more than adding the kind of convenience features a larger project might need. The staff dashboard uses plain vanilla JavaScript with no frontend framework. The backend has no ORM — all database queries are written in plain SQL. The Android app only adds the Vosk and TFLite SDKs on top of the standard Android libraries. A minimal stack makes the prototype easier to understand, evaluate, and reproduce, which is the right priority at this stage.

---

## 7.3 Iterative Implementation Process

As described in Chapter 3, the system was built across four iterations. Each cycle produced a working prototype that was tested before moving on to the next.

### 7.3.1 Iteration 1 — On-Device Speech Recognition

**What was built:**
- Android application skeleton with microphone permission handling
- Vosk model loaded from the app's assets directory
- Audio capture using Android's `AudioRecord` API at 16kHz, 16-bit mono PCM
- Energy-based voice activity detection (VAD)
- Real-time transcription display on screen

**Issues encountered:**

The first model tried was the larger Vosk English model (1.8GB). On a budget Android tablet it took over 15 seconds to start and crashed with out-of-memory errors — not ideal for a hotel room. This led to switching to `vosk-model-small-en-in-0.4` (~36MB), which loads in about 3 seconds and has reasonable accuracy. The Indian English acoustic model also turned out to be a better match for Sri Lankan English accents than any US English model, which made the choice straightforward.

Getting the VAD right took some trial and error. The initial silence timeout was too short and kept cutting off recordings mid-sentence during natural pauses. After testing, a 1,500ms silence timeout and a 10,000ms maximum recording duration worked well — long enough for natural speech, short enough to not hang in a noisy room.

### 7.3.2 Iteration 2 — Intent Classification Pipeline

**What was built:**
- Three MobileBERT model variants (A, B, C) trained using `step3_train_three_models.py`
- TFLite conversion pipeline (`step5_convert_best_model.py`)
- TFLite model integrated into the Android app via `NLUService.kt`
- Custom BERT word-piece tokeniser written in Kotlin
- Rule-based keyword dictionary with pre-compiled regex patterns
- Hybrid classification logic with a 0.60 confidence threshold
- Label map and vocabulary loaded from app assets

**Issues encountered:**

One practical problem was that the standard HuggingFace tokeniser is Python-only and cannot run on Android. Tokenisation is the step that converts raw text into the numeric token IDs the model expects — so without it, the model cannot work on-device. To solve this, a custom tokeniser was built in Kotlin that replicates the same logic: it loads the BERT vocabulary (30,522 tokens) from `vocab.json`, splits words into subword pieces, adds the required special tokens ([CLS], [SEP], [UNK]), pads the result to 32 tokens, and truncates anything longer. The output is exactly what MobileBERT was trained to receive.

During early testing, the neural model sometimes gave unexpectedly low confidence on simple requests when Vosk introduced small transcription errors. For example, "I need towels" could come through as "i need tawels" and get a reduced score. This was the observation that led directly to the rule-based keyword matching layer — catching clear, simple requests before they even reach the neural model.

**The three-model research design** was the core of this iteration. The central experiment was running three training sessions with identical hyperparameters but different input data, to see how the type of training text affects NLU performance when the model is fed Vosk transcriptions at inference time:
- **Model A** — trained on clean text only (`new_hotel_dataset.csv`)
- **Model B** — trained on Vosk-transcribed text only (`vosk_only_dataset.csv`)
- **Model C** — trained on a mix of clean and Vosk-paired data (`paired_dataset.csv`)

Model C came out on top on the Vosk test set and was deployed as the production model.

### 7.3.3 Iteration 3 — Backend and Real-Time Communication

**What was built:**
- FastAPI application with all REST endpoints
- SQLite database with a five-table schema
- WebSocket connection manager for guest and dashboard channels
- Department routing engine (database-driven with a keyword fallback)
- Staff dashboard as a single self-contained HTML file
- Full request flow from the Android app through the server to the dashboard

**Issues encountered:**

WebSocket connection stability on Android was a real problem. When the device screen turned off or Wi-Fi briefly dropped, the client would silently disconnect and not reconnect. An exponential backoff reconnection mechanism was added in `WebSocketService.kt`:

```kotlin
private fun scheduleReconnect() {
    val delay = Math.min(2000L * (1 shl reconnectAttempts), MAX_RECONNECT_DELAY) // 30s cap
    reconnectHandler.postDelayed({
        reconnectAttempts++
        internalConnect()
    }, delay)
}
```

The retry delays go: 2s → 4s → 8s → 16s → 30s (capped). This avoids flooding the server during long outages while still recovering fast from brief drops.

The initial staff dashboard showed requests from all departments at once. With even a small test dataset this became visually cluttered. Department filtering was added so staff only see their own department's queue after logging in.

### 7.3.4 Iteration 4 — Integration and Refinement

**What was built:**
- Voice confirmation flow (TTS readback → listen for yes/no → submit or cancel)
- Request cancellation by voice ("cancel order [number]")
- Guest rating system (1–5 stars)
- Staff-to-guest messaging with TTS announcement on the guest device
- Request department transfer from the dashboard
- Request history sorted by status priority
- Transcription cleaning (stripping greetings like "Hi Sera")
- Network profile management for switching between server addresses

**Issues encountered:**

The voice confirmation flow created a feedback loop — the TTS output was sometimes picked up by the microphone and transcribed as the guest's reply. This was fixed by adding a short delay between TTS finishing and the microphone turning back on, and by keeping the confirmation recording window shorter than a standard request.

Parsing spoken request IDs for the cancellation feature needed to handle both digit strings and spoken numbers. A regex pattern handles this by matching cancellation phrases and pulling out the numeric ID — so both "cancel order 146" and "cancel order one four six" work correctly.

---

## 7.4 Component Implementation Details

### 7.4.1 Android Guest Application

The guest app is organised around a `MainActivity` that sets up and coordinates six core classes:

**Table 7.2: Android Application Components**

| Class | File | Responsibility |
|-------|------|---------------|
| AudioRecorder | AudioRecorder.kt | Captures 16kHz, 16-bit PCM mono audio in 4,096-byte chunks; energy-based VAD |
| VoskService | VoskService.kt | On-device speech-to-text using `vosk-model-small-en-in-0.4` |
| NLUService | NLUService.kt | Hybrid intent classification (keyword matching → MobileBERT TFLite) |
| ApiService | ApiService.kt | HTTP REST client (OkHttp 4.11.0) for submitting and retrieving requests |
| WebSocketService | WebSocketService.kt | Persistent WebSocket connection with exponential backoff reconnection |
| TextToSpeechService | TextToSpeechService.kt | Android native TTS (speech rate 0.9x, pitch 1.0, Locale.US) |

The UI is built entirely in Jetpack Compose inside `VoiceAssistantScreen`. It updates reactively as request states change — no manual refresh or polling needed. The microphone button has an animated pulse effect while recording, and a 20-bar audio level visualiser shows the captured signal strength in real time.

The on-device ML files are bundled in the APK under `assets/models/`:

**Table 7.3: On-Device ML Assets**

| Asset | Size | Purpose |
|-------|------|---------|
| `models/vosk-model-small-en-in-0.4/` | ~36MB | Offline speech recognition model (Indian English) |
| `models/nlu/hotel_mobilebert.tflite` | 26MB | Intent classification model (Model C — noise-aware) |
| `models/nlu/vocab.json` | ~623KB | BERT word-piece vocabulary (30,522 tokens) |
| `models/nlu/label_map.json` | <1KB | Intent index-to-name mapping (18 intents) |

The TFLite model takes three inputs — `input_ids`, `attention_mask`, and `token_type_ids` — each of shape `[1, 32]` (int32). It outputs logits of shape `[1, 18]` (float32). The `token_type_ids` input is all zeros because this is single-sentence classification, but the model architecture requires it to be present.

### 7.4.2 NLU Training Pipeline

The training pipeline is six sequential Python scripts in `nlu-model/research/`:

**Table 7.4: NLU Pipeline Scripts**

| Script | Purpose | Output |
|--------|---------|--------|
| `step1_create_dataset.py` | Generates 10,080 clean hotel utterances using Claude Haiku API | `new_hotel_dataset.csv` |
| `step2_generate_vosk_noise.py` | Runs each utterance through gTTS → ffmpeg → Vosk to produce transcribed pairs | `vosk_transcriptions.csv`, `vosk_only_dataset.csv`, `paired_dataset.csv` |
| `step3_train_three_models.py` | Fine-tunes three MobileBERT variants (A, B, C) with identical hyperparameters | Model A, B, C checkpoints |
| `step4_evaluate.py` | Evaluates all three models on clean and Vosk test sets | `evaluation_results.json`, confusion matrix PNGs |
| `step5_convert_best_model.py` | Converts Model C (best checkpoint) to TFLite | `hotel_mobilebert_v2.tflite` (copied to Android assets as `hotel_mobilebert.tflite`) |
| `step6_wer_analysis.py` | Computes WER and CER statistics on Vosk transcriptions | Per-intent and overall error rate analysis |

**Training configuration (identical across all three models):**

All three models were trained with the same hyperparameters — the full settings and the reasoning behind each choice are in Chapter 3 (Table 3.9). The only thing that differs between models is the training data. The data splits used were:
- **Train/val split:** 85%/15% stratified split of each model's training dataset
- **Test set:** 20% held-out from `vosk_transcriptions.csv` (2,016 samples), shared across all three models for fair comparison

**Table 7.6: Training Dataset Summary**

| Model | Training Data | Records | Test Accuracy (Vosk) | F1 Macro (Vosk) |
|-------|--------------|---------|---------------------|-----------------|
| Model A | `new_hotel_dataset.csv` (clean text) | 10,080 | 89.34% | 0.8908 |
| Model B | `vosk_only_dataset.csv` (Vosk-transcribed) | 10,080 | 96.38% | 0.9636 |
| Model C | `paired_dataset.csv` (clean + Vosk mixed) | 14,864 | **99.06%** | **0.9905** |

Model C was selected for deployment. The accuracy gap between Model A on clean input (98.07%) and Model A on Vosk output (89.34%) — a difference of 8.73 percentage points — is the core research finding. Model C, trained on mixed data, closes this gap entirely, reaching 99.06% on the same Vosk test set.

**Model conversion (step5_convert_best_model.py):**

The best Model C checkpoint goes through three stages to become a TFLite file the Android app can use:

1. **PyTorch → TensorFlow** — The saved PyTorch weights are loaded directly into a TensorFlow version of MobileBERT using the `from_pt=True` flag in `TFMobileBertForSequenceClassification`. This avoids retraining from scratch in TF.
2. **TensorFlow → SavedModel** — The model is wrapped in a `tf.Module` with a concrete serving function that fixes all three input shapes to `[1, 32]`, which is what the converter needs to trace the model graph.
3. **SavedModel → TFLite** — The TFLite converter applies `tf.lite.Optimize.DEFAULT` (dynamic range quantisation: weights are quantised to INT8, activations stay float32) with `SELECT_TF_OPS` enabled for BERT operations that standard TFLite builtins cannot handle.

This brings the model down from ~94MB (PyTorch) to 26MB (TFLite) — a 72% size reduction. Full INT8 quantisation was deliberately skipped because BERT-family models are sensitive to aggressive quantisation and can lose accuracy when all activations are quantised too.

The final file `hotel_mobilebert_v2.tflite` is copied into the Android assets directory as `hotel_mobilebert.tflite`.

### 7.4.3 Backend Server

The backend is kept deliberately simple — a single FastAPI application in `backend/app/` with just three files:

**Table 7.7: Backend File Structure**

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application, all 17 route handlers, WebSocket endpoints, department routing engine |
| `database.py` | SQLite schema definition, table creation, and all query functions |
| `models.py` | Pydantic models for request/response validation |

To start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Binding to `0.0.0.0` (rather than `localhost`) makes the server reachable from any device on the hotel's local Wi-Fi, which is what lets the Android tablets and browser dashboards connect to it.

The WebSocket connection manager keeps two registries in memory — a dictionary that maps room numbers to individual guest device connections, and a list of all active dashboard connections. When a request comes in via `POST /api/submit-request`, the server:

1. Saves the request to SQLite
2. Looks up the department from the `intent_department_mapping` table
3. Falls back to keyword analysis if the intent is not in the mapping
4. Falls back to Front Desk if neither method gives a match
5. Broadcasts a `new_request` WebSocket event to all connected dashboards
6. Returns the assigned department and request ID to the guest device

When staff update a request status, the server pushes a `status_update` event to that specific room's WebSocket channel, which the guest device announces using TTS.

### 7.4.4 Staff Dashboard

The staff dashboard is a single self-contained `dashboard.html` file served by FastAPI at `/dashboard`. Using a single file with embedded CSS and JavaScript was a deliberate choice — it removes the need for build tools, bundlers, or dependency management, and can be deployed on any machine with a browser.

The dashboard connects to `/ws/dashboard` using the native browser WebSocket API. On connection, the server immediately sends an `initial` event with all existing requests, so the dashboard has full state without needing a separate HTTP call. From there, events like `new_request`, `status_update`, `department_update`, `staff_message`, and `rating_update` trigger incremental DOM updates without refreshing the page.

Colour coding is consistent between the guest app and the dashboard:

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

The system runs entirely within the hotel's existing local Wi-Fi network. No extra network hardware is needed.

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
4. Open network profile settings on each tablet and enter the server's local IP address
5. Place tablets in guest rooms

The network profile feature in the Android app lets multiple server addresses be stored in `SharedPreferences` and switched without recompiling the app. This was useful for testing across different network environments during development.

> **Note — prototype deployment only.** The configuration above reflects what was actually used during development and evaluation: a laptop running SQLite, manual IP entry, and HTTP without TLS. A real hotel deployment would require a different stack — a production-grade server, PostgreSQL, MDM-managed tablets, and encrypted connections. Those differences are covered in Chapter 10 (Section 10.3, Table 10.1) and the production infrastructure recommendations in Chapter 11 (Section 11.2.9).

---

## 7.6 Current Status

At the point of submission, the prototype is fully functional within its defined scope. Table 7.10 maps each requirement from Chapter 4 to its current implementation status.

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

**Known limitations of the current prototype:**

- **Room number configuration is manual.** The room number on each tablet has to be set through the app settings. A production system would use a mobile device management (MDM) platform to handle this automatically.
- **No staff authentication.** The dashboard login only records a department and name — there is no password or role-based access control.
- **Single-room load testing only.** The system has been tested with one concurrent guest device. Multi-room load testing with many simultaneous connections has not been done yet.
- **Hardcoded server address.** The server IP is stored manually in `SharedPreferences`. A production deployment would manage this centrally.
- **No structured logging.** The backend and training scripts use `print` statements for debugging. This worked fine during development but makes it harder to diagnose issues at scale. Structured logging with severity levels and timestamps would be needed before any real-world deployment (discussed further in Chapter 10, Section 10.3).

---

## 7.7 Summary

The system was built across four development cycles, each producing a testable prototype. The most important decisions came out of what those iterations revealed: switching from the 1.8GB Vosk model to the 36MB Indian English variant, adding the hybrid NLU pipeline after seeing failures with the neural-only approach, and designing the three-model training experiment that forms the core research contribution.

The implementation uses Kotlin 2.0.21 for the Android client, Python with HuggingFace Transformers and FastAPI for the training pipeline and server, and vanilla HTML/CSS/JavaScript for the staff dashboard. Dependencies were kept minimal throughout by design. All functional requirements have been met, and the system works end-to-end from voice input on the guest tablet to a notification appearing on the staff dashboard. The next chapter presents the evaluation of this implementation.
