# CHAPTER 3: METHODOLOGY

## 3.1 Introduction

This chapter explains the research methodology, system architecture, technology choices, dataset preparation, model training, and evaluation strategy for the proposed low-cost offline voice assistant. Every design decision is explained in the context of the research objectives and the real constraints of the target environment: small to mid-sized hotels in Sri Lanka with limited IT infrastructure and tight budgets.

---

## 3.2 Research Approach

This research follows a Design Science Research (DSR) methodology, as described by Hevner et al. (2004). DSR focuses on creating and evaluating IT artefacts that solve real organisational problems. It was the natural fit for this project for three reasons:

1. **The research requires a working system.** The objectives cannot be met with theoretical models or surveys. Measuring whether an offline voice assistant performs acceptably on budget hardware requires actually building one and running it. Approaches like case studies or literature reviews cannot produce that evidence.

2. **The research gaps are practical.** As identified in Chapter 2, no existing work has applied offline, on-device NLU — using open-source edge STT and compressed mobile transformers — to the hospitality domain. Closing that gap requires a demonstrable implementation, not a conceptual proposal.

3. **Evaluation is built into the methodology.** DSR explicitly requires the artefact to be tested against defined criteria. This maps directly onto the research objectives: measuring NLU accuracy across clean and transcribed inputs, speech recognition WER, system latency, and deployment cost. The evaluation is not an afterthought — it is how the research contribution is established.

The DSR process for this project followed five phases:

1. **Problem Identification** — Identifying operational and technical gaps through literature review (Chapter 2).
2. **Solution Design** — Defining system requirements, architecture, and the experimental design for comparing NLU training strategies (Chapter 3).
3. **Artefact Development** — Building the Android application, backend server, staff dashboard, and three NLU model variants (Chapter 4).
4. **Evaluation** — Measuring NLU accuracy across training conditions, speech recognition WER, system latency, and cost against cloud alternatives (Chapter 5).
5. **Communication** — Reporting findings, limitations, and contributions (Chapters 6 and 7).

---

## 3.3 Development Approach

The system was built using iterative prototyping. This was not a textbook preference — it was the only sensible approach given how much technical uncertainty existed upfront.

The system combines on-device speech recognition, neural intent classification, and real-time local network communication. None of these could be fully designed before testing, because their real-world behaviour on budget hardware was unknown:

- Speech recognition accuracy could only be measured by running Vosk on an actual Android device with real voice input. No design document could predict how well the model would handle Sri Lankan English in a real hotel room environment.
- Intent classification performance depended heavily on how well the model generalised to imperfect speech-to-text output — something that only became visible after running the full pipeline end-to-end.
- End-to-end latency could only be measured with all components integrated and running together.

A sequential approach would have delayed these discoveries until it was too late to act on them. For example, the first prototype immediately revealed that the larger Vosk model caused load times exceeding 15 seconds on a budget tablet — a critical constraint that directly shaped model selection.

The system was developed across four iterations:

| Iteration | Focus | Key Output | Key Decision Made |
|-----------|-------|------------|-------------------|
| 1 | On-device speech recognition | Vosk integrated into Android app with live transcription | `vosk-model-small-en-in-0.4` selected; Indian English acoustic model matches Sri Lankan accent better than US English variants |
| 2 | Intent classification pipeline | MobileBERT fine-tuned, converted to TFLite, deployed on-device | Hybrid classification adopted after purely neural approach struggled on simple keyword-heavy requests |
| 3 | Backend and real-time communication | FastAPI server, SQLite database, WebSocket integration | WebSocket confirmed as viable for real-time guest-to-staff updates over a local network |
| 4 | System integration and evaluation | End-to-end system with staff dashboard and benchmarking | Full pipeline latency measured; NLU accuracy compared across three training conditions |

Each iteration ended with a review against the research objectives, and the findings informed the next phase.

---

## 3.4 System Requirements

### 3.4.1 Functional Requirements

Functional requirements were derived from common hotel room service operations, use cases documented by Buhalis and Moldavska (2021, 2022), and the capabilities of existing commercial solutions such as Alexa for Hospitality. Table 3.1 presents the requirements using the MoSCoW prioritisation method.

**Table 3.1: Functional Requirements**

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | The system shall convert guest voice input to text using on-device speech recognition without internet connectivity | Must |
| FR-02 | The system shall classify guest voice requests into one of 18 predefined hotel service intent categories | Must |
| FR-03 | The system shall automatically route classified requests to the appropriate hotel department | Must |
| FR-04 | The system shall provide voice confirmation of the understood request before submission, with the guest able to confirm or reject via voice | Must |
| FR-05 | The system shall transmit submitted requests to a central hotel server via local network | Must |
| FR-06 | The system shall display request status (pending, in progress, completed, cancelled) in real-time on the guest device | Must |
| FR-07 | The system shall provide a web-based staff dashboard with department-specific request queues | Must |
| FR-08 | The system shall allow staff to update request status and transfer requests between departments | Must |
| FR-09 | The system shall allow guests to cancel pending requests via voice command | Should |
| FR-10 | The system shall allow guests to rate completed services on a 1–5 scale | Should |
| FR-11 | The system shall enable staff to send text messages to guest devices via the dashboard | Should |
| FR-12 | The system shall support dark and light visual themes based on system preference | Could |
| FR-13 | The system shall support multiple server network profiles for different network environments | Could |

### 3.4.2 Non-Functional Requirements

**Table 3.2: Non-Functional Requirements**

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR-01 | End-to-end response latency (voice input to voice confirmation) | < 5 seconds | Comparable to a human telephone response |
| NFR-02 | Speech-to-text word error rate | < 20% | Sufficient for intent classification accuracy |
| NFR-03 | Intent classification accuracy | > 85% | Minimum threshold for reliable service routing |
| NFR-04 | System availability on local network | > 99% uptime | Hotels operate 24/7 |
| NFR-05 | Data privacy | Zero external data transmission | All voice data processed on-device or within hotel LAN |
| NFR-06 | Per-room hardware cost | < $150 USD | Must be affordable for budget hotels |
| NFR-07 | Guest learning curve | Zero setup required | Guests should not need accounts or training |
| NFR-08 | Device storage footprint | < 500 MB (models + app) | Must fit on budget Android tablets |

---

## 3.5 System Architecture

### 3.5.1 Architecture Overview

The system follows a three-tier architecture: (1) a guest-facing Android application that handles all on-device speech and language processing, (2) a central hotel server managing request routing and storage, and (3) a web-based staff dashboard for operational management. All three tiers operate entirely within the hotel's local area network (LAN) — no data leaves the building.

**Figure 3.1: High-Level System Architecture**

```
+================================================================+
|                    HOTEL LOCAL AREA NETWORK                    |
|                                                                |
|  +----------------------------+    +------------------------+  |
|  |   GUEST ROOM DEVICE        |    |    HOTEL SERVER        |  |
|  |   (Android Tablet)         |    |    (Any PC/Laptop)     |  |
|  |                            |    |                        |  |
|  |  +--------+  +----------+  |    |  +--------+ +------+   |  |
|  |  | Vosk   |  |MobileBERT|  | HTTP|  |FastAPI | |SQLite|   |  |
|  |  | STT    |->| NLU      |------->|  |Backend |->|  DB  |   |  |
|  |  |(small- |  |(26MB     |  |    |  +--------+ +------+   |  |
|  |  | en-in) |  | TFLite)  |  |    |      |                 |  |
|  |  +--------+  +----------+  |    |      | WebSocket       |  |
|  |       ^           |        |    |      v                 |  |
|  |  Microphone   Intent +     |    |  +--------+           |  |
|  |       |     Confidence     |    |  |  WS    |           |  |
|  |  +--------+                |    |  | Hub    |           |  |
|  |  | Android|  <--WebSocket------|  +--------+           |  |
|  |  | TTS    |                |    |      |                 |  |
|  |  +--------+                |    +------+-----------------+  |
|  +----------------------------+           |                    |
|                                           | WebSocket          |
|                                           v                    |
|                                   +------------------+         |
|                                   | STAFF DASHBOARD  |         |
|                                   | (Web Browser)    |         |
|                                   | - Dept Queues    |         |
|                                   | - Status Mgmt    |         |
|                                   | - Messaging      |         |
|                                   | - Notifications  |         |
|                                   +------------------+         |
+================================================================+
```

The decision to run both STT and NLU on the guest device, rather than on the server, is motivated by three factors: (1) privacy — raw audio never leaves the device; (2) reduced server load — the server only receives structured text requests; and (3) resilience — the device can transcribe and classify independently if the server is temporarily unavailable.

### 3.5.2 Voice Processing Pipeline

The voice processing pipeline describes the complete sequence of operations from microphone activation to final confirmation.

**Figure 3.2: Voice Processing Pipeline**

```
Guest taps             Voice Activity        Speech-to-Text         Transcription
microphone  --------->  Detection    ------->  (Vosk)       ------->  Cleaning
button                 (RMS energy            (On-device,            (Remove filler
                        threshold 0.02,        16kHz PCM,             words, normalise
                        1500ms silence         vosk-small-            to lowercase)
                        timeout,               en-in-0.4)
                        10s max duration)           |                      |
                              |                     v                      v
                              v              Audio chunked           Cancel Detection
                     Audio Recording          at 4096 bytes          (Regex pattern
                     (16kHz, 16-bit,                                  matching for
                      mono PCM)                                       "cancel order #X")
                                                                           |
                                                              +-----------+-----------+
                                                              |                       |
                                                        Cancel Match            No Match
                                                              |                       |
                                                        Voice Confirm          NLU Classification
                                                        Cancel (Y/N)           (Tier 1: Keywords
                                                                               Tier 2: MobileBERT)
                                                                                      |
                                                                              Confidence Check
                                                                                      |
                                                                        +-------------+----------+
                                                                        |                        |
                                                                  Above Threshold         Below Threshold
                                                                        |                        |
                                                                  Voice Confirm            TTS: "Sorry,
                                                                  Submit (Y/N)             could not
                                                                                           understand"
```

### 3.5.3 Hybrid NLU Pipeline

A two-tier hybrid NLU pipeline was designed to balance speed, accuracy, and reliability. The hybrid approach emerged during iteration 2, when the purely neural model occasionally under-performed on simple, keyword-heavy requests due to minor transcription variations.

**Tier 1 — Rule-Based Keyword Matching**

The first tier applies deterministic regex pattern matching against a curated dictionary of hotel service phrases. Patterns are pre-compiled at startup to avoid overhead during inference. When a match is found, the intent is returned immediately with a fixed confidence of 0.99, without invoking the neural model. This handles unambiguous, high-frequency requests (e.g., "I need extra towels," "room service menu") with zero inference latency.

The keyword dictionary was refined iteratively. Early versions used single-word keywords (e.g., "water," "ice") which produced false positives for ambiguous phrases. The dictionary was updated to use multi-word contextual phrases (e.g., "bottled water," "swimming pool") to prevent misclassification.

**Tier 2 — MobileBERT Neural Model**

Requests with no keyword match are passed to the fine-tuned MobileBERT model in TFLite format. The model processes the tokenised input and outputs softmax probabilities across all 18 intent classes. The class with the highest probability is selected as the predicted intent.

**Figure 3.3: Hybrid NLU Classification Flow**

```
Cleaned Transcription
        |
        v
+------------------+       Match Found        +--------------------+
| Tier 1: Keyword  | -----------------------> | Intent + 0.99      |
| Dictionary       |                          | Confidence         |
| (regex, pre-     |                          +--------------------+
|  compiled)       |
+------------------+
        |
        | No Match
        v
+------------------+       Confidence          +--------------------+
| Tier 2:          |       >= threshold        | Intent +           |
| MobileBERT       | -----------------------> | Model Confidence   |
| (hotel_mobilebert|                          +--------------------+
|  _v2.tflite,     |
|  26MB)           |
+------------------+
        |
        | Confidence < threshold
        v
+------------------+
| Rejection:       |
| "Could not       |
| understand"      |
+------------------+
```

### 3.5.4 Communication Architecture

The system uses two communication protocols for different purposes:

**HTTP REST API** handles transactional operations between the Android app and the server — request submission, cancellation, rating, and history retrieval. This provides reliable request-response semantics with built-in error handling.

**WebSocket** handles real-time, bidirectional communication. Two categories of connections are maintained:

1. **Guest WebSocket** (`ws://<server>/ws/guest/<room_number>`): Delivers real-time status updates and staff messages to the guest device. Messages are read aloud using Android TTS. The client uses exponential backoff reconnection (starting at 2,000ms, doubling per attempt, capped at 30 seconds).

2. **Dashboard WebSocket** (`ws://<server>/ws/dashboard`): Delivers new request notifications and status updates to all connected staff dashboard instances in real-time.

### 3.5.5 Database Design

SQLite was used as the database engine, selected for its zero-configuration deployment and single-file storage — both important for a system that needs to be set up quickly in hotels with no dedicated IT team. The database comprises two tables:

**Table 3.3: Database Schema — requests**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique request identifier |
| room_number | TEXT | NOT NULL | Guest room number |
| request_text | TEXT | NOT NULL | Original voice transcription |
| intent | TEXT | | Classified intent category |
| department | TEXT | NOT NULL | Routed department |
| status | TEXT | DEFAULT 'pending' | Request lifecycle status |
| rating | INTEGER | | Guest rating (1–5, NULL if unrated) |
| created_at | TEXT | NOT NULL | ISO 8601 timestamp |
| completed_at | TEXT | | ISO 8601 timestamp of completion |

**Table 3.4: Database Schema — staff_messages**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique message identifier |
| request_id | INTEGER | NOT NULL, FOREIGN KEY | Reference to parent request |
| message | TEXT | NOT NULL | Staff message content |
| staff_name | TEXT | NOT NULL | Name of sending staff member |
| created_at | TEXT | NOT NULL | ISO 8601 timestamp |

### 3.5.6 Department Routing

Requests are automatically routed to one of five hotel departments based on the classified intent:

**Table 3.5: Intent-to-Department Routing**

| Department | Intents Routed |
|------------|---------------|
| Housekeeping | room_cleaning, towel_request, toiletries_request, blanket_request, pillow_request, laundry_service, do_not_disturb |
| Room Service | food_order |
| Maintenance | maintenance, temperature_control, lighting_control |
| Front Desk | checkout_billing, noise_complaint, emergency, wake_up_call, misc_request |
| Concierge | concierge_general, concierge_taxi |

Staff can transfer requests between departments via the dashboard when automatic routing is incorrect or escalation is needed.

---

## 3.6 Technology Selection and Justification

### 3.6.1 Speech-to-Text: Vosk (`vosk-model-small-en-in-0.4`)

**Table 3.6: STT Technology Comparison**

| Criterion | Vosk | Whisper | Google Cloud STT | Amazon Transcribe |
|-----------|------|---------|-----------------|-------------------|
| Offline Capability | Yes | Partial (server) | No | No |
| On-Device Mobile | Yes | No (too large) | No | No |
| Cost | Free (open-source) | Free (open-source) | $0.006/15 sec | $0.024/min |
| Privacy | Full (local) | Partial | No (cloud) | No (cloud) |
| Accuracy (English) | Good | Excellent | Excellent | Excellent |
| Latency | Low (on-device) | Medium (server) | Medium (network) | Medium (network) |

**Justification**: Vosk was selected because it is the only STT engine that supports fully offline, on-device operation on Android. The `vosk-model-small-en-in-0.4` variant was specifically chosen over the US English models because its Indian English acoustic model is a closer phonetic match to Sri Lankan English accents — a critical consideration for this deployment context. It is open-source and free of any per-request licensing costs. While Whisper achieves higher accuracy, it cannot run on a mobile device, and cloud-based alternatives violate the system's offline and privacy requirements.

### 3.6.2 Intent Classification: MobileBERT

**Table 3.7: NLU Model Comparison**

| Criterion | MobileBERT | DistilBERT | BERT-BASE | Rasa DIET |
|-----------|------------|------------|-----------|-----------|
| TFLite Model Size | 26 MB | ~260 MB | Not feasible | Not feasible |
| Mobile Inference | 50–150 ms | ~300 ms | Not feasible | Not feasible |
| On-Device TFLite | Yes | Limited | No | No |
| Architecture | Mobile-optimised | Distilled | Full | Server framework |

**Justification**: MobileBERT was chosen over DistilBERT and BERT-BASE because its 26MB TFLite size makes it practical for Android deployment, its inference latency of 50–150ms enables real-time classification, and its architecture was designed specifically for resource-constrained devices through progressive knowledge distillation. Rasa DIET requires a running server process, conflicting with the on-device requirement.

### 3.6.3 Backend Framework: FastAPI

**Justification**: FastAPI was selected based on its native asynchronous support (required for WebSocket handling), automatic API documentation generation, built-in request validation through Pydantic models, and compatibility with the Python ecosystem used for model training. Flask was considered but lacks native async support; Django was considered too heavyweight for an API-only backend.

### 3.6.4 Database: SQLite

**Justification**: SQLite was selected over PostgreSQL and MySQL because it requires zero configuration, uses a single file (simplifying backup), and performs adequately for the expected load — estimated hundreds of requests per day across all rooms. It is also natively supported in Python's standard library. For hotels exceeding 500 rooms, migration to PostgreSQL would be recommended.

### 3.6.5 Android: Jetpack Compose with Material Design 3

**Justification**: Jetpack Compose was chosen for its declarative UI paradigm, which handles reactive updates from WebSocket events cleanly. Material Design 3 provides built-in support for dark/light theming via `isSystemInDarkTheme()`. Kotlin 2.0.21 was used as the development language, aligned with modern Android development practice.

**Table 3.8: Android Application Configuration**

| Parameter | Value |
|-----------|-------|
| Language | Kotlin 2.0.21 |
| UI Framework | Jetpack Compose (Material Design 3) |
| Compile SDK | 34 (Android 14) |
| Minimum SDK | 26 (Android 8.0 Oreo) |
| Target SDK | 34 |
| STT Model | vosk-model-small-en-in-0.4 |
| NLU Model | hotel_mobilebert_v2.tflite (26 MB) |
| Audio Format | 16 kHz, 16-bit PCM, mono |
| Audio Chunk Size | 4,096 bytes |
| TTS Engine | Android native TextToSpeech |

### 3.6.6 Text-to-Speech: Android Native TTS

**Justification**: Android's built-in TextToSpeech engine was selected over offline alternatives because it requires no additional storage (pre-installed on all Android devices), produces acceptable voice quality for short confirmation messages, and requires no setup — directly supporting the zero-setup guest experience requirement.

---

## 3.7 Dataset Preparation

### 3.7.1 Intent Category Design

The 18 intent categories were designed to cover the most common guest service requests in hotel operations, identified through analysis of hotel room service menus from Sri Lankan hotels, use cases from Buhalis and Moldavska (2021, 2022), and the service categories supported by Alexa for Hospitality.

**Table 3.9: Intent Categories**

| Intent Category | Description |
|----------------|-------------|
| towel_request | Requests for towels |
| room_cleaning | Housekeeping and cleaning requests |
| food_order | Food, beverage, and room service orders |
| toiletries_request | Bathroom amenity requests |
| pillow_request | Pillow and bedding requests |
| temperature_control | Heating and cooling requests |
| blanket_request | Blanket and comforter requests |
| maintenance | Technical and repair requests |
| laundry_service | Laundry and dry cleaning requests |
| concierge_general | General information and service queries |
| wake_up_call | Alarm and wake-up requests |
| concierge_taxi | Transportation and taxi requests |
| do_not_disturb | Privacy and do-not-disturb requests |
| lighting_control | Light adjustment requests |
| noise_complaint | Noise and disturbance complaints |
| emergency | Emergency and urgent assistance |
| checkout_billing | Checkout and billing inquiries |
| misc_request | Uncategorised or general requests |

### 3.7.2 Dataset Construction

The clean dataset (`new_hotel_dataset.csv`) contains **10,080 labelled utterances** — exactly **560 examples per intent** across all 18 categories, ensuring a perfectly balanced distribution. The average sentence length is 7.16 words. All text was normalised to lowercase to match Vosk's output format.

The dataset was generated using the Claude Haiku API to produce natural language variations, expanded with paraphrasing to cover formal, casual, indirect, and abbreviated phrasing styles. All utterances were formatted in Vosk output style — lowercase, no punctuation, contractions written without apostrophes.

### 3.7.3 Vosk Transcription Pairing — The Core Experimental Dataset

This step is the methodologically significant contribution of the data collection process. Each of the 10,080 clean utterances was converted to audio using Google TTS (`gTTS`, with `tld='co.in'` for Indian English pronunciation), converted to 16kHz mono WAV using ffmpeg, and then transcribed by the Vosk engine (`vosk-model-small-en-in-0.4`). This produced a paired dataset (`vosk_transcriptions.csv`) containing a clean and a Vosk-transcribed version of every utterance.

The transcription process introduced the following error characteristics:

| Metric | Value |
|--------|-------|
| Overall Word Error Rate (WER) | 11.43% |
| Character Error Rate (CER) | 4.61% |
| WER on changed sentences only | 23.84% |
| Sentences changed by Vosk | 4,819 (47.8%) |
| Sentences unchanged by Vosk | 5,261 (52.2%) |
| Highest WER by intent | temperature_control (16.83%) |
| Lowest WER by intent | emergency (6.78%) |

These paired records are the foundation of the three-model experimental design. Without them, it would not be possible to isolate the accuracy gap caused by the STT step alone.

### 3.7.4 Three Training Datasets

Three training datasets were derived from the paired data to train one model variant each:

**Table 3.10: Training Datasets**

| Dataset | Records | Contents | Model Trained |
|---------|---------|----------|---------------|
| new_hotel_dataset.csv | 10,080 | Clean text only | Model A (baseline) |
| vosk_only_dataset.csv | 10,080 | Vosk-transcribed text only | Model B |
| paired_dataset.csv | 14,864 | Clean + Vosk mixed (deduplicated) | Model C (proposed fix) |

The `paired_dataset.csv` contains 14,864 records rather than 20,160 because duplicate entries (sentences where Vosk produced identical output to the clean text) were removed to avoid redundancy.

### 3.7.5 Train/Validation/Test Split

Each model is trained using a stratified 85%/15% train/validation split of its respective training dataset. The test set is a held-out 20% of `vosk_transcriptions.csv` (2,016 samples), shared across all three models to ensure a fair and directly comparable evaluation.

---

## 3.8 Model Training

### 3.8.1 MobileBERT Fine-Tuning

All three model variants are fine-tuned from the same base checkpoint — `google/mobilebert-uncased` from HuggingFace — using an identical training configuration. The only variable is the training data, which allows the results to isolate the effect of training data type on NLU performance.

**Table 3.11: Training Configuration**

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Base Model | google/mobilebert-uncased | Smallest BERT variant designed for on-device deployment |
| Task | Multi-class classification (18 intents) | — |
| Epochs | 5 (with early stopping, patience = 2) | Training stops if F1 macro does not improve for 2 consecutive epochs |
| Learning Rate | 3e-5 | Standard for BERT fine-tuning (Devlin et al., 2019) |
| Batch Size | 16 | Suited to CPU training constraints |
| Optimiser | AdamW | Standard for transformer fine-tuning |
| Warmup Ratio | 10% of training steps | Stabilises early training |
| Weight Decay | 0.01 | L2 regularisation |
| Max Gradient Norm | 1.0 | Gradient clipping |
| LR Scheduler | Linear decay | — |
| Max Sequence Length | 32 tokens | Over 98% of hotel requests are under 10 words |
| Best Model Selection | Highest F1 macro on validation set | — |
| Seed | 42 | Reproducibility |

The maximum sequence length of 32 tokens was set after inspecting the training data, where over 98% of utterances contained fewer than 10 words (approximately 15 tokens after tokenisation). Shorter sequences reduce inference time and memory usage during on-device execution.

Training was run on CPU. All three models together are estimated to take approximately 6–9 hours to complete.

### 3.8.2 TensorFlow Lite Conversion

After training, the best-performing model checkpoint is converted to TFLite format for Android deployment. The conversion pipeline is:

```
PyTorch Checkpoint  →  TensorFlow SavedModel  →  TensorFlow Lite (.tflite)
```

Dynamic range quantisation (`tf.lite.Optimize.DEFAULT`) is applied rather than full INT8 quantisation. BERT-family models are sensitive to aggressive quantisation. DEFAULT optimisation converts weights to INT8 while keeping activations in float32 — achieving a significant size reduction without the accuracy degradation that full integer quantisation can cause in smaller transformer models.

The resulting TFLite model (`hotel_mobilebert_v2.tflite`) has the following specification:

**Table 3.12: TFLite Model Specification**

| Property | Value |
|----------|-------|
| File Size | 26 MB |
| Input: input_ids | Shape [1, 32], dtype int32 |
| Input: attention_mask | Shape [1, 32], dtype int32 |
| Input: token_type_ids | Shape [1, 32], dtype int32 |
| Output: logits | Shape [1, 18], dtype float32 |
| Expected Inference Time | 50–150 ms on mobile device |
| Supported Ops | TFLITE_BUILTINS + SELECT_TF_OPS |

---

## 3.9 Evaluation Methodology

The evaluation is structured around four dimensions, each tied directly to a research objective.

### 3.9.1 NLU Accuracy — Three-Model Comparison

This is the core evaluation of the research. The three model variants (A, B, C) are evaluated on the same 2,016-sample held-out test set under two input conditions:

- **Clean text**: Evaluates performance without transcription errors (upper-bound condition)
- **Vosk-transcribed text**: Evaluates performance under real pipeline conditions (where the accuracy gap becomes visible)

Metrics reported for each model and condition:

- **Accuracy**: Overall proportion of correct classifications
- **Precision, Recall, F1-score**: Per-intent and macro-averaged
- **Confusion matrix**: To identify systematic misclassification between semantically similar intents

The comparison between Model A on Vosk output versus Model C on Vosk output is the direct answer to the core research question: does noise-aware training close the accuracy gap introduced by real speech recognition in the offline pipeline?

An additional benchmark comparison is made against a server-side BERT-BASE classifier on the same test set, to contextualise performance relative to a larger, unconstrained model.

### 3.9.2 Speech Recognition Accuracy

- **Metric**: Word Error Rate (WER)
- **Formula**: `WER = (Substitutions + Insertions + Deletions) / Total Reference Words`
- **Method**: Vosk transcriptions compared against ground-truth reference text using the `jiwer` Python library
- **Benchmark**: Whisper (server-side) on the same corpus, as an accuracy ceiling
- **Variables analysed**: Accent variation, background noise level, utterance length, intent category

### 3.9.3 System Latency

- **Metric**: End-to-end response time from voice input completion to system confirmation
- **Method**: Timestamps logged at each pipeline stage — STT processing, NLU classification (keyword check and/or model inference), HTTP submission, WebSocket delivery, and TTS playback
- **Reported statistics**: Mean, minimum, maximum latency across multiple test requests
- **Goal**: Identify which pipeline stages contribute most to total latency

### 3.9.4 Cost-Effectiveness

- **Metric**: Per-room deployment cost (hardware + software + maintenance)
- **Method**: Itemised cost comparison across three scenarios:
  1. This system (commodity Android tablet, local server, no recurring fees)
  2. Cloud-based alternatives (Alexa for Hospitality, Google Nest for Hotels)
  3. Traditional phone-based room service (staffing cost estimate)
- **Projection period**: 3 years for a hypothetical 50-room hotel

---

## 3.10 Summary

This chapter has presented the research methodology, system architecture, technology selections, dataset preparation, model training procedures, and evaluation strategy for the proposed voice assistant system.

Design Science Research was chosen because the objectives require both a working system and its systematic evaluation. Iterative prototyping was used over a sequential approach because the technical unknowns in combining multiple AI components on resource-constrained hardware could only be resolved through working prototypes.

The most methodologically significant aspect of the data collection is the construction of a paired clean-and-Vosk-transcribed dataset — 10,080 utterances passed through a TTS-to-Vosk pipeline to simulate real pipeline input. This enables the three-model comparison (clean-trained, Vosk-trained, and mixed-trained) that directly addresses whether noise-aware training can close the accuracy gap introduced by on-device speech recognition.

Key architectural decisions — including the `vosk-model-small-en-in-0.4` for Indian English acoustic matching, the hybrid two-tier NLU pipeline, and WebSocket-based real-time communication over LAN — are all justified in the context of the research objectives and the deployment constraints of budget Sri Lankan hotels. The following chapter details the implementation of these design decisions.
