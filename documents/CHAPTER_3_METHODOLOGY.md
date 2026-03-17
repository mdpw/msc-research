# CHAPTER 3: METHODOLOGY

## 3.1 Introduction

This chapter presents the research methodology adopted for the design, development, and evaluation of the proposed low-cost, offline voice assistant system for hotel room service operations. The chapter begins with an overview of the research approach, followed by a detailed description of the system requirements, architecture, dataset preparation, model training, technology selection rationale, and evaluation strategy. Each design decision is justified in the context of the research objectives and the constraints imposed by the target deployment environment: small to mid-sized hotels in Sri Lanka with limited IT infrastructure and budgets.

## 3.2 Research Approach

This research follows a Design Science Research (DSR) methodology, as described by Hevner et al. (2004). DSR is concerned with the creation and evaluation of IT artefacts intended to solve identified organisational problems. The methodology is particularly suitable for this research as it involves the design of a novel software system (the artefact) to address a practical problem (inefficient hotel room service operations) and the evaluation of this artefact against defined performance criteria.

The DSR process adopted comprises the following phases:

1. **Problem Identification**: Identification of operational inefficiencies and cost barriers in hotel room service through literature review and analysis of existing commercial solutions (Chapter 2).
2. **Solution Design**: Definition of system requirements, architecture design, and technology selection based on the identified research gaps (Chapter 3).
3. **Artefact Development**: Implementation of the voice assistant system comprising an Android application, backend server, and staff dashboard (Chapter 4).
4. **Evaluation**: Assessment of the artefact against defined metrics including NLU accuracy, speech recognition accuracy, system latency, and cost-effectiveness (Chapter 5).
5. **Communication**: Presentation of findings, contributions, and limitations (Chapters 6 and 7).

## 3.3 System Requirements

### 3.3.1 Functional Requirements

The functional requirements were derived from an analysis of common hotel room service operations, the use cases documented in the literature (Buhalis & Moldavska, 2021, 2022), and the capabilities of existing commercial solutions (Alexa for Hospitality). Table 3.1 presents the functional requirements organised by priority using the MoSCoW method (Must, Should, Could, Won't).

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
| FR-10 | The system shall allow guests to rate completed services on a 1-5 scale | Should |
| FR-11 | The system shall enable staff to send text messages to guest devices via the dashboard | Should |
| FR-12 | The system shall support dark and light visual themes based on system preference | Could |
| FR-13 | The system shall support multiple server network profiles for different network environments | Could |

### 3.3.2 Non-Functional Requirements

**Table 3.2: Non-Functional Requirements**

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR-01 | End-to-end response latency (voice input to voice confirmation) | < 5 seconds | Comparable to human telephone response time |
| NFR-02 | Speech-to-text word error rate | < 20% | Sufficient for intent classification accuracy |
| NFR-03 | Intent classification accuracy | > 85% | Minimum threshold for reliable service routing |
| NFR-04 | System availability on local network | > 99% uptime | Hotels operate 24/7; system must be reliable |
| NFR-05 | Data privacy | Zero external data transmission | All voice data processed on-device or within hotel LAN |
| NFR-06 | Per-room hardware cost | < $150 USD | Must be affordable for budget hotels |
| NFR-07 | Guest learning curve | Zero setup required | Guests should not need to create accounts or learn the system |
| NFR-08 | Device storage footprint | < 500 MB (models + app) | Must fit on budget Android tablets with limited storage |

## 3.4 System Architecture

### 3.4.1 Architecture Overview

The system follows a three-tier architecture comprising: (1) the guest-facing Android application performing on-device speech and language processing, (2) a central hotel server handling request management and routing, and (3) a web-based staff dashboard for operational management. All three tiers operate within the hotel's local area network (LAN), ensuring that no data is transmitted to external servers.

**Figure 3.1: High-Level System Architecture**

```
+================================================================+
|                    HOTEL LOCAL AREA NETWORK                      |
|                                                                  |
|  +----------------------------+    +-------------------------+   |
|  |   GUEST ROOM DEVICE        |    |    HOTEL SERVER          |  |
|  |   (Android Tablet)         |    |    (Any PC/Laptop)       |  |
|  |                            |    |                          |  |
|  |  +--------+  +----------+ |    |  +--------+  +--------+  |  |
|  |  | Vosk   |  |MobileBERT| | HTTP|  |FastAPI |  |SQLite  |  |  |
|  |  | STT    |->| NLU      |------>|  |Backend |->|Database|  |  |
|  |  |(205MB) |  | (26MB)   | |    |  +--------+  +--------+  |  |
|  |  +--------+  +----------+ |    |      |                    |  |
|  |       ^           |       |    |      | WebSocket          |  |
|  |  Microphone   Intent +    |    |      v                    |  |
|  |       |     Confidence    |    |  +--------+               |  |
|  |  +--------+               |    |  |  WS    |               |  |
|  |  | Android|  <-- WebSocket-----|  | Hub    |               |  |
|  |  | TTS    |               |    |  +--------+               |  |
|  |  +--------+               |    |      |                    |  |
|  +----------------------------+    +------+--------------------+  |
|                                           |                      |
|                                           | WebSocket            |
|                                           v                      |
|                                   +------------------+           |
|                                   | STAFF DASHBOARD   |          |
|                                   | (Web Browser)     |          |
|                                   |                   |          |
|                                   | - Dept Queues     |          |
|                                   | - Status Mgmt     |          |
|                                   | - Messaging       |          |
|                                   | - Notifications   |          |
|                                   +------------------+           |
+================================================================+
```

The architectural decision to perform STT and NLU processing on the guest device, rather than on the central server, is motivated by three factors: (1) privacy preservation, as raw audio data never leaves the guest device; (2) reduced server load, as the server only receives structured text data; and (3) resilience, as guest devices can transcribe and classify requests independently of server availability, queuing submissions until connectivity is restored.

### 3.4.2 Voice Processing Pipeline

The voice processing pipeline describes the sequence of operations from the moment a guest initiates a voice request to the delivery of the final confirmation. Figure 3.2 illustrates the complete pipeline.

**Figure 3.2: Voice Processing Pipeline**

```
Guest taps             Voice Activity        Speech-to-Text         Transcription
microphone  --------->  Detection    ------->  (Vosk)       ------->  Cleaning
button                 (Energy-based          (On-device,            (Remove filler
                        silence detect,        16kHz PCM,             words, greetings,
                        1500ms timeout)        205MB model)           normalise text)
                              |                                            |
                              v                                            v
                     Audio Recording                              Cancel Detection
                     (16kHz, 16-bit,                              (Regex pattern
                      mono PCM)                                    matching for
                                                                   "cancel order #X")
                                                                       |
                                                          +-----------+-----------+
                                                          |                       |
                                                    Cancel Match            No Match
                                                          |                       |
                                                    Voice Confirm          NLU Classification
                                                    Cancel (Y/N)           (Tier 1: Keywords
                                                          |                 Tier 2: MobileBERT)
                                                          v                       |
                                                    Cancel API              Confidence Check
                                                                           (>= 0.60 threshold)
                                                                                  |
                                                                    +-------------+----------+
                                                                    |                        |
                                                              Above Threshold         Below Threshold
                                                                    |                        |
                                                              Voice Confirm            TTS: "Sorry,
                                                              Submit (Y/N)             could not
                                                                    |                  understand"
                                                    +---------------+
                                                    |               |
                                                   Yes              No
                                                    |               |
                                              HTTP Submit      TTS: "Cancelled"
                                              to Server
                                                    |
                                              TTS: "Request
                                              #X received"
```

### 3.4.3 Hybrid NLU Pipeline

A key architectural decision in this research is the adoption of a hybrid, two-tier NLU classification pipeline that combines rule-based keyword matching with neural model inference. This design balances speed, accuracy, and reliability.

**Tier 1: Rule-Based Keyword Matching**

The first tier applies deterministic keyword pattern matching against a curated dictionary of hotel service phrases. When a match is found, the system assigns the corresponding intent with a fixed confidence score of 0.99, bypassing the neural model entirely. This tier handles unambiguous, high-frequency requests (e.g., "I need extra towels," "room service menu") with zero inference latency and guaranteed correctness.

The keyword dictionary was designed with specificity to prevent false positives. Early iterations used single-word keywords (e.g., "water," "ice," "pool"), which produced incorrect classifications for ambiguous phrases (e.g., "water polo" classified as food_order). The dictionary was subsequently refined to use multi-word contextual phrases (e.g., "bottled water," "glass of water," "swimming pool") that provide sufficient context for unambiguous classification.

**Tier 2: MobileBERT Neural Model**

Requests that do not match any keyword pattern are passed to the second tier: a fine-tuned MobileBERT model converted to TensorFlow Lite format for on-device inference. The model processes the tokenised input text and outputs softmax probabilities across all 18 intent classes. The class with the highest probability is selected as the predicted intent, with the softmax score serving as the confidence value.

A minimum confidence threshold of 0.60 is applied to the model output. Requests classified below this threshold are rejected with a voice message asking the guest to rephrase, rather than being submitted with an incorrect classification. This threshold was determined empirically to balance false rejection rate against misclassification rate.

**Figure 3.3: Hybrid NLU Classification Flow**

```
Cleaned Transcription
        |
        v
+------------------+       Match Found        +------------------+
| Tier 1: Keyword  | -----------------------> | Intent + 0.99    |
| Dictionary       |                           | Confidence       |
+------------------+                           +------------------+
        |
        | No Match
        v
+------------------+       Confidence          +------------------+
| Tier 2:          |       >= 0.60             | Intent +         |
| MobileBERT       | -----------------------> | Model Confidence |
| (TFLite, 26MB)   |                           +------------------+
+------------------+
        |
        | Confidence < 0.60
        v
+------------------+
| Rejection:       |
| "Could not       |
| understand"      |
+------------------+
```

### 3.4.4 Communication Architecture

The system employs two communication protocols for different purposes:

**HTTP REST API** is used for transactional operations between the Android application and the hotel server. Request submission, cancellation, rating, and history retrieval all use standard HTTP POST/GET methods. This protocol provides reliable, request-response semantics with built-in error handling and retry capability.

**WebSocket** is used for real-time, bidirectional communication. Two categories of WebSocket connections are maintained:

1. **Guest WebSocket** (`ws://<server>/ws/guest/<room_number>`): Delivers real-time status updates, department changes, and staff messages from the server to the guest's Android device. The device uses Android TTS to read staff messages aloud to the guest.

2. **Dashboard WebSocket** (`ws://<server>/ws/dashboard`): Delivers new request notifications, status updates, and rating updates to all connected staff dashboard instances. Enables the dashboard to reflect changes made by other staff members in real-time.

### 3.4.5 Database Design

The system uses SQLite as its database engine, selected for its zero-configuration deployment, single-file storage, and sufficient performance for the expected request volumes (hundreds per day across all rooms). The database comprises two tables:

**Table 3.3: Database Schema — requests Table**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique request identifier |
| room_number | TEXT | NOT NULL | Guest room number |
| request_text | TEXT | NOT NULL | Original voice transcription |
| intent | TEXT | | Classified intent category |
| department | TEXT | NOT NULL | Routed department name |
| status | TEXT | DEFAULT 'pending' | Request lifecycle status |
| rating | INTEGER | | Guest rating (1-5, NULL if unrated) |
| created_at | TEXT | NOT NULL | ISO 8601 timestamp of creation |
| completed_at | TEXT | | ISO 8601 timestamp of completion |

**Table 3.4: Database Schema — staff_messages Table**

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY, AUTOINCREMENT | Unique message identifier |
| request_id | INTEGER | NOT NULL, FOREIGN KEY | Reference to parent request |
| message | TEXT | NOT NULL | Staff message content |
| staff_name | TEXT | NOT NULL | Name of sending staff member |
| created_at | TEXT | NOT NULL | ISO 8601 timestamp |

### 3.4.6 Department Routing

Guest requests are automatically routed to one of five hotel departments based on the classified intent. Table 3.5 presents the complete intent-to-department mapping, which was designed based on typical hotel organisational structures.

**Table 3.5: Intent-to-Department Routing**

| Department | Intents Routed | Rationale |
|------------|---------------|-----------|
| Housekeeping | room_cleaning, towel_request, toiletries_request, blanket_request, pillow_request, laundry_service, do_not_disturb | Room amenity and cleanliness-related requests |
| Room Service | food_order | Food, beverage, and dining-related requests |
| Maintenance | maintenance, temperature_control, lighting_control | Technical and engineering-related requests |
| Front Desk | checkout_billing, noise_complaint, emergency, wake_up_call, misc_request | Administrative, safety, and unclassified requests |
| Concierge | concierge_general, concierge_taxi | Guest services, transportation, and information requests |

Staff members have the ability to transfer requests between departments through the dashboard interface when automatic routing is incorrect or when a request requires escalation.

## 3.5 Technology Selection and Justification

### 3.5.1 Speech-to-Text: Vosk

**Table 3.6: STT Technology Comparison**

| Criterion | Vosk | Whisper | Google Cloud STT | Amazon Transcribe |
|-----------|------|---------|-----------------|-------------------|
| Offline Capability | Yes | Partial (server) | No | No |
| On-Device Mobile | Yes (205MB) | No (too large) | No | No |
| Cost | Free (open-source) | Free (open-source) | $0.006/15 sec | $0.024/min |
| Privacy | Full (local) | Partial | No (cloud) | No (cloud) |
| Accuracy (English) | Good | Excellent | Excellent | Excellent |
| Model Size | 50MB - 1.8GB | 150MB - 6GB | N/A (cloud) | N/A (cloud) |
| Latency | Low (on-device) | Medium (server) | Medium (network) | Medium (network) |

**Justification**: Vosk was selected as the STT engine based on three primary criteria: (1) it supports fully offline, on-device operation on Android, which is essential for the privacy and connectivity requirements; (2) the vosk-model-en-us-0.22-lgraph variant (205MB) provides an acceptable balance between model size and accuracy for commodity Android tablets; and (3) it is open-source and free of licensing costs, eliminating recurring per-request charges. While Whisper achieves higher accuracy, its computational requirements preclude on-device mobile deployment, and cloud-based alternatives violate the system's privacy and offline operation requirements.

### 3.5.2 Intent Classification: MobileBERT

**Table 3.7: NLU Model Comparison**

| Criterion | MobileBERT | DistilBERT | BERT-BASE | Rasa DIET |
|-----------|------------|------------|-----------|-----------|
| Model Size | 25 MB (TFLite) | ~260 MB | ~440 MB | ~500 MB+ |
| Mobile Inference | 50-150 ms | ~300 ms | Not feasible | Not feasible |
| On-Device TFLite | Yes | Limited | No | No |
| Accuracy (GLUE) | 77.7 | 79.0 | 78.3 | Varies |
| Architecture | Mobile-optimised | Distilled | Full | Server framework |

**Justification**: MobileBERT was selected over DistilBERT and BERT-BASE for three reasons: (1) its 25MB TFLite model size is practical for mobile deployment, being 10x smaller than DistilBERT; (2) its inference latency of 50-150ms on mobile devices enables real-time classification within the pipeline; and (3) its architecture was specifically designed for resource-limited devices through progressive knowledge distillation, unlike DistilBERT which was designed as a general-purpose smaller BERT. Rasa DIET was excluded because it requires a running server process, conflicting with the on-device processing requirement.

### 3.5.3 Backend Framework: FastAPI

**Justification**: FastAPI was selected for the backend server based on: (1) native asynchronous support required for WebSocket connections; (2) automatic API documentation generation via OpenAPI/Swagger; (3) built-in request validation through Pydantic models; and (4) Python ecosystem compatibility with the data science tools used for model training. Alternative frameworks considered included Flask (lacking native async support) and Django (excessive complexity for the API-only backend).

### 3.5.4 Database: SQLite

**Justification**: SQLite was selected over PostgreSQL and MySQL based on: (1) zero-configuration deployment requiring no separate database server process; (2) single-file storage simplifying backup and migration; (3) sufficient performance for the expected load (estimated hundreds of requests per day across all rooms); and (4) native support in Python's standard library. For larger hotel deployments exceeding 500 rooms, migration to PostgreSQL is recommended as future work.

### 3.5.5 Android UI Framework: Jetpack Compose with Material Design 3

**Justification**: Jetpack Compose was selected as the Android UI framework for: (1) declarative UI paradigm enabling reactive updates from WebSocket events; (2) native Material Design 3 support for modern, accessible interface design; (3) built-in theme system supporting dark and light modes via `isSystemInDarkTheme()`; and (4) Kotlin-first design aligning with modern Android development practices.

**Table 3.8: Android Application Configuration**

| Parameter | Value |
|-----------|-------|
| Language | Kotlin |
| UI Framework | Jetpack Compose (Material Design 3) |
| Compile SDK | 34 (Android 14) |
| Minimum SDK | 26 (Android 8.0 Oreo) |
| Target SDK | 34 |
| STT Model | vosk-model-en-us-0.22-lgraph (205 MB) |
| NLU Model | hotel_mobilebert.tflite (25.12 MB) |
| Audio Format | 16 kHz, 16-bit PCM, mono |
| TTS Engine | Android native TextToSpeech |

### 3.5.6 Text-to-Speech: Android Native TTS

**Justification**: The Android built-in TextToSpeech engine was selected over Coqui TTS and other offline TTS solutions based on: (1) zero additional storage requirement as it is pre-installed on all Android devices; (2) acceptable voice quality for short confirmation messages typical of hotel service interactions; (3) no additional model download or configuration required, supporting the zero-setup guest experience requirement. The TTS is configured with a speech rate of 0.9x (slightly slower than default) and standard pitch of 1.0x for clear, natural-sounding hotel assistant responses.

## 3.6 Dataset Preparation

### 3.6.1 Intent Category Design

The 18 intent categories were designed to cover the most common guest service requests in hotel operations. The categories were identified through: (1) analysis of hotel room service menus and guest service directories from Sri Lankan hotels; (2) the use cases documented by Buhalis and Moldavska (2021, 2022); and (3) the service categories supported by Alexa for Hospitality.

**Table 3.9: Intent Categories and Distribution**

| Intent Category | Description | Training Samples | Percentage |
|----------------|-------------|-----------------|------------|
| towel_request | Requests for towels | 508 | 10.2% |
| room_cleaning | Housekeeping and cleaning requests | 412 | 8.3% |
| food_order | Food, beverage, and room service orders | 337 | 6.8% |
| toiletries_request | Bathroom amenity requests | 333 | 6.7% |
| pillow_request | Pillow and bedding requests | 306 | 6.2% |
| temperature_control | Heating and cooling requests | 304 | 6.1% |
| blanket_request | Blanket and comforter requests | 260 | 5.2% |
| maintenance | Technical and repair requests | 256 | 5.2% |
| laundry_service | Laundry and dry cleaning requests | 247 | 5.0% |
| concierge_general | General information and service queries | 244 | 4.9% |
| wake_up_call | Alarm and wake-up requests | 237 | 4.8% |
| concierge_taxi | Transportation and taxi requests | 237 | 4.8% |
| do_not_disturb | Privacy and do-not-disturb requests | 237 | 4.8% |
| lighting_control | Light adjustment requests | 226 | 4.5% |
| noise_complaint | Noise and disturbance complaints | 214 | 4.3% |
| emergency | Emergency and urgent assistance | 214 | 4.3% |
| checkout_billing | Checkout and billing inquiries | 208 | 4.2% |
| misc_request | Uncategorised or general requests | 192 | 3.9% |
| **Total** | | **4,971** | **100%** |

### 3.6.2 Data Collection and Curation

The training dataset was manually curated to include natural language variations that reflect how hotel guests actually phrase service requests. For each intent category, examples were created covering:

- **Formal phrasing**: "I would like to request additional towels, please."
- **Casual phrasing**: "need more towels"
- **Indirect phrasing**: "the towels in my room are used up"
- **Common misspellings and speech recognition artefacts**: "towal request", "tawel pleas"
- **Abbreviated forms**: "xtra towels", "rm service"

All text was normalised to lowercase to match the output format of the Vosk speech recognition engine, which produces lowercase transcriptions. The dataset was saved in CSV format with two columns: `text` (the utterance) and `intent` (the classification label).

### 3.6.3 Train-Validation Split

The dataset was divided into training and validation sets using stratified random sampling to maintain proportional representation of each intent class across both sets.

**Table 3.10: Dataset Split**

| Set | Samples | Percentage |
|-----|---------|------------|
| Training | 4,226 | 85% |
| Validation | 746 | 15% |
| **Total** | **4,972** | **100%** |

Stratified splitting ensures that intent classes with fewer examples (e.g., misc_request with 192 samples) maintain proportional representation in the validation set, preventing evaluation bias.

## 3.7 Model Training

### 3.7.1 MobileBERT Fine-Tuning Process

The MobileBERT model was fine-tuned using the HuggingFace Transformers library with PyTorch as the backend. The pre-trained `google/mobilebert-uncased` checkpoint was used as the base model, with a sequence classification head added for 18 intent classes.

**Table 3.11: Training Configuration**

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Base Model | google/mobilebert-uncased | Smallest BERT variant designed for mobile |
| Max Sequence Length | 32 tokens | Hotel voice commands are typically < 10 words |
| Batch Size | 32 | Optimal for dataset size of ~5,000 examples |
| Learning Rate | 3e-5 | Standard for BERT fine-tuning (Devlin et al., 2019) |
| Epochs | 8 | Empirically determined; validation loss plateaus after epoch 6 |
| Optimiser | AdamW | Standard for transformer fine-tuning |
| Train/Validation Split | 85% / 15% | Stratified to preserve class distribution |
| Tokeniser | MobileBertTokenizer | Vocabulary size: 30,522 tokens |

The maximum sequence length of 32 tokens was chosen based on analysis of the training data, which showed that 98% of hotel service requests contain fewer than 10 words (approximately 15 tokens after tokenisation). This reduced sequence length improves inference speed and reduces memory consumption during on-device execution.

### 3.7.2 TensorFlow Lite Conversion

Following training, the PyTorch model was converted to TensorFlow Lite format for deployment on Android devices. The conversion process involved:

1. Exporting the fine-tuned PyTorch model to ONNX format
2. Converting from ONNX to TensorFlow SavedModel format
3. Converting from SavedModel to TensorFlow Lite using the TFLite converter with SELECT_TF_OPS delegate for operation compatibility

The resulting TFLite model has the following specifications:

**Table 3.12: TFLite Model Specifications**

| Property | Value |
|----------|-------|
| File Size | 25.12 MB |
| Input: input_ids | Shape [1, 32], dtype int32 |
| Input: attention_mask | Shape [1, 32], dtype int32 |
| Output: logits | Shape [1, 18], dtype float32 |
| Expected Inference Time | 50-150 ms on mobile device |
| Optimisations | DEFAULT, SELECT_TF_OPS |

## 3.8 Evaluation Strategy

The evaluation of the proposed system is structured around four dimensions: NLU model accuracy, speech recognition accuracy, system latency, and cost analysis.

### 3.8.1 NLU Model Evaluation

The MobileBERT intent classifier is evaluated on the held-out validation set (746 examples) using standard classification metrics:

- **Accuracy**: Proportion of correctly classified examples across all intent classes.
- **Precision**: For each intent class, the proportion of predictions for that class that are correct (minimising false positives).
- **Recall**: For each intent class, the proportion of actual examples of that class that are correctly identified (minimising false negatives).
- **F1-Score**: Harmonic mean of precision and recall, providing a balanced metric for each class.
- **Macro Average**: Unweighted mean of per-class metrics, treating all classes equally regardless of support.
- **Weighted Average**: Weighted mean of per-class metrics, accounting for class imbalance.

A confusion matrix is generated to identify systematic misclassification patterns between intent classes.

### 3.8.2 Speech-to-Text Evaluation

The Vosk STT engine is evaluated using Word Error Rate (WER), the standard metric for speech recognition accuracy:

```
WER = (Substitutions + Insertions + Deletions) / Total Words in Reference
```

The WER evaluation involves:
- Recording hotel service request utterances from multiple speakers
- Comparing Vosk transcriptions against ground truth text
- Benchmarking against Whisper (server-side) as an accuracy ceiling
- Analysing WER variation across accent groups and intent categories

The `jiwer` Python library is used for automated WER calculation.

### 3.8.3 System Latency Evaluation

End-to-end system latency is measured by instrumenting the voice processing pipeline with timestamps at each stage:

1. **Audio Recording + VAD**: Time from microphone activation to silence detection
2. **STT Processing**: Time for Vosk transcription
3. **NLU Classification**: Time for intent classification (keyword check + model inference if needed)
4. **Network Transmission**: Time for HTTP request submission to server
5. **TTS Response**: Time for voice confirmation playback

Measurements are collected over multiple requests and reported as mean, minimum, and maximum values.

### 3.8.4 Cost Analysis

A comparative cost analysis is conducted across three scenarios:

1. **This System**: Commodity Android tablet + local server + zero recurring costs
2. **Cloud-Based Alternative**: Alexa for Hospitality / Google Nest with cloud subscriptions
3. **Traditional Method**: Phone-based room service (staffing costs)

The analysis considers initial hardware costs, recurring subscription fees, maintenance costs, and total cost of ownership (TCO) over a 3-year period for a hypothetical 50-room hotel.

## 3.9 Summary

This chapter has presented the research methodology, system architecture, technology selections, dataset preparation, model training procedures, and evaluation strategy for the proposed voice assistant system. The Design Science Research approach provides a structured framework for the creation and evaluation of the system artefact. Key architectural decisions, including on-device STT and NLU processing, a hybrid classification pipeline, and WebSocket-based real-time communication, are justified in the context of the research objectives and deployment constraints. The following chapter details the implementation of these design decisions.
