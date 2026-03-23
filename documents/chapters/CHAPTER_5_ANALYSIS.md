# CHAPTER 5: ANALYSIS

## 5.1 Introduction

This chapter analyses the system requirements from Chapter 4 and translates them into a concrete system design. It covers how the system is conceptually structured, what the key interactions look like through use cases and process flows, how the data is organised, and which technologies were selected and why. The technology evaluation section in particular explains not just what was chosen, but why the alternatives were ruled out — since the constraints of this project (offline operation, budget hardware, privacy, no IT support) made most common choices unsuitable.

---

## 5.2 High-Level System Conceptual View

The system is structured across three distinct layers, all connected over the hotel's local Wi-Fi network. No external internet connection is needed for any core operation. The figure below shows this layout.

**Figure 5.1: High-Level System Conceptual View**

```
+================================================================+
|                    HOTEL LOCAL AREA NETWORK                    |
|                                                                |
|  +---------------------+        +---------------------------+  |
|  |   GUEST ROOM LAYER  |        |    SERVER LAYER           |  |
|  |  (Android Tablet)   |        |  (Any PC/Laptop on LAN)   |  |
|  |                     |        |                           |  |
|  |  [Microphone Input] |        |  [FastAPI Backend]        |  |
|  |         |           |        |  [SQLite Database]        |  |
|  |  [Vosk STT]         | HTTP   |  [WebSocket Hub]          |  |
|  |  (on-device)        |------->|                           |  |
|  |         |           |        +---------------------------+  |
|  |  [MobileBERT NLU]   |                  |                    |
|  |  (on-device TFLite) |                  | WebSocket          |
|  |         |           |                  v                    |
|  |  [Android TTS]      |        +---------------------------+  |
|  |  <---WebSocket------|--------|   STAFF DASHBOARD LAYER   |  |
|  +---------------------+        |   (Web Browser)           |  |
|                                 |   [Dept Request Queues]   |  |
|                                 |   [Status Management]     |  |
|                                 |   [Messaging]             |  |
|                                 +---------------------------+  |
+================================================================+
```

The key design principle is that all AI processing — speech recognition and intent classification — happens on the guest device. The server is deliberately kept lightweight: it stores requests, routes them to departments, and pushes real-time updates to connected clients. This separation means the system is resilient to server-side issues and that raw audio never travels beyond the guest's tablet.

---

## 5.3 Use Case Model

Two primary actors interact with the system: the **Guest**, who uses voice and touch on the Android tablet, and the **Staff Member**, who manages requests through the web-based dashboard.

**Figure 5.2: UML Use Case Diagram**

*(See attached UML diagram — Figure 5.2)*

### 5.3.1 Use Case Descriptions

**Table 5.1: Use Case Descriptions**

| ID | Use Case | Actor | Description |
|----|----------|-------|-------------|
| UC-01 | Make voice service request | Guest | Guest presses the microphone button and speaks a service request. The system records audio, transcribes it using Vosk, classifies the intent using the hybrid NLU pipeline, and presents a confirmation screen. |
| UC-02 | Confirm or reject request | Guest | The system reads back the recognised request via Android TTS. Guest says "yes" to confirm or "no" to reject. On confirmation, the request is submitted to the server via HTTP POST. |
| UC-03 | View request history | Guest | Guest views a scrollable list of current and past requests with live status indicators, sorted by priority: in progress → pending → completed → cancelled. |
| UC-04 | Cancel request | Guest | Guest says "cancel order [number]". The system extracts the request ID using regex, confirms via voice, and sends a cancel request to the server. |
| UC-05 | Rate completed service | Guest | After a request is marked completed, the guest can submit a 1–5 star rating. The rating is sent to the server and becomes visible on the staff dashboard. |
| UC-06 | Receive real-time updates | Guest | When staff update a request status or send a message, the guest device receives a WebSocket notification and announces it via TTS. |
| UC-07 | Log in to dashboard | Staff | Staff member selects their department and enters their name to access their department-filtered request queue. |
| UC-08 | View department requests | Staff | Dashboard shows all requests routed to the staff member's department, with status badges, timestamps, and room numbers. |
| UC-09 | Update request status | Staff | Staff clicks "In Progress" or "Complete". The change is immediately broadcast to the relevant guest device via WebSocket. |
| UC-10 | Send message to guest | Staff | Staff types a message (e.g., "We'll be there in 5 minutes") which is delivered to the guest device and read aloud via TTS. |
| UC-11 | Transfer request | Staff | Staff reassigns a request to a different department using a dropdown selector. The request immediately appears in the new department's queue. |

---

## 5.4 Process Flow Analysis

### 5.4.1 Voice Request Pipeline

The voice request pipeline covers everything from the guest pressing the microphone button to the staff dashboard being updated. The key constants that govern this flow come directly from the implementation: a 1,500ms silence timeout to detect end of speech, a maximum recording duration of 10,000ms, a minimum NLU confidence threshold of 0.60, and a maximum tokenisation length of 32 tokens.

**Figure 5.3: Voice Request Pipeline**

```
Guest presses
microphone button
       |
       v
[Start Audio Recording]
16kHz, 16-bit PCM, mono
4,096-byte chunks
       |
       v
[Voice Activity Detection]
RMS energy threshold: 0.02
       |
       +---> Speech energy detected? --No--> Keep waiting (max 10,000ms)
       |
      Yes
       |
       v
[Record until silence]
1,500ms silence timeout
       |
       v
[Vosk STT Processing]
vosk-model-small-en-in-0.4
On-device, real-time streaming
       |
       v
[Clean Transcription]
Remove greetings (e.g., "Hi Sera")
Normalise to lowercase
       |
       v
[Check cancellation pattern]
Regex: "cancel order [number]"
       |
       +---> Match? --Yes--> Extract request ID
       |                           |
       |                    Voice confirmation
       |                           |
       |                    Cancel via HTTP --> End
       |
      No
       |
       v
[Tier 1: Keyword Matching]
Pre-compiled regex patterns
Case-insensitive, word boundary matching
       |
       +---> Match found? --Yes--> Intent + confidence 0.99 --> [Confirmation Step]
       |
      No
       |
       v
[Tier 2: MobileBERT TFLite]
Tokenise (max 32 tokens)
hotel_mobilebert_v2.tflite (26MB)
Softmax over 18 intent classes
       |
       v
[Confidence check]
       |
       +---> < 0.60? --Yes--> TTS: "Sorry, could not understand" --> End
       |
      >= 0.60
       |
       v
[Confirmation Step]
TTS: "You'd like [intent]. Shall I submit this?"
Record yes/no response
       |
       +---> "No" --> TTS: "Request cancelled" --> End
       |
      "Yes"
       |
       v
[HTTP POST to server]
Submit request with room number,
transcription, intent, department
       |
       v
[Server Processing]
Store in SQLite
Route to department
Broadcast via WebSocket to dashboard
       |
       v
[TTS: "Your request has been submitted"]
End
```

### 5.4.2 Staff Request Handling Flow

**Figure 5.4: Staff Request Handling Flow**

```
New request appears
on staff dashboard
(WebSocket notification)
       |
       v
[Staff reviews request]
Request text, intent,
room number, timestamp
       |
       v
[Correct department?]
       |
       +---> No --> Transfer to correct dept via dropdown --> End (for this staff member)
       |
      Yes
       |
       v
[Mark as "In Progress"]
Status update broadcast to guest via WebSocket
Guest device announces update via TTS
       |
       v
[Need to message guest?]
       |
       +---> Yes --> Type and send message
       |                    |
       |             Guest receives TTS announcement
       |
      No
       |
       v
[Fulfil request]
Staff performs the physical service
       |
       v
[Mark as "Completed"]
Completed timestamp recorded in SQLite
Guest receives WebSocket notification + TTS
       |
       v
[Guest rates service — optional]
1–5 star rating submitted via HTTP
Rating visible on staff dashboard
       |
       v
End
```

---

## 5.5 Entity-Relationship Model

The system's data model is deliberately simple — two tables are sufficient for all current functionality. The `requests` table is the core entity, and `staff_messages` exists in a one-to-many relationship with it (one request can have multiple messages from staff).

**Figure 5.5: Entity-Relationship Diagram**

*(See attached ER diagram — Figure 5.5)*

```
+---------------------------+          +---------------------------+
|         requests          |          |      staff_messages       |
+---------------------------+          +---------------------------+
| PK  id          INTEGER   |1       * | PK  id          INTEGER   |
|     room_number TEXT       |----------| FK  request_id  INTEGER   |
|     request_text TEXT      |          |     message     TEXT      |
|     intent      TEXT       |          |     staff_name  TEXT      |
|     department  TEXT       |          |     created_at  TEXT      |
|     status      TEXT       |          +---------------------------+
|     rating      INTEGER    |
|     created_at  TEXT       |
|     completed_at TEXT      |
+---------------------------+
```

**requests** stores every service request submitted by a guest. The `status` field tracks the request lifecycle: `pending` → `in_progress` → `completed` (or `cancelled`). The `intent` and `department` fields are populated by the NLU pipeline before the request reaches the server. The `rating` field is nullable — it is only set if the guest provides feedback after completion.

**staff_messages** stores messages sent from staff to a guest room. Each message is linked to a specific request, so staff communications are contextually tied to the service interaction they relate to.

SQLite was chosen as the database engine for this prototype. Its zero-configuration, single-file design is practical for small hotel deployments where there is no dedicated IT staff to manage a database server. For larger deployments, the schema is designed to be migration-compatible with PostgreSQL, requiring only a driver change. This is discussed further in Section 5.5.4.

---

## 5.6 Technology Evaluation and Selection

Technology selection for each component was guided by five constraints derived directly from the requirements in Chapter 4: offline capability (NFR-01), privacy preservation (NFR-02), low response latency (NFR-03), low-cost hardware deployment (NFR-04), and minimal IT infrastructure overhead (NFR-08).

It is worth noting that these constraints reflect two simultaneous objectives: producing a prototype that is valid enough to evaluate the core research hypotheses, and demonstrating a credible path to real-world deployment. Where the prototype uses a simpler option — for example SQLite instead of PostgreSQL — the production-grade alternative is identified explicitly. Table 5.2 summarises the selected stack.

**Table 5.2: Technology Stack Summary**

| Component | Prototype Selection | Production Recommendation | Primary Reason for Prototype Choice |
|-----------|--------------------|--------------------------|------------------------------------|
| Speech Recognition | Vosk (vosk-model-small-en-in-0.4, ~36MB) | Custom Vosk language model or Whisper (server-side) | Real-time streaming; native Android SDK; fully offline; Indian English accent support |
| Intent Classification | MobileBERT TFLite (26MB) + Rule-based hybrid | Expanded dataset; multi-language models | Purpose-built for mobile; ~62ms latency; no server required |
| Backend Framework | FastAPI (Python) | FastAPI with load balancing | Native async + WebSocket; Python ML ecosystem |
| Database | SQLite | PostgreSQL | Zero-configuration; no DBA required; single-file |
| Real-Time Communication | WebSocket (OkHttp / Starlette) | WebSocket with message queue (Redis) | Full-duplex; no broker dependency |
| Mobile Platform | Android Native (Kotlin/Jetpack Compose) | Android with MDM provisioning | Low-cost hardware; native ML SDK; local availability |
| Text-to-Speech | Android Native TTS | Coqui TTS (custom hotel voice) | Pre-installed; zero cost; no additional model required |
| Staff Dashboard | Web-based (HTML/CSS/JS) | React/Vue with role-based auth | Browser-accessible; zero installation; rapid iteration |
| Server Configuration | Single IP/port (SharedPreferences) | MDM-managed certificates | Sufficient for controlled prototype evaluation |

---

### 5.6.1 Speech Recognition: Vosk vs Whisper vs Cloud APIs vs CMU Sphinx

Speech-to-text is the entry point of the entire pipeline, so it is the most consequential component selection. Four candidates were evaluated.

**Vosk** is an open-source offline speech recognition toolkit from Alpha Cephei, supporting over 20 languages with models ranging from 50MB to 1.8GB (Alpha Cephei, 2023). It runs entirely on-device with real-time streaming recognition and no network dependency.

**Whisper**, from OpenAI (Radford et al., 2023), was trained on 680,000 hours of multilingual data and achieves state-of-the-art accuracy. Model sizes range from ~150MB (tiny) to ~6GB (large). While the tiny variant can run on Android, it processes audio in batches after recording ends rather than streaming in real time.

**Google Cloud Speech-to-Text and Amazon Transcribe** are commercial cloud services with excellent accuracy but require constant internet connectivity and charge per minute of audio. Google Cloud STT is billed at $0.006 per 15 seconds (Google Cloud, 2024), creating ongoing costs that are incompatible with this project's constraints.

**CMU Sphinx (PocketSphinx)** is an older offline engine (~30MB) built on pre-deep-learning acoustic models (Huggins-Daines et al., 2006). Development has been largely inactive since 2019, and accuracy is significantly below modern approaches.

**Table 5.3: Speech Recognition Technology Comparison**

| Criteria | Vosk (~36MB) | Whisper (tiny) | Google Cloud STT | CMU Sphinx |
|----------|:---:|:---:|:---:|:---:|
| Offline Capable | Yes | Yes | No | Yes |
| Model Size | ~36MB | 150MB | N/A (cloud) | ~30MB |
| Real-Time Streaming | Yes | No (batch) | Yes | Yes |
| Accuracy (General English) | Good | Good | Excellent | Fair |
| South Asian Accent Robustness | Good (Indian EN model) | Good | Excellent | Poor |
| Native Android SDK | Yes | No (wrapper required) | Yes | Yes |
| Computational Requirements | Low | Moderate | N/A | Very Low |
| Cost | Free | Free | $0.006/15 sec | Free |
| Active Development | Yes | Yes | Yes | Minimal |
| Satisfies NFR-01 (Offline) | ✓ | ✓ | ✗ | ✓ |
| Satisfies NFR-02 (Privacy) | ✓ | ✓ | ✗ | ✓ |

**Selection: Vosk (vosk-model-small-en-in-0.4)**

Vosk was chosen for three reasons. First, it streams recognition in real time — processing audio as the guest speaks rather than waiting for a complete recording. This is essential for keeping end-to-end latency under 5 seconds (NFR-03). Whisper's batch approach adds an extra wait after the guest finishes speaking, which is incompatible with natural conversational interaction.

Second, Vosk provides a native Android SDK with clean Java/Kotlin integration (Alpha Cephei, 2023). Getting Whisper onto Android requires ONNX conversion and a custom inference wrapper, introducing fragility during a research prototype. During early development, the larger Vosk model (1.8GB) was tested and caused app load times exceeding 15 seconds on a budget tablet. The `vosk-model-small-en-in-0.4` variant at ~36MB resolved this. More importantly, this model was trained on Indian English acoustic data, which is a far better acoustic match for Sri Lankan English than any US English model — Sri Lankan English shares similar vowel quality, intonation patterns, and consonant articulation with Indian English. This acoustic alignment is reinforced by Sri Lankan tourism data: India is Sri Lanka's single largest source of international tourists, contributing 416,974 arrivals in 2024 and representing approximately 20% of all international visitors (Sri Lanka Tourism Development Authority, 2024). Selecting a model trained on Indian English therefore directly addresses the dominant guest accent group that the system will encounter in practice.

Third, all audio stays on the device. There is no point in the pipeline where raw voice data leaves the guest's tablet, directly satisfying the privacy requirement (NFR-02).

Cloud solutions were excluded for violating both NFR-01 and NFR-02. CMU Sphinx was excluded due to poor accuracy and inactive development.

**Production pathway:** A custom Vosk language model fine-tuned on hotel vocabulary would reduce the WER for domain-specific terms. Alternatively, server-side Whisper (medium or large) could be deployed within the hotel LAN for better accent robustness while maintaining privacy.

---

### 5.6.2 Intent Classification: MobileBERT vs DistilBERT vs Rasa DIET vs Rule-Based

Intent classification determines how a transcribed guest utterance gets routed to the right department. Four approaches were evaluated.

**Table 5.4: Intent Classification Technology Comparison**

| Criteria | MobileBERT (TFLite) | DistilBERT (TFLite) | Rasa DIET | Rule-Based Only |
|----------|:---:|:---:|:---:|:---:|
| Model Size | ~26MB | ~67MB | ~100MB+ server | <1MB |
| On-Device Inference | Yes | Yes | No (server-side) | Yes |
| Inference Latency (mobile) | ~62ms | ~150ms | N/A | <5ms |
| Classification Accuracy | High (99.06% on Vosk test set) | High | High | Moderate |
| Handles Natural Phrasing Variation | Yes | Yes | Yes | No |
| Requires Running Server | No | No | Yes | No |
| Native TFLite Conversion | Yes | Requires extra steps | Not supported | N/A |
| Purpose-Built for Mobile | Yes | No | No | N/A |
| Satisfies NFR-03 (Latency) | ✓ | Marginal | Dependent on server | ✓ |

**Selection: Hybrid MobileBERT TFLite + Rule-Based Pipeline**

MobileBERT is the only transformer model architecturally designed for resource-constrained mobile devices. It was trained using progressive knowledge distillation from an inverted-bottleneck BERT-LARGE teacher model (Sun et al., 2020), achieving 4.3x compression and 5.5x speedup over BERT-BASE while scoring 77.7 on the GLUE benchmark. Its 26MB TFLite size is less than half that of DistilBERT (~67MB) (Sanh et al., 2019), and ~62ms inference latency makes real-time classification practical on a budget tablet.

DistilBERT offers comparable accuracy but no meaningful size or speed advantage in this mobile context. Rasa DIET (Bunk et al., 2020) was excluded because it requires a persistent server process, conflicting with the on-device processing requirement. A purely rule-based system was insufficient on its own — guests express the same request in too many ways for any keyword list to handle exhaustively.

During prototype development, however, the neural-only approach revealed a recurring issue: the model occasionally produced lower-than-expected confidence on simple, keyword-heavy requests when minor transcription variants appeared. For example, "I need towels" was sometimes classified as `pillow_request` with a confidence of 0.72. This led to the adoption of a **hybrid two-tier pipeline**: rule-based keyword matching as a fast path for unambiguous requests (confidence fixed at 0.99), with MobileBERT as fallback for anything complex or indirect. A minimum confidence threshold of 0.60 ensures that low-confidence neural predictions are rejected rather than incorrectly routed.

The rule-based tier also provides a useful speed advantage — common requests like "extra towels" or "room service" are resolved in under 5ms without ever invoking the neural model.

**Production pathway:** The keyword dictionary in the prototype requires manual upkeep — new phrasings must be added explicitly. In a production system, a larger dataset and continuous learning from real interactions would reduce reliance on manual rules over time.

---

### 5.6.3 Backend Framework: FastAPI vs Flask vs Django vs Express.js

**Table 5.5: Backend Framework Comparison**

| Criteria | FastAPI | Flask | Django | Express.js |
|----------|:---:|:---:|:---:|:---:|
| Native Async Support | Yes | No | Partial (v3.1+) | Yes |
| Native WebSocket Support | Yes (Starlette) | Requires Flask-SocketIO | Requires Django Channels | Requires Socket.IO |
| Automatic API Documentation | Yes (Swagger/OpenAPI) | No | No | No |
| Python ML Ecosystem | Yes | Yes | Yes | No |
| Dependency Footprint | Light | Light | Heavy | Light |
| Request Validation | Built-in (Pydantic) | Manual | Built-in (Forms) | Manual |
| Performance (requests/sec) | High | Moderate | Moderate | High |

**Selection: FastAPI**

FastAPI (Ramírez, 2024) was chosen primarily for its native async support. The server needs to simultaneously manage WebSocket connections from multiple guest room devices and the staff dashboard without blocking. Flask's synchronous model would require Flask-SocketIO, adding a dependency and changing the concurrency model. Django was too heavyweight — the system needs a focused API server and WebSocket hub, not a full web framework with ORM, templating, and an admin panel.

Express.js was excluded because the entire ML stack — model training, evaluation, and the NLU classification pipeline — is Python-based. A Node.js backend would require a second language environment and duplicated business logic.

FastAPI's Pydantic request validation also proved useful during development, automatically rejecting malformed API calls with clear error messages. The auto-generated Swagger docs at `/docs` allowed rapid endpoint testing without extra tooling.

**Production pathway:** A single FastAPI process is sufficient for a controlled research environment. Serving multiple hotel properties would require nginx as a load balancer, multiple worker processes, and a Redis-backed WebSocket message queue for reliability.

---

### 5.6.4 Database: SQLite vs PostgreSQL vs MySQL vs MongoDB

**Table 5.6: Database Technology Comparison**

| Criteria | SQLite | PostgreSQL | MySQL | MongoDB |
|----------|:---:|:---:|:---:|:---:|
| Server Process Required | No | Yes | Yes | Yes |
| Configuration Required | None | Moderate | Moderate | Moderate |
| Concurrent Write Performance | Limited | Excellent | Good | Good |
| Storage Footprint | Minimal (<1MB per hotel) | ~100MB+ | ~200MB+ | ~300MB+ |
| Data Model | Relational | Relational | Relational | Document |
| DBA Expertise Required | None | Yes | Yes | Yes |
| Backup Method | Copy single file | pg_dump | mysqldump | mongodump |

**Selection: SQLite**

SQLite (Hipp, 2024) is more than adequate for a single hotel property generating an estimated 100–200 service requests per day. That volume produces negligible write concurrency — well within SQLite's practical limits. It requires no server process, no installation configuration, and no ongoing administration, directly supporting the minimal IT infrastructure requirement (NFR-08).

The data is naturally relational — requests belong to rooms, staff messages belong to requests — which made document-oriented MongoDB an awkward fit. PostgreSQL and MySQL were not excluded for any technical shortcoming, but because deploying either requires user account setup, network binding, and ongoing maintenance that cannot be guaranteed at small hotels with no IT staff.

**Production pathway:** SQLite's limited concurrent write support becomes a real constraint above 500 rooms or across multiple properties. The schema and FastAPI CRUD layer are designed to be PostgreSQL-compatible with only a driver and connection string change.

---

### 5.6.5 Real-Time Communication: WebSocket vs HTTP Polling vs Server-Sent Events vs MQTT

**Table 5.7: Real-Time Communication Technology Comparison**

| Criteria | WebSocket | HTTP Polling | Server-Sent Events | MQTT |
|----------|:---:|:---:|:---:|:---:|
| Bidirectional | Yes | Simulated | No (server → client only) | Yes |
| Latency | Low | High (polling interval) | Low | Low |
| Connection Overhead | Single handshake | Repeated connections | Single connection | Single connection |
| Requires External Broker | No | No | No | Yes (e.g., Mosquitto) |
| Native Android Support | Yes (OkHttp) | Yes | Limited | Requires library |
| Battery Impact on Tablets | Low | High | Low | Low |

**Selection: WebSocket (OkHttp on Android / Starlette on server)**

WebSocket (Fette and Melnikov, 2011) provides full-duplex, bidirectional communication over a single persistent TCP connection. The system needs bidirectional messaging at three points: guest devices sending requests and receiving status updates, the staff dashboard receiving requests and sending messages, and the server broadcasting changes to all connected clients. This bidirectionality rules out Server-Sent Events, which are server-to-client only.

HTTP polling was excluded because it introduces latency between the event and the notification — a guest waiting for a "request in progress" update would not receive it until the next polling cycle. Repeated connection overhead would also drain tablet battery significantly. MQTT (OASIS, 2019) was excluded because it requires a separate broker process (like Mosquitto), adding infrastructure complexity that contradicts NFR-08 with no practical benefit for a point-to-point LAN scenario.

The prototype uses two WebSocket endpoint types: `/ws/guest/{room_number}` for per-room guest connections and `/ws/dashboard` for staff dashboard connections. Connection registries are maintained in server memory. Guest devices use OkHttp 4.11.0 with exponential backoff reconnection (starting at 2,000ms, doubling per attempt, capped at 30 seconds).

**Production pathway:** In-memory connection state is lost on server restart. A production deployment would use a Redis-backed channel layer to persist connection state and support horizontal scaling.

---

### 5.6.6 Mobile Platform: Android Native vs iOS vs Flutter vs React Native

**Table 5.8: Mobile Platform Comparison**

| Criteria | Android (Kotlin) | iOS (Swift) | Flutter | React Native |
|----------|:---:|:---:|:---:|:---:|
| Device Cost Range (Sri Lanka) | $50–$150 | $300+ (iPad) | Depends on target | Depends on target |
| Local Hardware Availability | High | Limited | Target-dependent | Target-dependent |
| Vosk SDK Support | Native | Native | Community plugin | Community plugin |
| TFLite SDK Support | Native | Core ML (conversion needed) | Community plugin | Community plugin |
| On-Device TTS | Built-in | Built-in | Plugin required | Plugin required |
| ML Inference Performance | Native JNI | Native | Near-native | JS bridge overhead |
| Jetpack Compose / Material 3 | Yes | N/A | Via Flutter widgets | N/A |

**Selection: Android Native (Kotlin 2.0.21 with Jetpack Compose)**

Android was chosen for three reasons tied directly to the deployment context. First, commodity Android tablets are available at $50–$150 from local retailers across Sri Lanka, compared to $300+ for entry-level iPads — directly satisfying NFR-04. Android devices can also be repaired and replaced through local technicians, unlike iOS hardware.

Second, both Vosk and TensorFlow Lite provide first-party native Android SDKs. Community-maintained Flutter and React Native plugins for these libraries introduce version compatibility risks and performance overhead that could undermine core system stability during evaluation.

Third, Jetpack Compose with Material Design 3 provides a reactive UI model that maps naturally to the system's state-driven interface. Recording state, processing state, real-time request list updates, and WebSocket-driven status changes all fit cleanly into Compose's observable state model. This proved especially important in resolving a recomposition issue encountered during development (discussed in Section 6.x).

**Production pathway:** The prototype uses a hardcoded room number and a manually configured server IP in SharedPreferences. Production deployment would use an MDM platform (e.g., VMware Workspace ONE) to remotely provision room numbers and server credentials across all devices without physical access.

---

### 5.6.7 Text-to-Speech: Android Native TTS vs Coqui TTS vs Amazon Polly

**Table 5.9: Text-to-Speech Technology Comparison**

| Criteria | Android Native TTS | Coqui TTS | Amazon Polly |
|----------|:---:|:---:|:---:|
| Additional Storage | None (pre-installed) | ~100–500MB per model | N/A (cloud) |
| Voice Quality | Acceptable | High (natural) | High (natural) |
| Custom Voice Support | No | Yes | Yes |
| Offline Operation | Yes | Yes | No |
| Configuration Required | None | Model download + setup | API key + internet |
| Cost | Free | Free | $4 per 1M characters |

**Selection: Android Native TTS**

Android's built-in TTS engine is pre-installed on all Android devices, requires no additional storage, and needs no configuration. For the purpose of evaluating this prototype — where the research focus is on NLU accuracy and pipeline latency, not voice quality — the native TTS is more than adequate for confirmation messages and status announcements.

**Production pathway:** In a real hotel deployment, voice quality directly affects the guest experience. Coqui TTS (Eren et al., 2021) would allow a custom hotel assistant voice to be deployed on-device or on the hotel server, maintaining offline operation with significantly better audio quality. Amazon Polly was excluded for requiring cloud connectivity (violates NFR-01) and costing $4 per one million characters synthesised (Amazon Web Services, 2024).

---

## 5.7 Summary

This chapter has translated the system requirements into a concrete conceptual architecture, use case model, process flows, data model, and technology stack. The three-layer architecture — guest device handling all AI processing, server managing coordination and persistence, staff dashboard handling operational management — reflects the core design constraints of offline operation and privacy preservation.

The technology evaluation has shown that in every component, the choices are constrained rather than arbitrary. Cloud-based alternatives were consistently excluded for violating the offline and privacy requirements, and heavier frameworks were excluded for the operational overhead they would impose in hotels with no IT support. The `vosk-model-small-en-in-0.4` Indian English model, MobileBERT TFLite, FastAPI, SQLite, and Android Native were each the best fit given the specific deployment context of small Sri Lankan hotels. The following chapter presents the implementation of this architecture and the key engineering decisions made during development.

---

## References

Alpha Cephei (2023) *Vosk offline speech recognition* [Software]. Available at: https://alphacephei.com/vosk/ (Accessed: 8 March 2026).

Amazon Web Services (2024) *Amazon Polly pricing*. Available at: https://aws.amazon.com/polly/pricing/ (Accessed: 8 March 2026).

Bunk, T., Varshneya, D., Vlasov, V. and Nichol, A. (2020) 'DIET: Lightweight language understanding for dialogue systems', *arXiv preprint arXiv:2004.09936*.

CMU Sphinx (2023) *PocketSphinx* [Software repository]. Available at: https://github.com/cmusphinx/pocketsphinx (Accessed: 8 March 2026).

Eren, G., Gölge, E. and the Coqui TTS Team (2021) *Coqui TTS: A deep learning toolkit for text-to-speech* [Software]. Available at: https://github.com/coqui-ai/TTS (Accessed: 8 March 2026).

Fette, I. and Melnikov, A. (2011) *The WebSocket protocol*, RFC 6455. Internet Engineering Task Force (IETF).

Google (2024a) *TensorFlow Lite: Machine learning for mobile and edge devices*. Available at: https://www.tensorflow.org/lite (Accessed: 8 March 2026).

Google (2024b) *Jetpack Compose*. Available at: https://developer.android.com/jetpack/compose (Accessed: 8 March 2026).

Google Cloud (2024) *Cloud Speech-to-Text pricing*. Available at: https://cloud.google.com/speech-to-text/pricing (Accessed: 8 March 2026).

Hipp, D.R. (2024) *SQLite* [Software]. Available at: https://www.sqlite.org (Accessed: 8 March 2026).

Huggins-Daines, D. et al. (2006) 'Pocketsphinx: A free, real-time continuous speech recognition system for hand-held devices', in *Proceedings of ICASSP 2006*, Vol. 1, pp. I-185–I-188.

OASIS (2019) *MQTT version 5.0*. OASIS Standard.

Radford, A. et al. (2023) 'Robust speech recognition via large-scale weak supervision', in *Proceedings of ICML 2023*, PMLR 202, pp. 28492–28518.

Ramírez, S. (2024) *FastAPI* [Software]. Available at: https://fastapi.tiangolo.com (Accessed: 8 March 2026).

Sanh, V. et al. (2019) 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter', *arXiv preprint arXiv:1910.01108*.

Sun, Z. et al. (2020) 'MobileBERT: a compact task-agnostic BERT for resource-limited devices', in *Proceedings of ACL 2020*, pp. 2158–2170.
