# CHAPTER 5: ANALYSIS

## 5.1 Introduction

This chapter takes the requirements from Chapter 4 and works through how they shape the actual system design. It covers the overall system structure, the key interactions between users and the system (use cases), how the data is stored, and which technologies were chosen and why. The technology decisions are probably the most important part of this chapter — because the constraints of this project (no internet, budget hardware, privacy requirements, no IT staff on site) ruled out most of the obvious choices, and the reasoning behind each decision matters for understanding the final design.

---

## 5.2 High-Level System Conceptual View

The system is split into three layers, all running over the hotel's local Wi-Fi. There is no internet connection needed for anything the system does day-to-day.

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

The key design decision here is that all the AI work — speech recognition and intent classification — happens on the guest's tablet. The server is kept deliberately lightweight: it stores requests, maps them to the right department, and pushes real-time updates to connected clients. That said, it is important to be honest about what this means for reliability in the current prototype. If the server is temporarily unreachable, the STT and NLU still work fine on the device, but the guest cannot actually submit their request. The app tells them to use the room's land line phone instead. There is no local queue that saves the request and retries once the server comes back — if submission fails, the request is simply lost. This is a known limitation of the prototype and is discussed as future work in Section 11.2.13. What is guaranteed regardless of server availability is that raw audio never leaves the tablet at any point, which directly satisfies the privacy requirement (NFR-02).

---

## 5.3 Use Case Model

There are two actors: the **Guest**, who interacts through voice and touch on the Android tablet, and the **Staff Member**, who uses the web dashboard to manage requests.

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

The voice pipeline handles everything from when the guest presses the microphone button through to the request appearing on the staff dashboard. The specific values in the flow below come directly from the implementation: a 1,500ms silence timeout to detect end of speech, a maximum recording duration of 10,000ms, a minimum NLU confidence threshold of 0.60, up to 2 retries on low-confidence results, and a maximum tokenisation length of 32 tokens.

**Figure 5.3: Voice Request Pipeline**

```
Guest presses
microphone button
       |
       v
[Start Audio Recording]
16kHz, 16-bit PCM, mono
Dynamic buffer (min × 2)
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
hotel_mobilebert.tflite (26MB)
Softmax over 18 intent classes
       |
       v
[Confidence check]
       |
       +---> < 0.60? --Yes--> TTS: "I couldn't quite understand, please try again"
       |                           |
       |                    Retry (max 2 attempts)
       |                           |
       |                    Still < 0.60? --> TTS: "Sorry, could not understand" --> End
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
transcription, and intent
       |
       v
[Server Processing]
Map intent → department (via intent_department_mapping)
Store in SQLite
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

The database has five tables. Three of them — `rooms`, `departments`, and `intent_department_mapping` — are configuration tables that get seeded once when the server first starts up. The two tables that change during normal operation are `requests` (the main entity) and `staff_messages`, which sits in a one-to-many relationship with it, since one request can have multiple messages from staff.

**Figure 5.5: Entity-Relationship Diagram**

*(See attached ER diagram — Figure 5.5)*

```
+-------------------+     +------------------------+
|      rooms        |     |      departments       |
+-------------------+     +------------------------+
| PK  id  INTEGER   |     | PK  id  INTEGER        |
|     room_number   |     |     name  TEXT UNIQUE  |
|     floor         |     |     description  TEXT  |
|     room_type     |     +------------------------+
+-------------------+               |
        |                           |
        |          +--------------------------------+
        |          |   intent_department_mapping    |
        |          +--------------------------------+
        |          | PK  id          INTEGER        |
        |          |     intent      TEXT UNIQUE    |
        |          | FK  department_name  TEXT      |
        |          +--------------------------------+
        |                           |
        v                           v
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

**requests** is where every guest service request ends up. The `status` column tracks the lifecycle: `pending` → `in_progress` → `completed` (or `cancelled`). The `intent` field is classified on the guest's device by the NLU pipeline and sent up to the server as part of the HTTP POST. The server then looks up `intent_department_mapping` to figure out which department should handle it and writes the result into `department`. The `rating` column is left null unless the guest chooses to rate the service after it is completed.

**staff_messages** holds any messages that staff send to a guest room. Each message is tied to a specific request, so the conversation always stays in context.

**rooms** and **departments** are seeded at server startup and stay static across the deployment. The `intent_department_mapping` table is what lets the routing logic stay flexible — it maps each of the 18 intent categories to one of five departments (Housekeeping, Room Service, Maintenance, Front Desk, Concierge). If the hotel adds a new intent later, only this table needs updating, not the application code.

SQLite was chosen as the database for this prototype. It needs no server process, no configuration, and no ongoing maintenance — practical for small hotels where there is no dedicated IT staff. For larger deployments, the schema is designed to be PostgreSQL-compatible with only a driver and connection string change. This is discussed further in Section 5.6.4.

---

## 5.6 Technology Evaluation and Selection

Every component in this system was evaluated against four constraints that came directly from the requirements in Chapter 4: offline capability (NFR-01), privacy (NFR-02), low response latency (NFR-03), and low hardware cost (NFR-04).

It is also worth being transparent about the prototype-vs-production distinction here. Some choices — like SQLite over PostgreSQL — are practical for a research prototype but would need to change at scale. Where that is the case, the production alternative is identified explicitly. Table 5.2 gives the full picture of what was selected and why.

**Table 5.2: Technology Stack Summary**

| Component | Prototype Selection | Production Recommendation | Primary Reason for Prototype Choice |
|-----------|--------------------|--------------------------|------------------------------------|
| Speech Recognition | Vosk (vosk-model-small-en-in-0.4, ~36MB) | Custom Vosk language model fine-tuned on hospitality vocabulary | Real-time streaming; native Android SDK; fully offline; Indian English accent support |
| Intent Classification | MobileBERT TFLite (`hotel_mobilebert.tflite`, 26MB) + Rule-based hybrid | Expanded dataset; multi-language models | Purpose-built for mobile; ~62ms latency; no server required |
| Backend Framework | FastAPI (Python) | FastAPI with load balancing | Native async + WebSocket; Python ML ecosystem |
| Database | SQLite | PostgreSQL | Zero-configuration; no DBA required; single-file |
| Real-Time Communication | WebSocket (OkHttp / Starlette) | WebSocket with message queue (Redis) | Full-duplex; no broker dependency |
| Mobile Platform | Android Native (Kotlin/Jetpack Compose) | Android with MDM provisioning | Low-cost hardware; native ML SDK; local availability |
| Text-to-Speech | Android Native TTS | Piper TTS (custom hotel voice model) | Pre-installed; zero cost; no additional model required |
| Staff Dashboard | Web-based (HTML/CSS/JS) | React/Vue with role-based auth | Browser-accessible; zero installation; rapid iteration |
| Server Configuration | Single IP/port (SharedPreferences) | MDM-managed certificates | Sufficient for controlled prototype evaluation |

---

### 5.6.1 Speech Recognition: Vosk vs Whisper vs Cloud APIs vs CMU Sphinx

Speech-to-text is the entry point of the whole pipeline, so this was the most important component decision. Four options were evaluated.

**Vosk** is an open-source offline speech recognition toolkit from Alpha Cephei. It supports over 20 languages with model sizes ranging from 50MB to 1.8GB (Alpha Cephei, 2023) and runs entirely on the device with real-time streaming — no network needed.

**Whisper** from OpenAI (Radford et al., 2023) was trained on 680,000 hours of multilingual audio and achieves very strong accuracy. Models range from ~150MB (tiny) to ~6GB (large). The tiny variant can technically run on Android, but it works in batch mode — it waits until the recording is done before processing, rather than transcribing in real time.

**Google Cloud Speech-to-Text and Amazon Transcribe** are commercial cloud services with excellent accuracy, but they both require a constant internet connection and charge per minute of audio. Google Cloud STT costs $0.006 per 15 seconds (Google Cloud, 2024), which is an ongoing cost that simply does not work for the target hotels.

**CMU Sphinx (PocketSphinx)** is an older offline engine (~30MB) built on pre-deep-learning acoustic models (Huggins-Daines et al., 2006). It has seen very little development since 2019 and its accuracy is noticeably below modern systems.

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

Vosk was chosen for three reasons. The first is real-time streaming. Vosk processes audio as the guest speaks, so there is no waiting period after they finish. Whisper works the other way — it waits for the full recording and then processes it, which adds noticeable latency and makes the interaction feel unnatural. This is a problem for keeping end-to-end response time under 5 seconds (NFR-03).

The second reason is that Vosk has a proper native Android SDK with straightforward Kotlin integration (Alpha Cephei, 2023). Getting Whisper running on Android would require converting the model to ONNX format and building a custom inference wrapper — extra complexity that could easily break things on a research prototype. Vosk also offers a range of model sizes, and the `vosk-model-small-en-in-0.4` variant at ~36MB was chosen specifically to stay within the memory constraints of a budget tablet (discussed in Chapter 7). Beyond the size, this model was trained on Indian English acoustic data, which matters a lot for this deployment. Sri Lankan English is acoustically similar to Indian English — shared vowel quality, intonation patterns, and consonant articulation — so a US English model would be a poor acoustic fit. The tourism angle reinforces this: India is Sri Lanka's largest source of international visitors, with 416,974 arrivals in 2024 making up roughly 20% of all international tourists (Sri Lanka Tourism Development Authority, 2024). The `vosk-model-small-en-in-0.4` model directly addresses the most common guest accent the system will encounter.

The third reason is privacy. Audio never leaves the tablet at any point in the pipeline, which satisfies NFR-02 directly. Cloud options were ruled out for violating both NFR-01 and NFR-02. CMU Sphinx was ruled out for its poor accuracy and the fact that it is no longer actively maintained.

**Production pathway:** A custom Vosk language model fine-tuned on hospitality vocabulary — room service items, department names, South Asian English pronunciation patterns — would reduce the WER for domain-specific terms while keeping STT fully on-device.

---

### 5.6.2 Intent Classification: MobileBERT vs DistilBERT vs Rasa DIET vs Rule-Based

Once the guest's speech is transcribed, the system needs to figure out what they actually want — and route it to the right department. Four approaches were evaluated.

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

MobileBERT is the only transformer model that was specifically designed for mobile devices with limited memory and processing power. It was trained using progressive knowledge distillation — essentially learning from a much larger BERT-LARGE teacher model — achieving 4.3× compression and 5.5× speedup over standard BERT-BASE while still scoring 77.7 on the GLUE benchmark (Sun et al., 2020). At 26MB in TFLite format, it is less than half the size of DistilBERT (~67MB) (Sanh et al., 2019), and its ~62ms inference time makes real-time classification on a budget tablet actually practical.

DistilBERT is capable, but at ~67MB and ~150ms inference time it is more than twice the size of MobileBERT and significantly slower on mobile hardware — a weaker fit for budget tablets under the NFR-04 constraint. Rasa DIET (Bunk et al., 2020) was ruled out because it needs a persistent server process to run, which goes against the goal of keeping all NLU on the device. A purely rule-based system was also not enough on its own — guests say the same thing in too many different ways for a keyword list to cover reliably.

That said, a neural-only approach has a known limitation in this pipeline: because Vosk does not always produce identical transcriptions for the same spoken phrase, the same intent can arrive at the classifier with slightly different surface forms. A model trained on clean text may assign lower confidence to a valid request simply because the transcription contains a minor variant. This is a well-recognised challenge when chaining STT and NLU components (Zhang, L. et al., 2022), and it creates a risk of misclassification even on straightforward requests. To address this, a **hybrid two-tier pipeline** was designed: a fast rule-based keyword matching layer handles clear-cut, high-frequency requests with a fixed confidence of 0.99, with MobileBERT only invoked for anything more indirect or ambiguous. If the model's confidence still falls below 0.60 after up to two retries, the request is rejected rather than incorrectly routed. The full behaviour of this pipeline in practice is discussed in Chapter 7.

As a side benefit, the rule-based layer is very fast — common requests like "extra towels" or "room service" are resolved in under 5ms without ever invoking the neural model.

**Production pathway:** The keyword dictionary needs manual upkeep as new phrasings are encountered. In a production system, logging real interactions and periodically retraining the model would reduce the need for manual rule additions over time.

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

FastAPI (Ramírez, 2024) was chosen mainly because it handles async natively. The server has to keep WebSocket connections open with multiple guest room tablets and the staff dashboard at the same time, while also handling HTTP requests — all without blocking. Flask's synchronous design would need Flask-SocketIO bolted on to achieve this, which adds a dependency and complicates the concurrency model. Django was overkill — the system just needs a focused API server and WebSocket hub, not a full web framework with an ORM, templating engine, and admin panel built in.

Express.js was not suitable because the entire ML stack — training, evaluation, and the NLU pipeline — is Python-based. Adding a Node.js backend would mean maintaining two separate language environments and duplicating business logic.

One practical benefit that came up during development: FastAPI's Pydantic validation automatically rejects malformed API calls with clear error messages, which made debugging much faster. The auto-generated Swagger docs at `/docs` also meant endpoint testing did not need any extra tooling.

**Production pathway:** A single FastAPI process is fine for a controlled research environment. Scaling to multiple hotel properties would need nginx as a load balancer, multiple worker processes, and a Redis-backed message queue to make WebSocket state reliable across restarts.

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

A small hotel generating around 100–200 service requests per day does not push SQLite anywhere near its limits (Hipp, 2024). Write concurrency at that volume is negligible. SQLite needs no server process, no installation, and no ongoing admin — which is exactly what is needed for hotels with no dedicated IT staff.

The data itself is naturally relational: requests belong to rooms, messages belong to requests. MongoDB's document model would be an awkward fit for this. PostgreSQL and MySQL are both technically capable options, but deploying either one at a small hotel requires setting up user accounts, configuring network binding, and keeping it maintained — none of which can be assumed when there is no IT staff on site.

**Production pathway:** SQLite serialises writes, which is sufficient for small hotel deployments but would become a bottleneck under sustained concurrent load — for example, during peak periods when many rooms submit requests simultaneously. The schema and FastAPI CRUD layer are already written to be PostgreSQL-compatible, so switching would only require a driver and connection string change.

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

WebSocket (Fette and Melnikov, 2011) keeps a single persistent TCP connection open and allows both sides to send messages at any time. The system needs this in three places: guest tablets sending requests and receiving status updates, the staff dashboard receiving requests and sending messages back, and the server pushing changes to all connected clients. This bidirectional requirement rules out Server-Sent Events straight away — they only go one way (server to client).

HTTP polling would work technically, but the latency is a problem. A guest waiting to hear that their request is "in progress" would not find out until the next polling cycle. Repeated connection setup would also drain the tablet battery much faster than a persistent WebSocket connection. MQTT (OASIS, 2019) was also considered but needs a separate broker process like Mosquitto running on the network — that is unnecessary infrastructure complexity for a simple point-to-point LAN scenario.

The server maintains separate WebSocket endpoints for guest devices and the staff dashboard, with connection state held in memory. The Android client handles reconnection automatically using exponential backoff, capping at a maximum delay to avoid indefinite disconnection.

**Production pathway:** Because connection state is in-memory, it is lost if the server restarts. For a single-property deployment, a Redis Pub/Sub layer would persist WebSocket state and allow the server to restart without dropping all connected clients. For multi-property deployments — where events need to be consumed by multiple downstream systems simultaneously — Apache Kafka (Kreps et al., 2011) is a stronger fit, as it provides persistent message replay and decouples event producers from consumers entirely. Both approaches are discussed as future work in Section 11.2.13.

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

Android came down to three practical reasons. First, commodity Android tablets cost $50–$150 at local retailers across Sri Lanka. Entry-level iPads start at over $300 — that is more than twice the price per room, which makes iOS unviable for the target hotels (NFR-04). Android hardware is also easier to repair and replace locally.

Second, both Vosk and TensorFlow Lite have first-party native Android SDKs. Flutter and React Native do have community plugins for these libraries, but community plugins for ML inference bring version compatibility risks that could cause hard-to-diagnose failures during evaluation. Keeping to the native SDKs avoids that uncertainty.

Third, Jetpack Compose with Material Design 3 is a good fit for a UI that is driven by changing state — recording state, processing state, live request list updates, and WebSocket-driven status changes all map cleanly onto Compose's observable state model. This design alignment carries through to the implementation, where Compose's recomposition behaviour becomes directly relevant, as discussed in Chapter 6.

**Production pathway:** The prototype uses a fixed room number and a server IP stored manually in SharedPreferences. A real deployment would use an MDM platform like VMware Workspace ONE to remotely configure room numbers and server credentials across all tablets without touching each device physically.

---

### 5.6.7 Text-to-Speech: Android Native TTS vs Piper TTS vs Amazon Polly

**Table 5.9: Text-to-Speech Technology Comparison**

| Criteria | Android Native TTS | Piper TTS | Amazon Polly |
|----------|:---:|:---:|:---:|
| Additional Storage | None (pre-installed) | ~50–100MB per voice model | N/A (cloud) |
| Voice Quality | Acceptable | High (natural) | High (natural) |
| Custom Voice Support | No | Yes | Yes |
| Offline Operation | Yes | Yes | No |
| Configuration Required | None | Model download + setup | API key + internet |
| Cost | Free | Free | $4 per 1M characters |

**Selection: Android Native TTS**

Android's built-in TTS engine is already on every Android device. No extra storage, no setup, no cost. For this prototype — where the research question is about NLU accuracy and pipeline latency rather than voice quality — the native engine is perfectly good enough for reading back confirmations and status announcements.

**Production pathway:** Voice quality matters in a real hotel, since it shapes how guests perceive the assistant. Piper TTS (Hansen, 2023) is an open-source neural TTS engine designed specifically for edge and embedded deployment — it runs on low-power hardware (including Raspberry Pi) and produces natural-sounding voices from compact models (~50–100MB). It is actively maintained and used in production by the Home Assistant and Rhasspy smart home platforms, which share similar on-device, offline constraints. A custom English voice model fine-tuned for hotel-specific phrasing could be deployed on the Android device itself, keeping TTS fully on-device and offline. Amazon Polly was not suitable because it requires cloud connectivity (violates NFR-01) and costs $4 per million characters synthesised (Amazon Web Services, 2024).

---

## 5.7 Summary

This chapter has laid out how the requirements from Chapter 4 became a concrete design. The three-layer architecture — AI processing on the guest device, coordination and storage on the server, operational management on the staff dashboard — is a direct result of the offline and privacy constraints. Every component decision followed the same logic: cloud options were consistently ruled out for violating NFR-01 and NFR-02, heavier frameworks were ruled out for the operational overhead they would add in hotels with no IT support, and the final choices — `vosk-model-small-en-in-0.4`, MobileBERT TFLite, FastAPI, SQLite, and Android Native — were each the best fit for the specific context of small Sri Lankan hotels.

The following chapter covers the actual implementation of this architecture and the key engineering decisions that came up along the way.
