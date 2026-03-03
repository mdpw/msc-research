# THESIS TEMPLATE
# Copy this structure into Microsoft Word and fill in each section

---

# COVER PAGE

**[University Name]**
**Faculty of Computing / School of Computing**

**Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models**

A thesis submitted in partial fulfilment of the requirements for the degree of
**Master of Science in [Your Programme Name]**

**[Your Full Name]**
**Registration No: [Your Reg Number]**

**Supervisor: [Supervisor Name]**

**[Month] 2025**

---

# DECLARATION

I declare that this thesis is my own work and has not been submitted in any form for another degree or diploma at any university or other institution of tertiary education. Information derived from the published or unpublished work of others has been acknowledged in the text and a list of references is given.

[Your Name]
[Date]

---

# ABSTRACT
*(Write this LAST — 250-300 words summarising the entire thesis)*

The hospitality industry in developing economies faces significant challenges in modernising guest service operations due to the high costs of cloud-based voice assistants, privacy concerns, and unreliable internet connectivity. This research presents the design, development, and evaluation of a low-cost, offline-capable voice assistant system for hotel room service operations, targeting small to mid-sized hotels in Sri Lanka. The system employs on-device speech-to-text processing using Vosk (205MB model) and a fine-tuned MobileBERT intent classifier (92% accuracy across 18 hotel service categories) deployed as a TensorFlow Lite model on commodity Android tablets. A hybrid natural language understanding pipeline combines rule-based keyword matching with neural model inference to achieve robust intent classification. The backend architecture utilises FastAPI with SQLite for request management and WebSocket for real-time bidirectional communication between guest devices and a staff management dashboard. Evaluation results demonstrate [X]% word error rate for speech recognition on hotel-domain utterances with South Asian English speakers, [X]ms average end-to-end latency, and an estimated per-room deployment cost of $[X], representing a [X]% reduction compared to commercial cloud-based alternatives. The system addresses critical research gaps at the intersection of edge AI, privacy-preserving computing, and hospitality technology, demonstrating that effective voice-based service automation is achievable without cloud dependency using commodity hardware. This research contributes a replicable model for hotels in developing economies seeking to enhance guest services while preserving privacy and minimising infrastructure costs.

**Keywords:** Voice Assistant, Edge AI, Offline Speech Recognition, MobileBERT, Hotel Technology, Natural Language Understanding, Privacy-Preserving AI

---

# ACKNOWLEDGEMENTS

*(Thank your supervisor, family, friends, anyone who helped with testing, etc.)*

---

# TABLE OF CONTENTS

*(Auto-generate in Word: References > Table of Contents)*

---

# LIST OF FIGURES

*(Auto-generate in Word after inserting figures with captions)*

---

# LIST OF TABLES

*(Auto-generate in Word after inserting tables with captions)*

---

# LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| ASR | Automatic Speech Recognition |
| BERT | Bidirectional Encoder Representations from Transformers |
| CRUD | Create, Read, Update, Delete |
| DHCP | Dynamic Host Configuration Protocol |
| GDPR | General Data Protection Regulation |
| GLUE | General Language Understanding Evaluation |
| HTTP | Hypertext Transfer Protocol |
| IoT | Internet of Things |
| LAN | Local Area Network |
| NLU | Natural Language Understanding |
| PCM | Pulse Code Modulation |
| REST | Representational State Transfer |
| SQL | Structured Query Language |
| STT | Speech-to-Text |
| TFLite | TensorFlow Lite |
| TTS | Text-to-Speech |
| UI | User Interface |
| UX | User Experience |
| VAD | Voice Activity Detection |
| WER | Word Error Rate |
| WebSocket | Full-duplex communication protocol |

---

# CHAPTER 1: INTRODUCTION

## 1.1 Background

*(2-3 paragraphs about the hospitality industry, technology adoption, and voice assistants)*

The hospitality industry is a cornerstone of the Sri Lankan economy, contributing approximately X% to the national GDP. Hotels constantly seek innovative approaches to enhance guest experience while managing operational costs...

The emergence of voice-based virtual assistants, such as Amazon Alexa for Hospitality and Google Nest for Hotels, has introduced new possibilities for automating guest service operations...

However, these solutions rely heavily on cloud infrastructure, require continuous internet connectivity, and raise significant privacy concerns, making them impractical for the majority of hotels in developing countries like Sri Lanka...

## 1.2 Research Problem

*(1-2 paragraphs clearly stating the problem)*

Small to mid-sized hotels in Sri Lanka face multiple barriers to adopting voice-based service automation: (1) high recurring costs of cloud-based solutions, (2) unreliable internet connectivity in tourism areas, (3) privacy concerns with cloud-transmitted guest voice data, and (4) lack of localised solutions for South Asian English accents...

## 1.3 Research Question

How effective is a commodity hardware-based, offline voice assistant system in reducing operational costs and improving service efficiency compared to traditional room service methods in Sri Lankan hotels?

## 1.4 Research Objectives

### 1.4.1 Primary Objectives
1. Design and develop a fully functional offline voice assistant system for hotel room service operations using commodity Android devices.
2. Implement on-device speech-to-text using Vosk and intent classification using a fine-tuned MobileBERT model.
3. Build a real-time hotel management backend with automated department routing and staff dashboard.
4. Evaluate the system's accuracy, latency, and cost-effectiveness compared to cloud-based alternatives.

### 1.4.2 Secondary Objectives
5. Create a hotel-specific intent classification dataset covering 18 service request categories.
6. Implement voice-based interaction for the complete request lifecycle.
7. Demonstrate privacy-preserving edge computing for hospitality AI.

## 1.5 Scope and Limitations

*(Define what your research covers and what it does NOT cover)*

This research focuses on English-language voice interactions for hotel room service requests. Multi-language support (Sinhala, Tamil) is identified as future work. The system is evaluated in a controlled environment rather than a live hotel deployment...

## 1.6 Research Contribution

*(2-3 sentences on what is novel about your work)*

This research presents the first integrated, offline-capable voice assistant system specifically designed for hotel operations in developing economies, combining on-device STT, fine-tuned NLU, and a real-time staff management system on commodity Android hardware...

## 1.7 Thesis Organisation

This thesis is organised as follows: Chapter 2 reviews related literature. Chapter 3 describes the methodology. Chapter 4 presents the system design and architecture. Chapter 5 details the implementation. Chapter 6 presents evaluation results. Chapter 7 discusses findings and limitations. Chapter 8 concludes with future work.

---

# CHAPTER 2: LITERATURE REVIEW

## 2.1 Voice Assistants in the Hospitality Industry

*(Review papers A1-A5 from RESEARCH_ANALYSIS.md)*

### 2.1.1 Commercial Voice Assistant Deployments in Hotels

Buhalis and Moldavska (2021) conducted a qualitative study examining voice-based AI digital assistants in hotel environments through 28 semi-structured interviews...

Amazon's Alexa for Hospitality platform represents the most widely deployed commercial solution...

### 2.1.2 Guest Experience and Adoption Factors

Buhalis and Moldavska (2022) further investigated VA-enabled hotel-guest interactions, documenting use cases including room controls, room service ordering, and wake-up alarms...

### 2.1.3 Privacy Concerns in Hotel Voice Technology

*(Discuss privacy issues identified in literature — Purdue study, GDPR implications)*

## 2.2 Offline and Edge-Based Speech Recognition

*(Review papers B1-B5)*

### 2.2.1 Vosk Offline Speech Recognition

Alpha Cephei's Vosk framework provides lightweight, offline-capable speech recognition supporting multiple languages...

### 2.2.2 Whisper and Large-Scale ASR Models

Radford et al. (2023) presented Whisper, achieving state-of-the-art accuracy across 99 languages trained on 680,000 hours of data...

### 2.2.3 Edge Computing for Speech Processing

*(Discuss edge ASR research — Raspberry Pi deployments, IoT speech control)*

## 2.3 Small-Scale NLU Models for Resource-Constrained Devices

*(Review papers C1-C5)*

### 2.3.1 BERT and Knowledge Distillation

Devlin et al. (2019) introduced BERT, which revolutionised NLU tasks through bidirectional pre-training...

### 2.3.2 MobileBERT for Mobile Deployment

Sun et al. (2020) proposed MobileBERT, achieving 4.3x compression and 5.5x speedup over BERT-BASE while retaining competitive performance...

### 2.3.3 DistilBERT and Alternative Approaches

Sanh et al. (2019) introduced DistilBERT, retaining 97% of BERT's accuracy while being 40% smaller and 60% faster...

### 2.3.4 Rasa DIET Architecture

Bunk et al. (2020) presented DIET (Dual Intent and Entity Transformer), a multi-task architecture for intent classification and entity recognition...

## 2.4 Privacy-Preserving AI in Service Industries

*(Review papers D1-D4)*

### 2.4.1 Edge Intelligence Frameworks

Shi et al. (2016) established the theoretical foundation for edge computing, advocating for data processing at the network edge...

### 2.4.2 On-Device AI Model Deployment

*(Discuss quantisation, pruning, TFLite conversion for mobile deployment)*

## 2.5 Task-Oriented Dialogue Systems

*(Review papers E1-E4)*

### 2.5.1 Intent Classification Datasets and Methods

Larson et al. (2022) surveyed intent classification datasets for task-oriented dialogue, identifying gaps in domain-specific dataset availability...

## 2.6 Technology Adoption in Sri Lankan Tourism

*(Review Sri Lanka-specific research)*

### 2.6.1 Digital Transformation in Sri Lankan Hotels

*(Discuss the Connecting the Connected paper and Sri Lanka's digital tourism readiness)*

## 2.7 Research Gap Analysis

*(Summarise the 5 gaps identified in RESEARCH_ANALYSIS.md)*

Table 2.1: Comparison Matrix of Existing Solutions

| Solution | Offline | On-Device | Hospitality | Low-Cost | Privacy | Staff Integration | End-to-End |
|----------|---------|-----------|-------------|----------|---------|-------------------|------------|
| Alexa for Hospitality | No | No | Yes | No | Partial | Yes | Yes |
| Vosk Edge STT | Yes | Yes | No | Yes | Yes | No | No |
| MobileBERT | Yes | Yes | No | Yes | Yes | No | No |
| Rasa DIET | Partial | No | No | No | Partial | No | Partial |
| **This Research** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### 2.7.1 Gap 1: No Integrated End-to-End Offline System for Hospitality
### 2.7.2 Gap 2: No On-Device Hospitality-Specific NLU
### 2.7.3 Gap 3: No Privacy-Preserving Voice Processing in Hotels
### 2.7.4 Gap 4: No Real-Time Bidirectional Guest-Staff Communication
### 2.7.5 Gap 5: No Cost-Effective Deployment on Commodity Hardware

## 2.8 Summary

*(1 paragraph summarising the literature review and justifying your research)*

---

# CHAPTER 3: METHODOLOGY

## 3.1 Research Approach

*(Describe Design Science Research methodology or Prototyping approach)*

This research follows a Design Science Research (DSR) methodology, which involves the creation and evaluation of IT artefacts intended to solve identified organisational problems...

## 3.2 System Requirements

### 3.2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | System shall convert guest voice input to text offline | Must |
| FR-02 | System shall classify guest intent into 18 categories | Must |
| FR-03 | System shall route requests to appropriate department | Must |
| FR-04 | System shall provide voice confirmation before submission | Must |
| FR-05 | System shall display request status in real-time | Must |
| FR-06 | System shall allow voice-based request cancellation | Should |
| FR-07 | System shall support guest rating of completed requests | Should |
| FR-08 | System shall enable staff-to-guest messaging | Should |
| FR-09 | System shall support dark/light theme modes | Could |
| FR-10 | System shall support multiple network profiles | Could |

### 3.2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | End-to-end response latency | < 5 seconds |
| NFR-02 | STT accuracy (WER) | < 20% |
| NFR-03 | Intent classification accuracy | > 85% |
| NFR-04 | System availability | 99% uptime on LAN |
| NFR-05 | Privacy | Zero external data transmission |
| NFR-06 | Hardware cost per room | < $150 USD |

## 3.3 Dataset Preparation

### 3.3.1 Intent Category Design

*(Explain how you identified the 18 intent categories — hotel operations analysis)*

### 3.3.2 Data Collection and Curation

*(Describe how the 4,971 training examples were created)*

Table 3.1: Dataset Distribution by Intent Category

| Intent | Training Samples | Percentage |
|--------|-----------------|------------|
| towel_request | 508 | 10.2% |
| room_cleaning | 412 | 8.3% |
| food_order | 337 | 6.8% |
| toiletries_request | 333 | 6.7% |
| pillow_request | 306 | 6.2% |
| temperature_control | 304 | 6.1% |
| blanket_request | 260 | 5.2% |
| maintenance | 256 | 5.2% |
| laundry_service | 247 | 5.0% |
| concierge_general | 244 | 4.9% |
| concierge_taxi | 237 | 4.8% |
| wake_up_call | 237 | 4.8% |
| do_not_disturb | 237 | 4.8% |
| lighting_control | 226 | 4.5% |
| noise_complaint | 214 | 4.3% |
| emergency | 214 | 4.3% |
| checkout_billing | 208 | 4.2% |
| misc_request | 192 | 3.9% |
| **Total** | **4,971** | **100%** |

### 3.3.3 Data Augmentation and Quality

*(Describe variations: typos, abbreviations, colloquial phrasing)*

## 3.4 Model Training

### 3.4.1 MobileBERT Fine-Tuning

*(Describe the training process)*

Table 3.2: Model Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | google/mobilebert-uncased |
| Max Sequence Length | 32 tokens |
| Batch Size | 32 |
| Learning Rate | 3e-5 |
| Epochs | 8 |
| Optimizer | AdamW |
| Train/Validation Split | 85% / 15% (stratified) |
| Training Samples | 4,226 |
| Validation Samples | 746 |

### 3.4.2 TensorFlow Lite Conversion

*(Describe the model conversion process for on-device deployment)*

## 3.5 Evaluation Strategy

### 3.5.1 NLU Model Evaluation
### 3.5.2 STT Word Error Rate Evaluation
### 3.5.3 System Latency Measurement
### 3.5.4 Cost Analysis

---

# CHAPTER 4: SYSTEM DESIGN AND ARCHITECTURE

## 4.1 System Overview

*(High-level description with architecture diagram)*

**[INSERT Figure 4.1: System Architecture Diagram]**

```
Guest Room (Android)          Hotel Server (LAN)         Staff (Browser)
+-----------------+          +------------------+       +----------------+
| Mic -> VAD      |          | FastAPI Backend   |       | Web Dashboard  |
| Vosk STT        |--HTTP--->| SQLite Database   |<-WS-->| Dept Queues    |
| MobileBERT NLU  |          | WebSocket Hub     |       | Status Mgmt    |
| Android TTS     |<---WS----| Dept Routing      |       | Messaging      |
+-----------------+          +------------------+       +----------------+
```

## 4.2 Android Application Architecture

### 4.2.1 Component Overview

**[INSERT Figure 4.2: Android App Component Diagram]**

| Component | File | Responsibility |
|-----------|------|----------------|
| MainActivity | MainActivity.kt | UI, lifecycle, coordination |
| VoskService | VoskService.kt | On-device speech-to-text |
| NLUService | NLUService.kt | Intent classification (rules + MobileBERT) |
| AudioRecorder | AudioRecorder.kt | Audio capture with VAD |
| ApiService | ApiService.kt | HTTP communication with server |
| WebSocketService | WebSocketService.kt | Real-time updates |
| ServerConfig | ServerConfig.kt | Network profile management |

### 4.2.2 Voice Processing Pipeline

**[INSERT Figure 4.3: Voice Processing Pipeline Flowchart]**

### 4.2.3 Hybrid NLU Pipeline

*(Explain the two-tier classification: rule-based keywords at 0.99 confidence -> MobileBERT model fallback)*

**[INSERT Figure 4.4: Hybrid NLU Pipeline Diagram]**

### 4.2.4 User Interface Design

**[INSERT Figure 4.5: App Screenshot - Main Screen (Light Mode)]**
**[INSERT Figure 4.6: App Screenshot - Main Screen (Dark Mode)]**
**[INSERT Figure 4.7: App Screenshot - Voice Confirmation Dialog]**
**[INSERT Figure 4.8: App Screenshot - Request History]**

## 4.3 Backend Server Architecture

### 4.3.1 API Design

Table 4.1: REST API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | /api/submit-request | Submit guest voice request |
| GET | /api/request-history | Get room-specific request history |
| POST | /api/update-status | Update request status |
| POST | /api/update-department | Transfer request to department |
| POST | /api/cancel-request | Cancel guest request |
| POST | /api/send-message | Staff sends message to guest |
| POST | /api/rate-request | Guest rates completed service |
| GET | /api/departments | List all departments |

### 4.3.2 Database Schema

**[INSERT Figure 4.9: Database ER Diagram]**

Table 4.2: Requests Table Schema

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key, auto-increment |
| room_number | TEXT | Guest room identifier |
| request_text | TEXT | Original voice transcription |
| intent | TEXT | Classified intent category |
| department | TEXT | Routed department |
| status | TEXT | pending/in_progress/completed/cancelled |
| rating | INTEGER | Guest rating (1-5, default 0) |
| created_at | DATETIME | Request timestamp |
| completed_at | DATETIME | Completion timestamp |

### 4.3.3 Department Routing Logic

Table 4.3: Intent-to-Department Mapping

| Intent | Department |
|--------|-----------|
| towel_request, pillow_request, blanket_request, room_cleaning, toiletries_request | Housekeeping |
| food_order | Room Service |
| maintenance, temperature_control, lighting_control | Maintenance |
| checkout_billing, noise_complaint, emergency, do_not_disturb | Front Desk |
| concierge_general, concierge_taxi, wake_up_call, laundry_service | Concierge |

### 4.3.4 WebSocket Communication

*(Describe the real-time communication architecture)*

**[INSERT Figure 4.10: WebSocket Message Flow Diagram]**

## 4.4 Staff Dashboard Design

### 4.4.1 Dashboard Features

**[INSERT Figure 4.11: Staff Dashboard Screenshot]**
**[INSERT Figure 4.12: Dashboard - Request Management View]**

## 4.5 Network Architecture

**[INSERT Figure 4.13: Network Deployment Diagram]**

---

# CHAPTER 5: IMPLEMENTATION

## 5.1 Development Environment

Table 5.1: Development Tools and Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| Android App | Kotlin + Jetpack Compose | Kotlin 1.9.x |
| UI Framework | Material Design 3 | Material3 |
| STT Engine | Vosk | vosk-model-en-us-0.22-lgraph |
| NLU Model | MobileBERT (TFLite) | google/mobilebert-uncased |
| Backend | FastAPI (Python) | 0.100+ |
| Database | SQLite | 3.x |
| Real-time | WebSocket | Native |
| Dashboard | HTML/JavaScript | ES6 |
| Model Training | PyTorch + HuggingFace Transformers | 4.x |
| IDE | Android Studio | Hedgehog / Iguana |

## 5.2 Speech-to-Text Implementation

### 5.2.1 Vosk Model Integration

*(Describe how Vosk was integrated — model loading, audio format, transcription)*

```kotlin
// Code snippet: VoskService initialization
```

### 5.2.2 Voice Activity Detection

*(Describe the energy-based VAD implementation)*

```kotlin
// Code snippet: AudioRecorder VAD logic
```

## 5.3 Intent Classification Implementation

### 5.3.1 Rule-Based Keyword Matching (Tier 1)

*(Describe the keyword dictionary and matching logic)*

### 5.3.2 MobileBERT Model Inference (Tier 2)

*(Describe TFLite model loading, tokenisation, inference)*

### 5.3.3 Confidence Thresholding

*(Explain the 0.60 minimum confidence threshold and rejection logic)*

## 5.4 Voice Interaction Features

### 5.4.1 Voice Confirmation

*(Describe the yes/no voice confirmation flow with button fallback)*

### 5.4.2 Voice-Based Request Cancellation

*(Describe the cancel pattern regex, spoken number recognition)*

### 5.4.3 Text-to-Speech Response

*(Describe Android TTS integration for voice feedback)*

## 5.5 Backend Implementation

### 5.5.1 FastAPI Server
### 5.5.2 SQLite Database Operations
### 5.5.3 WebSocket Hub
### 5.5.4 Department Routing Engine

## 5.6 Staff Dashboard Implementation

### 5.6.1 Real-Time Request Management
### 5.6.2 Staff-to-Guest Messaging
### 5.6.3 Department Filtering

## 5.7 UI/UX Implementation

### 5.7.1 Material Design 3 Theme
### 5.7.2 Dark/Light Mode Support
### 5.7.3 Animated Microphone Button
### 5.7.4 Request Status Visualisation

## 5.8 Key Implementation Challenges and Solutions

Table 5.2: Implementation Challenges

| Challenge | Root Cause | Solution |
|-----------|-----------|----------|
| Requests not displaying | SnapshotStateList + remember() reference equality | Removed remember() wrapper |
| NLU false positives | Ambiguous single keywords ("water" -> food_order) | Replaced with multi-word contextual phrases |
| Thread safety crashes | API callbacks on background thread updating Compose state | Handler(Looper.getMainLooper()).post{} |
| JSON parsing crashes | getString() on null database fields | Changed to optString() with defaults |
| Spoken number recognition | Vosk transcribes "146" as "hundred and forty six" | wordsToNumber() converter function |

---

# CHAPTER 6: EVALUATION AND RESULTS

## 6.1 NLU Model Evaluation

### 6.1.1 Overall Performance

Table 6.1: MobileBERT Intent Classification Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 92.0% |
| Macro F1-Score | 0.917 |
| Weighted F1-Score | 0.919 |
| Training Loss (final) | 0.2939 |

### 6.1.2 Per-Intent Performance

Table 6.2: Classification Report by Intent

*(Copy from nlu-model/classification_report.txt — include precision, recall, F1 for each intent)*

| Intent | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| blanket_request | ... | ... | ... | ... |
| checkout_billing | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

**[INSERT Figure 6.1: Confusion Matrix Heatmap]**

### 6.1.3 Analysis of Misclassifications

*(Discuss which intents are confused with each other and why)*

## 6.2 STT Word Error Rate Evaluation

### 6.2.1 Experiment Setup

*(Describe: number of sentences, speakers, accent groups)*

Table 6.3: WER Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Test Sentences | 50 |
| Speakers | X |
| Accent Groups | Sri Lankan English, Neutral English |
| STT Systems | Vosk (on-device), Whisper Small (server) |

### 6.2.2 Overall WER Results

Table 6.4: Word Error Rate Comparison

| STT System | Overall WER | Sri Lankan English | Neutral English |
|------------|-------------|-------------------|-----------------|
| Vosk (on-device) | X% | X% | X% |
| Whisper Small (server) | X% | X% | X% |

### 6.2.3 WER by Intent Category

Table 6.5: WER by Intent Category (Vosk)

| Intent Category | WER | Common Errors |
|----------------|-----|---------------|
| food_order | X% | ... |
| room_cleaning | X% | ... |
| ... | X% | ... |

### 6.2.4 Analysis

*(Discuss: where Vosk struggles, accuracy vs offline trade-off, accent impact)*

## 6.3 System Latency Evaluation

### 6.3.1 Pipeline Stage Latency

Table 6.6: Average Latency per Pipeline Stage (N=15 requests)

| Stage | Average (ms) | Min (ms) | Max (ms) |
|-------|-------------|----------|----------|
| Audio Recording + VAD | X | X | X |
| Vosk STT Processing | X | X | X |
| NLU Classification | X | X | X |
| API Submission | X | X | X |
| TTS Response | X | X | X |
| **Total End-to-End** | **X** | **X** | **X** |

**[INSERT Figure 6.2: Latency Breakdown Bar Chart]**

## 6.4 Cost Analysis

Table 6.7: Cost Comparison — This System vs Commercial Alternatives

| Cost Factor | This System | Alexa for Hospitality | Google Nest |
|-------------|------------|----------------------|-------------|
| Device Cost (per room) | ~$80 (Android tablet) | ~$200 (Echo device) | ~$230 (Nest Hub) |
| Cloud Subscription | $0 (fully offline) | ~$X/month/room | ~$X/month/room |
| Server Hardware | ~$500 (one-time, shared) | N/A (cloud) | N/A (cloud) |
| Internet Dependency | None (LAN only) | Required (continuous) | Required (continuous) |
| Annual Cost (50 rooms) | ~$X | ~$X | ~$X |
| 3-Year TCO (50 rooms) | ~$X | ~$X | ~$X |

## 6.5 Summary of Results

*(1-2 paragraphs summarising key findings)*

---

# CHAPTER 7: DISCUSSION

## 7.1 Addressing Research Gaps

Table 7.1: Research Gap Resolution

| Research Gap | System Feature | Contribution Strength |
|-------------|---------------|----------------------|
| Gap 1: No end-to-end offline system | Complete STT+NLU+TTS+Backend+Dashboard | Strong |
| Gap 2: No on-device hospitality NLU | Fine-tuned MobileBERT TFLite (92%) | Strong |
| Gap 3: No privacy-preserving hotel VA | All processing on-device/LAN | Strong |
| Gap 4: No bidirectional guest-staff comm | WebSocket real-time architecture | Moderate |
| Gap 5: No commodity hardware deployment | Standard Android tablets | Strong |

## 7.2 Comparison with Existing Solutions

*(Compare your results with findings from literature review)*

## 7.3 Practical Implications

*(Discuss what this means for Sri Lankan hotels and developing economies)*

## 7.4 Limitations

1. **Single Language**: English only — no Sinhala/Tamil support.
2. **Limited Entity Extraction**: Basic quantity extraction only; no advanced slot filling.
3. **No Multi-Turn Dialogue**: Single-turn requests only.
4. **No Authentication Layer**: No device or API authentication implemented.
5. **Controlled Environment**: Not tested in a live hotel deployment.
6. **Limited Speaker Diversity**: WER evaluation with X speakers only.
7. **No Scalability Testing**: Concurrent multi-room load not evaluated.

## 7.5 Threats to Validity

### 7.5.1 Internal Validity
### 7.5.2 External Validity
### 7.5.3 Construct Validity

---

# CHAPTER 8: CONCLUSION AND FUTURE WORK

## 8.1 Conclusion

*(2-3 paragraphs summarising what you did, what you found, and what it means)*

This research designed, developed, and evaluated a low-cost, offline-capable voice assistant system for hotel room service operations targeting developing economies...

The evaluation demonstrated that on-device speech recognition using Vosk achieves X% WER on hotel-domain utterances, while the fine-tuned MobileBERT intent classifier achieves 92% accuracy across 18 service categories...

The system addresses five identified research gaps at the intersection of edge AI, privacy-preserving computing, and hospitality technology, demonstrating that effective voice-based service automation is achievable on commodity Android hardware without cloud dependency...

## 8.2 Future Work

1. **Multi-Language Support**: Extend STT and NLU to support Sinhala and Tamil for local guests.
2. **Multi-Turn Dialogue**: Implement conversational context management for complex requests.
3. **Advanced Entity Extraction**: Add slot filling for specific items, quantities, and times.
4. **Live Hotel Deployment**: Conduct user acceptance testing in a real hotel environment.
5. **Custom Vosk Language Model**: Train a hotel-specific language model to improve STT accuracy on domain vocabulary.
6. **Device Authentication**: Implement secure device-to-server authentication using certificates.
7. **Analytics Dashboard**: Add reporting and analytics for hotel management insights.

---

# REFERENCES

*(Use APA 7th Edition or IEEE format — check your university's requirement)*

*(Convert all references from RESEARCH_ANALYSIS.md to proper citation format)*

[1] Buhalis, D. & Moldavska, I. (2021). In-room Voice-Based AI Digital Assistants Transforming On-Site Hotel Services and Guests' Experiences. In *Information and Communication Technologies in Tourism 2021*, pp. 30-44. Springer.

[2] Buhalis, D. & Moldavska, I. (2022). Voice Assistants in Hospitality: Using Artificial Intelligence for Customer Service. *Journal of Hospitality and Tourism Technology*, 13(3), 386-403.

[3] Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. In *Proceedings of ACL 2020*, pp. 2158-2170.

[4] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *NeurIPS 2019 Workshop on Energy Efficient Machine Learning and Cognitive Computing*.

[5] Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *Proceedings of ICML 2023*.

[6] Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637-646.

[7] Bunk, T., Varber, D., Moez, H., Lechaux, L., & Nichol, A. (2020). DIET: Dual Intent and Entity Transformer. Rasa Technologies.

[8] Alpha Cephei (2020). Vosk - Offline Speech Recognition API. https://alphacephei.com/vosk/

*(Continue numbering all references from RESEARCH_ANALYSIS.md...)*

---

# APPENDICES

## Appendix A: Sample Training Data

*(Include 20-30 example rows from hotel_intents_production_ready.csv)*

## Appendix B: Complete Classification Report

*(Full classification_report.txt output)*

## Appendix C: API Request/Response Examples

*(Show sample JSON request/response for each endpoint)*

## Appendix D: WER Experiment Raw Data

*(Complete table of all test sentences, ground truth, Vosk output, Whisper output, individual WER)*

## Appendix E: Source Code Repository

The complete source code for this project is available at: [GitHub Repository URL]

---

# Word Formatting Tips

## Page Setup
- A4 paper, 1.5 line spacing
- Margins: Top 1", Bottom 1", Left 1.25", Right 1"
- Font: Times New Roman 12pt (body), 14pt bold (headings)
- Page numbers: bottom centre

## Auto-Generate in Word
1. Heading styles: Use Heading 1, Heading 2, Heading 3 consistently
2. Table of Contents: References tab > Table of Contents > Automatic
3. List of Figures: References tab > Insert Table of Figures > select "Figure"
4. List of Tables: References tab > Insert Table of Figures > select "Table"
5. Citations: Use Mendeley/Zotero plugin or Word's built-in citation manager

## Figure Captions
- Insert > Caption > "Figure 4.1: System Architecture Diagram"
- Place caption BELOW the figure

## Table Captions
- Insert > Caption > "Table 3.1: Dataset Distribution"
- Place caption ABOVE the table
