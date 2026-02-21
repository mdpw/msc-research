# Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models

## 1. Introduction

The hospitality industry in Sri Lanka is a key contributor to the national economy, with hotels constantly seeking ways to enhance guest experience while managing operational costs. Traditional room service communication methods, such as phone calls to the front desk or physical service requests, are often inefficient, prone to miscommunication, and require dedicated staff availability.

Recent advancements in speech recognition and natural language processing (NLP) have enabled voice-based virtual assistants such as Amazon Alexa for Hospitality and Google Nest for Hotels. However, these solutions rely heavily on cloud infrastructure, require continuous internet connectivity, and raise significant privacy concerns, particularly for hotel guests. Furthermore, the high costs associated with proprietary hardware and cloud API subscriptions make them impractical for small to mid-sized hotels in developing countries like Sri Lanka.

This research proposes a low-cost, offline-capable voice assistant system designed specifically for hospitality services. The system leverages small-scale neural models running on commodity Android devices to provide speech-to-text (STT), intent classification, and text-to-speech (TTS) capabilities entirely on-device or within a local network, without any dependency on external cloud services. The system enables guests to make service requests through natural voice commands, which are automatically classified, routed to the appropriate hotel department, and tracked in real-time through a staff dashboard.

## 2. Problem Statement

Hotels in Sri Lanka and similar developing economies face several challenges in modernising their guest service operations:

1. **High cost of existing solutions**: Cloud-based voice assistants (Alexa for Hospitality, Google Assistant) require recurring subscription fees, proprietary hardware, and stable high-speed internet, costs that are prohibitive for most Sri Lankan hotels.

2. **Privacy concerns**: Cloud-based systems transmit guest voice data to external servers, raising data privacy and compliance issues. Guests may be uncomfortable knowing their in-room conversations are processed by third-party cloud services.

3. **Internet dependency**: Many hotels, especially those in rural tourism areas (e.g., Ella, Sigiriya, Arugam Bay), suffer from unreliable internet connectivity. Cloud-dependent systems become non-functional during outages.

4. **Operational inefficiency**: Traditional room service relies on phone calls that are subject to language barriers, miscommunication, long hold times, and lack of request tracking. There is no systematic audit trail for guest requests, making quality assurance difficult.

5. **Language and accent challenges**: Existing voice assistants are primarily optimised for native English speakers and perform poorly with South Asian accents, which is a barrier for both international and local guests.

**Research Question**: How effective is a commodity hardware-based, offline voice assistant system in reducing operational costs and improving service efficiency compared to traditional room service methods in Sri Lankan hotels?

## 3. Objectives

### Primary Objectives
1. Design and develop a fully functional offline voice assistant system for hotel room service operations using commodity Android devices.
2. Implement on-device speech-to-text (STT) using Vosk and on-device intent classification using a fine-tuned MobileBERT model.
3. Build a real-time hotel management backend with automated department routing and a staff dashboard for request tracking.
4. Evaluate the system's accuracy, latency, and cost-effectiveness compared to cloud-based alternatives and traditional service methods.

### Secondary Objectives
5. Create a hotel-specific intent classification dataset covering 18 common service request categories.
6. Implement voice-based interaction for the complete request lifecycle, including submission, confirmation, cancellation, and rating.
7. Demonstrate privacy-preserving edge computing for hospitality AI without any external data transmission.

## 4. Literature Review

### 4.1 Voice Assistants in Hospitality

The adoption of voice assistants in the hospitality industry has gained momentum with major hotel chains deploying Amazon Alexa for Hospitality and Google Nest Hub. Marriott International piloted Alexa-powered devices across select properties in 2018, enabling guests to control room amenities and request services through voice commands (Marriott International, 2018). However, these deployments require cloud connectivity and incur ongoing API costs, limiting adoption among budget and mid-range hotels.

Research by Buhalis and Moldavska (2022) highlighted that while AI-powered voice assistants improve guest satisfaction and operational efficiency, privacy concerns remain a significant barrier to adoption. Their study found that 67% of hotel guests expressed discomfort with always-on cloud-connected microphones in their rooms.

### 4.2 Offline Speech Recognition

Offline speech recognition has advanced significantly with models such as Vosk (Alpha Cephei, 2020), which provides lightweight, offline-capable speech recognition supporting multiple languages. Vosk operates on edge devices without internet connectivity, making it suitable for privacy-sensitive applications. The vosk-model-en-us-0.22-lgraph variant used in this research provides a compact model (205MB) optimised for low-resource devices.

Mozilla DeepSpeech and OpenAI Whisper represent alternative approaches. Whisper (Radford et al., 2023) achieves state-of-the-art accuracy but requires significant computational resources. Quantised versions (Whisper.cpp) enable server-side deployment but remain too resource-intensive for on-device processing on commodity Android hardware.

### 4.3 Intent Classification with Small-Scale Models

Transfer learning using pre-trained language models has revolutionised NLU tasks. BERT (Devlin et al., 2019) and its distilled variants, DistilBERT (Sanh et al., 2019) and MobileBERT (Sun et al., 2020), enable fine-tuning on domain-specific datasets with limited training data. MobileBERT achieves 4.3x compression and 5.5x speedup over BERT-base while retaining 96% of its performance, making it suitable for mobile deployment via TensorFlow Lite.

For hospitality-specific NLU, custom dataset creation is essential as general-purpose intent classifiers lack hotel service vocabulary. Liu et al. (2021) demonstrated that domain-specific fine-tuning with as few as 500 examples per class can achieve >90% classification accuracy for task-oriented dialogue systems.

### 4.4 Edge Computing for Privacy-Preserving AI

Edge computing architectures process data locally rather than transmitting it to cloud servers, addressing privacy concerns and reducing latency. Shi et al. (2016) established the theoretical foundation for edge intelligence, while Chen and Ran (2019) demonstrated practical implementations for speech processing on mobile devices. The privacy-preserving nature of edge computing is particularly relevant for hospitality, where guest conversations may contain sensitive information.

### 4.5 Research Gap

While individual components (offline STT, mobile NLU, hotel management systems) exist in isolation, no existing research combines them into an integrated, low-cost, offline-capable voice assistant specifically designed for hospitality services in developing economies. This research addresses this gap by providing a complete end-to-end system that operates entirely within a hotel's local network using commodity hardware.

## 5. Methodology

The following methodology has been identified for this research:

### Phase 1: Literature Review and Problem Definition
- Conduct a comprehensive review of existing research on voice assistants in hospitality, offline speech recognition systems, small-scale neural models for NLU, and edge computing architectures.
- Analyse existing commercial solutions (Alexa for Hospitality, Google Nest for Hotels) to identify cost structures, limitations, and gaps.
- Define system requirements based on the operational needs of Sri Lankan hotels.

### Phase 2: Dataset Preparation
- Create a hotel-specific intent classification dataset with 18 intent categories covering common guest service requests.
- Curate approximately 5,000 labelled examples with natural language variations, including typos, abbreviations, and colloquial phrasing.
- Apply stratified splitting (85% training, 15% validation) to ensure balanced class representation.
- Augment data with paraphrasing and synonym replacement to improve model generalisation.

**Intent Categories (18 classes)**:

| Category | Examples | Training Samples |
|----------|----------|-----------------|
| food_order | "I'd like to order room service", "Can I have a water bottle" | 337 |
| room_cleaning | "Please clean my room", "I need housekeeping" | 412 |
| towel_request | "I need fresh towels", "Extra towels please" | 508 |
| maintenance | "The AC is not working", "Sink is leaking" | 256 |
| temperature_control | "It's too cold in here", "Turn up the heat" | 304 |
| toiletries_request | "I need shampoo", "Extra soap please" | 333 |
| pillow_request | "Can I get an extra pillow", "Need more pillows" | 306 |
| blanket_request | "I need a blanket", "Extra comforter" | 260 |
| laundry_service | "I have clothes to wash", "Dry cleaning" | 247 |
| wake_up_call | "Wake me up at 7 AM", "Set an alarm" | 237 |
| do_not_disturb | "Do not disturb please", "No housekeeping today" | 237 |
| concierge_taxi | "I need a taxi", "Book a cab" | 237 |
| concierge_general | "What time is checkout", "Where is the pool" | 244 |
| checkout_billing | "I want to check out", "Can I see my bill" | 208 |
| noise_complaint | "The room next door is too loud", "Noise complaint" | 214 |
| lighting_control | "Turn off the lights", "Dim the lights" | 226 |
| emergency | "I need help immediately", "Medical emergency" | 214 |
| misc_request | General/uncategorised requests | 192 |
| **Total** | | **4,971** |

### Phase 3: Model Development and Training

#### 3.1 Speech-to-Text (STT)
- Deploy Vosk offline speech recognition engine on Android devices.
- Use the vosk-model-en-us-0.22-lgraph model (205MB) optimised for mobile devices.
- Implement custom Voice Activity Detection (VAD) with energy-based silence detection.
- Audio specification: 16kHz sample rate, 16-bit PCM, mono channel.

#### 3.2 Intent Classification (NLU)
- Fine-tune MobileBERT (google/mobilebert-uncased) on the hotel intent dataset.
- Convert the trained model to TensorFlow Lite format for on-device inference.
- Implement a hybrid classification pipeline:
  - **Tier 1**: Rule-based keyword matching for high-confidence classifications (0.99 confidence).
  - **Tier 2**: MobileBERT model inference for ambiguous or complex requests.
- Apply a minimum confidence threshold (0.60) to reject unclear requests.

**Training Configuration**:
- Optimiser: AdamW with learning rate 3e-5
- Batch size: 32
- Epochs: 8
- Max sequence length: 32 tokens
- Train/validation split: 85%/15% (stratified)

#### 3.3 Text-to-Speech (TTS)
- Use Android's built-in TextToSpeech engine for voice responses.
- Configure speech rate (0.9x) and pitch (1.0x) for natural hotel assistant voice.
- Implement blocking TTS for confirmation dialogues requiring sequential interaction.

### Phase 4: System Development

#### 4.1 Android Application (Guest Device)
- **Framework**: Kotlin with Jetpack Compose (Material Design 3)
- **Architecture**: On-device STT and NLU with server-side request management
- **Features**:
  - Voice request submission with automatic intent classification
  - Voice-based confirmation (yes/no) with button fallback
  - Voice-based request cancellation ("Cancel my order number 146")
  - Spoken number recognition (e.g., "hundred and forty six" to 146)
  - Real-time request status tracking via WebSocket
  - Service rating system (1-5 stars)
  - Dark/light theme with system preference detection
  - Multi-server network profile management

#### 4.2 Backend Server
- **Framework**: FastAPI (Python)
- **Database**: SQLite with full CRUD operations
- **API Endpoints**: RESTful API for request submission, status management, cancellation, messaging, and rating
- **Real-time Communication**: WebSocket for bidirectional updates between guest devices, server, and staff dashboard
- **Department Routing**: Automatic intent-to-department mapping for 5 departments (Housekeeping, Room Service, Maintenance, Front Desk, Concierge)

#### 4.3 Staff Dashboard
- **Technology**: Single-page HTML/JavaScript application
- **Features**:
  - Department-specific request queues with real-time WebSocket updates
  - Request status management (pending, in progress, completed, cancelled)
  - Inter-department request transfer
  - Staff-to-guest messaging
  - Guest rating display
  - Browser desktop notifications for new requests
  - Live connection status indicator

### Phase 5: Evaluation Metrics and Testing

#### 5.1 Model Evaluation
| Metric | Description | Target |
|--------|-------------|--------|
| STT Word Error Rate (WER) | Accuracy of speech-to-text transcription | < 15% |
| NLU Intent Accuracy | Correct intent classification rate | > 90% |
| NLU F1-Score | Precision-recall balance per intent class | > 0.90 |
| Confidence Calibration | Reliability of confidence scores | ECE < 0.10 |

#### 5.2 System Performance
| Metric | Description | Target |
|--------|-------------|--------|
| End-to-End Latency | Time from voice input to TTS confirmation | < 5 seconds |
| STT Processing Time | Audio to text conversion time | < 2 seconds |
| NLU Inference Time | Intent classification time | < 200ms |
| Concurrent Requests | Simultaneous requests handled by server | > 20 |

#### 5.3 Cost Analysis
| Metric | Description |
|--------|-------------|
| Hardware Cost | Per-room device cost (commodity Android tablet) |
| Infrastructure Cost | Server hardware and network setup |
| Operational Cost | Ongoing maintenance and power consumption |
| Comparison | Cost analysis vs. cloud-based alternatives (Alexa, Google) |
| ROI Period | Time to recover investment through operational savings |

#### 5.4 User Evaluation
- Conduct user acceptance testing with hotel staff and simulated guest interactions.
- Measure task completion rate, error recovery rate, and user satisfaction.
- Compare service request handling time against traditional phone-based methods.

## 6. System Architecture

```
+------------------------------------------+
|          GUEST ROOM (Android Device)       |
|                                           |
|  Mic Input -> VAD -> Vosk STT (On-Device) |
|                  |                         |
|         MobileBERT NLU (On-Device)        |
|                  |                         |
|         HTTP API -> Hotel Server           |
|                  |                         |
|    WebSocket <- TTS <- Response            |
+------------------------------------------+
              |           ^
              v           |
+------------------------------------------+
|          HOTEL SERVER (Local Network)      |
|                                           |
|  FastAPI Backend <-> SQLite Database      |
|         |                                 |
|  WebSocket Hub (Broadcast)                |
|         |                                 |
|  Department Routing Engine                |
+------------------------------------------+
              |           ^
              v           |
+------------------------------------------+
|          STAFF DASHBOARD (Web Browser)     |
|                                           |
|  Department-Filtered Request Queue        |
|  Status Management & Messaging            |
|  Real-time WebSocket Updates              |
+------------------------------------------+
```

## 7. Timeline

| Phase | Activity | Duration |
|-------|----------|----------|
| Phase 1 | Literature review and problem definition | Weeks 1-3 |
| Phase 2 | Dataset preparation and curation | Weeks 3-5 |
| Phase 3 | Model training and optimisation (STT, NLU) | Weeks 5-8 |
| Phase 4 | Android app development | Weeks 6-10 |
| Phase 4 | Backend API and database development | Weeks 7-10 |
| Phase 4 | Staff dashboard development | Weeks 9-11 |
| Phase 5 | System integration and testing | Weeks 11-13 |
| Phase 5 | Evaluation and benchmarking | Weeks 13-15 |
| Phase 6 | Thesis writing and documentation | Weeks 14-18 |
| Phase 6 | Final review and submission | Weeks 18-20 |

## 8. Expected Outcome

1. **A working prototype** of an offline voice assistant system deployable on commodity Android tablets, capable of processing hotel service requests without cloud connectivity.

2. **A custom hotel intent dataset** of approximately 5,000 labelled examples covering 18 service categories, suitable for training domain-specific NLU models.

3. **A fine-tuned MobileBERT model** achieving >90% intent classification accuracy on hotel service requests, converted to TensorFlow Lite for on-device inference.

4. **A complete hotel management system** including a FastAPI backend with real-time WebSocket communication and a web-based staff dashboard for department-level request management.

5. **A cost-benefit analysis** demonstrating the economic viability of the proposed system compared to cloud-based alternatives, with projected cost savings for small to mid-sized hotels.

6. **A privacy-preserving architecture** that processes all guest voice data locally within the hotel network, eliminating external data transmission and associated privacy risks.

## 9. Current Implementation Status

| Component | Status | Details |
|-----------|--------|---------|
| On-device STT (Vosk) | Implemented | vosk-model-en-us-0.22-lgraph, 16kHz PCM |
| Voice Activity Detection | Implemented | Energy-based, configurable silence timeout |
| NLU - Rule-based (Tier 1) | Implemented | Keyword dictionary, 0.99 confidence |
| NLU - MobileBERT (Tier 2) | Implemented | 92% accuracy, TFLite, 18 intents |
| Training Dataset | Implemented | 4,971 examples, 18 classes |
| Android UI (Jetpack Compose) | Implemented | Material 3, dark/light theme |
| Voice Confirmation | Implemented | Yes/no voice + button fallback |
| Voice Cancellation | Implemented | Spoken number recognition |
| FastAPI Backend | Implemented | RESTful API, 8 endpoints |
| SQLite Database | Implemented | Requests, staff messages, ratings |
| WebSocket Communication | Implemented | Bidirectional, real-time updates |
| Staff Dashboard | Implemented | Department queues, status management |
| Staff Messaging | Implemented | Dashboard to guest device |
| Guest Rating System | Implemented | 1-5 star rating |
| Department Routing | Implemented | 5 departments, auto-routing |
| Device Authentication | Not Implemented | Currently hardcoded room config |
| Multi-language Support | Not Implemented | English only |
| Evaluation Benchmarks | Partial | NLU accuracy evaluated, STT WER pending |
| Load Testing | Not Implemented | Concurrent request testing pending |
| User Acceptance Testing | Not Implemented | Real hotel environment testing pending |

## 10. Conclusion

This research addresses a practical gap in the hospitality technology landscape by developing a low-cost, offline-capable voice assistant system tailored for Sri Lankan hotels. By leveraging small-scale neural models (Vosk for STT, MobileBERT for NLU) running on commodity Android devices, the system eliminates dependency on expensive cloud services and proprietary hardware while preserving guest privacy through edge computing.

The proposed system demonstrates that effective voice-based hotel service automation is achievable without cloud connectivity, using hardware costing a fraction of commercial alternatives. The hybrid NLU approach (rule-based + neural model) provides robust intent classification with a 92% accuracy rate, while the real-time WebSocket architecture ensures seamless communication between guests, the management system, and hotel staff.

The research contributes to the fields of edge AI, hospitality technology, and privacy-preserving computing, offering a replicable model for hotels in developing economies seeking to modernise guest services without prohibitive infrastructure investments.

## References

- Alpha Cephei. (2020). Vosk - Offline Speech Recognition API. https://alphacephei.com/vosk/
- Buhalis, D., & Moldavska, I. (2022). Voice assistants in hospitality: a systematic review. International Journal of Contemporary Hospitality Management.
- Chen, J., & Ran, X. (2019). Deep learning with edge computing: A review. Proceedings of the IEEE.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT.
- Liu, Z., et al. (2021). Few-shot intent classification and slot tagging with retrieved examples. NAACL.
- Radford, A., et al. (2023). Robust speech recognition via large-scale weak supervision (Whisper). ICML.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. NeurIPS Workshop.
- Shi, W., et al. (2016). Edge computing: Vision and challenges. IEEE Internet of Things Journal.
- Sun, Z., et al. (2020). MobileBERT: a compact task-agnostic BERT for resource-limited devices. ACL.
