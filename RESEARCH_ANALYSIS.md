# Academic Research Analysis: Low-Cost Offline Voice Assistant for Hospitality Services

---

## Part 1: Related Work Identification

### Category A: Voice Assistants in Hospitality

| # | Authors | Year | Title | Key Contribution | Limitations |
|---|---------|------|-------|-----------------|-------------|
| A1 | Buhalis, D. & Moldavska, I. | 2021 | In-room Voice-Based AI Digital Assistants Transforming On-Site Hotel Services and Guests' Experiences | Qualitative study (28 interviews) examining voice assistants from both technology providers' and guests' perspectives. Found benefits outweigh drawbacks but identified privacy and usability as key concerns. | No technical implementation; purely qualitative. Focused on cloud-based commercial solutions (Alexa, Google). No consideration of offline or low-cost alternatives. |
| A2 | Buhalis, D. & Moldavska, I. | 2022 | Voice Assistants in Hospitality: Using Artificial Intelligence for Customer Service | Comprehensive analysis of VA-enabled hotel-guest interactions. Documented use cases including room controls, room service ordering, and wake-up alarms. | Limited to cloud-dependent platforms. No cost analysis for developing-economy hotels. No technical architecture proposed. |
| A3 | Hwang, J. & Erdem, M. | 2025 | "U" in User Experience (UX) Stands for the Frontline Employee: A Case Study of Voice Assistant Technology Use in Hotels | Examined voice assistant impact on hotel employee workflows and UX. Highlighted how VA technology transforms frontline staff work environments. | Employee-focused only; no guest-side technical implementation. Based on existing commercial solutions. |
| A4 | Amazon | 2018-present | Alexa for Hospitality / Alexa Smart Properties | Industry-leading commercial deployment. Provides device management, custom skills, and property-level administration at scale. Privacy features include no-save voice recordings and microphone disconnect. | Requires constant cloud connectivity and AWS subscription. Proprietary hardware (Echo devices). 30% accuracy degradation with non-native English accents. Cost prohibitive for budget hotels in developing countries. |
| A5 | [NEEDS VERIFICATION] | 2021 | To Talk or to Touch: Unraveling Consumer Responses to Two Types of Hotel In-Room Technology | Comparative study of AI voice assistants vs. touch-screen tablets in hotel rooms. Analysed guest preferences and engagement patterns. | Focused on consumer behaviour, not technical implementation. Limited to cloud-based solutions. |

### Category B: Offline/Edge-Based Speech Recognition Systems

| # | Authors | Year | Title | Key Contribution | Limitations |
|---|---------|------|-------|-----------------|-------------|
| B1 | Korkmaz, A. et al. | 2025 | Real-Time Speech-to-Text on Edge: A Prototype System for Ultra-Low Latency Communication with AI-Powered NLP | Developed edge STT system using Vosk for ultra-low latency. Demonstrated deployment on constrained hardware (Raspberry Pi). | Focused on general communication, not hospitality domain. No intent classification pipeline integrated. |
| B2 | [Authors] | 2025 | Improving Speech Recognition Accuracy Using Custom Language Models with the Vosk Toolkit | Demonstrated custom language model integration with Vosk for domain-specific vocabulary. Achieved 40% WER reduction over off-the-shelf models. Fully offline operation. | Did not address hospitality-specific vocabulary. No end-to-end system integration with backend services. |
| B3 | Chantrapornchai, C. & Suchato, A. | 2022 | IoT Device Control with Offline Automatic Speech Recognition on Edge Device | Explored offline ASR on Raspberry Pi for IoT device control. Benchmarked edge processing capabilities for speech commands. | Limited to simple command recognition. No NLU pipeline. Not applied to service industry context. |
| B4 | Radford, A. et al. | 2023 | Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) | State-of-the-art accuracy across 99 languages. Trained on 680,000 hours of multilingual data. Robust against noise and accents. | Requires GPU (8GB+ VRAM for large models). Not suitable for on-device mobile deployment. Cloud or server dependency. |
| B5 | [Authors] | 2024 | Evaluation of Voice Recognition Platforms and Methods for Edge AI Devices | Comprehensive benchmark of ASR platforms on edge devices. Compared accuracy, latency, and resource consumption across Vosk, Whisper, and others. | General evaluation without domain-specific application. No integration with downstream NLU or service management. |

### Category C: Small-Scale NLU Models for Resource-Constrained Devices

| # | Authors | Year | Title | Key Contribution | Limitations |
|---|---------|------|-------|-----------------|-------------|
| C1 | Sun, Z. et al. | 2020 | MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices | 4.3x smaller and 5.5x faster than BERT-BASE. GLUE score 77.7. 62ms latency on Pixel 4. Progressive knowledge distillation from inverted-bottleneck BERT-LARGE teacher. | General NLP benchmarks only; not evaluated on domain-specific intent classification. No TFLite deployment evaluation for mobile inference. |
| C2 | Sanh, V. et al. | 2019 | DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter | Retains 97% of BERT's accuracy. 40% smaller, 60% faster. General-purpose knowledge distillation approach. | Larger than MobileBERT for mobile deployment. Not specifically optimised for on-device inference on Android. |
| C3 | Bujel, K. et al. | 2021 | Effectiveness of Pre-training for Few-shot Intent Classification | IntentBERT: fine-tuning BERT with ~1,000 labelled data surpasses existing pre-trained models for few-shot intent classification on novel domains. | Focused on few-shot scenarios; does not address full fine-tuning with domain-specific datasets. No mobile deployment consideration. |
| C4 | Rasa (Bunk, T. et al.) | 2020 | DIET: Dual Intent and Entity Transformer | Multi-task architecture handling both intent classification and entity recognition. Outperforms fine-tuned BERT while being 6x faster to train. Supports pluggable pre-trained embeddings. | Server-side framework; not designed for on-device mobile inference. Requires Rasa server infrastructure. |
| C5 | [Authors] | 2022 | Fine-Tuning BERT Models for Intent Recognition Using a Frequency Cut-Off Strategy for Domain-Specific Vocabulary Extension | Domain-specific vocabulary extension strategy for BERT intent recognition. Demonstrated improved accuracy on specialised vocabularies. | Not evaluated on hospitality domain. No mobile/edge deployment. |

### Category D: Privacy-Preserving AI in Service Industries

| # | Authors | Year | Title | Key Contribution | Limitations |
|---|---------|------|-------|-----------------|-------------|
| D1 | Shi, W. et al. | 2016 | Edge Computing: Vision and Challenges | Foundational framework for edge intelligence. Established theoretical basis for processing data at network edge to preserve privacy and reduce latency. | Theoretical framework; no practical implementation for speech/NLU. Pre-dates modern transformer models. |
| D2 | [Authors] | 2025 | Privacy-Preserving On-Device Speech Recognition Using Vosk with Domain-Specific Language Models | Developed custom Vosk models benchmarked against Google and Azure STT. Custom models outperformed cloud solutions with 40% WER reduction. Fully offline, privacy-preserving. | No downstream NLU or intent classification. Not applied to hospitality or service industry. |
| D3 | [Authors] | 2025 | Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models | Comprehensive survey of on-device AI model deployment techniques including quantisation, pruning, and knowledge distillation. | Survey paper; no novel implementation. Does not address hospitality-specific use cases. |
| D4 | Purdue University Research | 2018 | Amazon Alexa Devices in Hotels Raise Privacy Concerns | Identified specific privacy vulnerabilities in hotel Alexa deployments. Documented potential for malicious skills to eavesdrop on guests. | Focused on vulnerability identification, not solution development. No alternative privacy-preserving architecture proposed. |

### Category E: Task-Oriented Dialogue Systems for Domain-Specific Applications

| # | Authors | Year | Title | Key Contribution | Limitations |
|---|---------|------|-------|-----------------|-------------|
| E1 | Larson, S. et al. | 2022 | A Survey of Intent Classification and Slot-Filling Datasets for Task-Oriented Dialog | Comprehensive survey of datasets and methods for intent classification. Identified gaps in domain-specific dataset availability. | Survey only; no novel system or model proposed. No hospitality-specific dataset created. |
| E2 | [Authors] | 2022 | Joint Intent Detection Model for Task-oriented Human-Computer Dialogue System using Asynchronous Training | Joint model for intent detection and slot filling. Asynchronous training strategy for improved performance. | Server-side implementation. No edge/mobile deployment. Not hospitality domain. |
| E3 | [Authors] | 2023 | Unified Approach for Scalable Task-Oriented Dialogue System | Scalable framework for task-oriented dialogue. Multi-domain intent handling with transfer learning. | Cloud-dependent architecture. No offline capability. High computational requirements. |
| E4 | [Authors] | 2025 | AUTODIAL: Multi-task Dialogue Model | Parallel decoders for intent prediction, dialogue state tracking. 3-6x inference speedup with 11x fewer parameters. | Still relatively large for mobile deployment. No integration with STT/TTS for voice-based systems. |

---

## Part 2: Research Gap Analysis

### 2.1 Comparison Matrix

| Solution/Paper | Offline Capable | On-Device (Edge AI) | Hospitality Domain | Low-Cost Hardware | Privacy Preserving | Real-Time Staff Integration | End-to-End System |
|---------------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A1: Buhalis & Moldavska (2021) | - | - | Y | - | - | - | - |
| A2: Buhalis & Moldavska (2022) | - | - | Y | - | - | - | - |
| A4: Alexa for Hospitality | - | - | Y | - | Partial | Y | Y |
| B1: Edge STT with Vosk (2025) | Y | Y | - | Y | Y | - | - |
| B2: Custom Vosk LM (2025) | Y | Y | - | Y | Y | - | - |
| B3: IoT Edge ASR (2022) | Y | Y | - | Y | Y | - | - |
| B4: Whisper (2023) | Partial | - | - | - | Partial | - | - |
| C1: MobileBERT (2020) | Y | Y | - | Y | Y | - | - |
| C2: DistilBERT (2019) | Y | Partial | - | Partial | Y | - | - |
| C4: Rasa DIET (2020) | Partial | - | - | - | Partial | - | Partial |
| D1: Edge Computing (2016) | Y | Y | - | - | Y | - | - |
| D2: Privacy Vosk (2025) | Y | Y | - | Y | Y | - | - |
| **This Research** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** | **Y** |

**Legend**: Y = Fully addressed, Partial = Partially addressed, - = Not addressed

### 2.2 Identified Research Gaps

#### Gap 1: No Integrated End-to-End Offline Voice Assistant for Hospitality

**Description**: Existing research addresses individual components in isolation, STT on edge devices (B1, B2, B3), NLU model compression (C1, C2), or hospitality VA adoption studies (A1, A2), but no published work combines all components (STT + NLU + TTS + backend + staff dashboard) into a single integrated, offline-capable system specifically designed for hotel operations.

**Why it matters for developing economies**: Hotels in Sri Lanka and similar countries cannot afford to integrate multiple separate commercial solutions. A unified system that works out-of-the-box on commodity hardware removes the need for specialised IT staff and reduces total cost of ownership significantly.

#### Gap 2: Absence of On-Device NLU for Hospitality-Specific Intent Classification

**Description**: While MobileBERT (C1) and DistilBERT (C2) have been evaluated on general NLP benchmarks (GLUE, SQuAD), and Rasa DIET (C4) provides server-side intent classification, no published research evaluates fine-tuned compact transformer models on hospitality-specific intent datasets deployed as TFLite models for on-device inference on commodity Android devices.

**Why it matters for developing economies**: Cloud-based NLU services (Dialogflow, LUIS, Amazon Lex) incur per-request API costs that scale with usage. Hotels with hundreds of daily requests across dozens of rooms would face unsustainable recurring costs. On-device NLU eliminates this entirely.

#### Gap 3: Privacy-Preserving Voice Processing in Hospitality Without Cloud Dependency

**Description**: Research on privacy-preserving AI (D1, D2, D3) establishes theoretical frameworks and techniques, and studies document privacy concerns with hotel Alexa deployments (D4). However, no work demonstrates a practical privacy-preserving voice assistant system for hotels where all voice data is processed locally and never leaves the hotel's network.

**Why it matters for developing economies**: Data protection regulations are emerging across South and Southeast Asia. Hotels catering to international guests (particularly European tourists subject to GDPR expectations) need privacy-compliant solutions. Local processing eliminates cross-border data transfer concerns entirely.

#### Gap 4: Real-Time Bidirectional Communication Between Guest Voice Devices and Staff Systems

**Description**: Commercial solutions like Alexa for Hospitality (A4) provide one-directional service (guest to system). Research on hotel voice assistants (A1, A2) focuses on guest experience but does not address real-time staff notification, bidirectional messaging, department routing, and request lifecycle management integrated with voice devices.

**Why it matters for developing economies**: Hotels in developing economies often have complex department structures where a single request may need routing or escalation. Real-time bidirectional communication between guest devices and staff dashboards eliminates the need for expensive PBX/intercom systems and reduces response times.

#### Gap 5: Cost-Effective Voice AI Deployment Using Commodity Android Devices

**Description**: Existing implementations use either proprietary hardware (Amazon Echo for Alexa, Google Nest) or specialised IoT devices (Raspberry Pi with custom setups). No research demonstrates a production-viable voice assistant system running on standard, commercially available Android tablets that hotels can procure locally at minimal cost.

**Why it matters for developing economies**: Android tablets are widely available in Sri Lanka at price points of $50-150 USD, compared to $200+ for smart speakers with limited local availability. Using commodity hardware that local technicians can service and replace reduces both initial deployment and ongoing maintenance costs.

---

## Part 3: Solution Validation

### 3.1 Gap-to-Feature Mapping

| Research Gap | System Feature | Strength | Justification |
|-------------|---------------|----------|---------------|
| **Gap 1**: No integrated end-to-end offline system for hospitality | Complete pipeline: Vosk STT + MobileBERT NLU + Android TTS + FastAPI backend + WebSocket + Staff Dashboard | **STRONG** | System demonstrates full request lifecycle (voice input -> classification -> routing -> tracking -> voice response) operating entirely within hotel LAN. All 7 pipeline stages functional. |
| **Gap 2**: No on-device hospitality NLU | Fine-tuned MobileBERT (TFLite) on custom 4,971-example hotel dataset with 18 intents, 92% accuracy. Hybrid rule-based + neural pipeline. | **STRONG** | Novel contribution: first documented fine-tuning of MobileBERT for hospitality intent classification with TFLite on-device inference. Hybrid approach (keyword rules at 0.99 confidence + model fallback) is a practical innovation. |
| **Gap 3**: Privacy-preserving voice processing in hospitality | All STT (Vosk) and NLU (MobileBERT) processing on-device. No audio or text data leaves the hotel network. | **STRONG** | Zero external API calls. Voice data processed entirely on the Android device. Only structured request data (intent, text) transmitted to local server. Architectural guarantee of privacy. |
| **Gap 4**: Real-time bidirectional guest-staff communication | WebSocket architecture: guest devices <-> server <-> staff dashboard. Staff messaging, status updates, department transfers all delivered in real-time with TTS. | **MODERATE** | Bidirectional communication implemented and functional. However, system currently supports a single notification channel (WebSocket/dashboard). Production deployment would benefit from additional channels (push notifications, SMS fallback) and role-based access control. |
| **Gap 5**: Commodity hardware deployment | Runs on standard Android tablets. No proprietary hardware. Server runs on any Linux/Windows machine. | **STRONG** | Demonstrated on commodity Android devices. Vosk model (205MB) and MobileBERT TFLite (26MB) both fit comfortably on budget Android tablets. Total per-room hardware cost estimated at $50-150, compared to $200+ for commercial VA hardware plus recurring cloud subscriptions. |

### 3.2 Remaining Limitations

The following limitations should be acknowledged in the thesis:

1. **Single Language Support**: The current system supports English only. Sri Lankan hotels serve guests speaking Sinhala, Tamil, and numerous other languages. Multi-language STT and NLU would significantly increase practical applicability.

2. **No Formal STT Accuracy Evaluation**: While the NLU model has been evaluated (92% accuracy), the Vosk STT component lacks formal Word Error Rate (WER) benchmarking on hotel-domain speech, particularly with South Asian accents.

3. **Limited Entity Extraction**: The NLU pipeline classifies intents but performs only basic entity extraction (quantities via regex). Advanced slot filling (specific food items, exact times, room preferences) is not implemented.

4. **No Authentication/Security Layer**: The system lacks device authentication, API authentication, and role-based access control. A production deployment requires secure device-to-server communication.

5. **No Multi-Turn Dialogue**: The system handles single-turn requests only. Complex requests requiring clarification ("I'd like to order food" -> "What would you like?") are not supported.

6. **Scalability Not Evaluated**: The system has not been load-tested with concurrent requests from multiple rooms or evaluated for degradation under high-usage scenarios.

7. **No Comparative User Study**: No formal user acceptance testing or comparative study against traditional hotel service methods (phone calls) has been conducted.

8. **Hardcoded Room Configuration**: Room-device association is hardcoded rather than dynamically configurable, limiting deployment flexibility.

### 3.3 Recommended Additional Evaluation Experiments

#### Experiment 1: STT Word Error Rate (WER) Benchmark on Hotel-Domain Speech

**Objective**: Evaluate Vosk STT accuracy on hotel-specific vocabulary with varied accent profiles.

**Method**:
- Record 200-300 hotel service request utterances from 15-20 speakers with varied accents (native Sinhala/Tamil speakers, native English, South Asian English).
- Transcribe using Vosk (on-device) and compare against ground truth.
- Benchmark against Whisper (server-side) as an accuracy ceiling.
- Report WER per accent group and per intent category.

**Expected Contribution**: First documented WER evaluation of Vosk on hospitality-domain speech with South Asian accent representation.

#### Experiment 2: End-to-End Latency and Resource Consumption Measurement

**Objective**: Quantify the complete pipeline latency and device resource usage.

**Method**:
- Instrument the system to measure time at each stage: audio recording, VAD, STT, NLU classification, API submission, WebSocket delivery, TTS playback.
- Test on 3-4 different Android devices at varying price points ($50, $100, $200 tablets).
- Measure CPU usage, memory consumption, and battery impact during continuous operation.
- Compare against cloud-based pipeline (Google Speech API + Dialogflow) for latency and cost-per-request.

**Expected Contribution**: Empirical evidence that on-device processing achieves acceptable latency on commodity hardware, with cost-per-request comparison against cloud alternatives.

#### Experiment 3: Comparative User Acceptance Study

**Objective**: Compare the voice assistant system against traditional phone-based room service in a controlled hotel environment.

**Method**:
- Deploy the system in 5-10 test rooms at a partner hotel in Sri Lanka.
- Measure: task completion rate, average request-to-fulfilment time, guest satisfaction scores (Likert scale), staff efficiency metrics.
- Compare against a control group using traditional phone-based service requests.
- Conduct semi-structured interviews with both guests and staff.

**Expected Contribution**: Empirical validation of operational efficiency gains and guest satisfaction in a real-world Sri Lankan hotel environment.

---

## Research Contribution Statement

> This research presents the first integrated, offline-capable voice assistant system specifically designed for hotel room service operations in developing economies, combining on-device speech recognition (Vosk), a fine-tuned MobileBERT intent classifier (92% accuracy across 18 hotel service categories), and a real-time staff management dashboard, all operating on commodity Android hardware without cloud dependency. The system addresses a critical gap at the intersection of edge AI, privacy-preserving computing, and hospitality technology by demonstrating that effective voice-based service automation is achievable at a fraction of the cost of commercial alternatives, while ensuring complete guest privacy through local-only data processing.

---

## References

### Category A: Voice Assistants in Hospitality

1. Buhalis, D. & Moldavska, I. (2021). In-room Voice-Based AI Digital Assistants Transforming On-Site Hotel Services and Guests' Experiences. In *Information and Communication Technologies in Tourism 2021*, Springer.
   - https://link.springer.com/chapter/10.1007/978-3-030-65785-7_3
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC7798082/

2. Buhalis, D. & Moldavska, I. (2022). Voice Assistants in Hospitality: Using Artificial Intelligence for Customer Service. *Journal of Hospitality and Tourism Technology*, 13(3), 386-403.
   - https://www.emerald.com/jhtt/article/13/3/386/219401/Voice-assistants-in-hospitality-using-artificial

3. Hwang, J. & Erdem, M. (2025). "U" in User Experience (UX) Stands for the Frontline Employee: A Case Study of Voice Assistant Technology Use in Hotels. *Journal of Hospitality & Tourism Cases*.
   - https://journals.sagepub.com/doi/10.1177/21649987251361910

4. Amazon Alexa for Hospitality / Alexa Smart Properties.
   - https://developer.amazon.com/en-US/alexa/alexasmartproperties/hospitality

5. Echoes of Innovation: Exploring the Use of Voice Assistants to Boost Hotel Reputation (2025). *Journal of Theoretical and Applied Electronic Commerce Research*, 20(1), 46.
   - https://www.mdpi.com/0718-1876/20/1/46

6. Amazon Alexa Devices in Hotels Raise Privacy Concerns (2018). Purdue University.
   - https://www.purdue.edu/newsroom/archive/releases/2018/Q2/amazon-alexa-devices-in-hotels-raise-privacy-concerns-for-some.html

### Category B: Offline/Edge-Based Speech Recognition

7. Korkmaz, A. et al. (2025). Real-Time Speech-to-Text on Edge: A Prototype System for Ultra-Low Latency Communication with AI-Powered NLP. *Information*, 16(8), 685. MDPI.
   - https://www.mdpi.com/2078-2489/16/8/685

8. Improving Speech Recognition Accuracy Using Custom Language Models with the Vosk Toolkit (2025). arXiv.
   - https://arxiv.org/html/2503.21025v1

9. Comparative Analysis of Vosk Toolkit and Other Speech Recognition Frameworks (2025). Preprints.org.
   - https://www.preprints.org/manuscript/202505.0654

10. Chantrapornchai, C. & Suchato, A. (2022). IoT Device Control with Offline Automatic Speech Recognition on Edge Device. *IEEE Conference Publication*.
    - https://ieeexplore.ieee.org/document/10010962/

11. Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). *ICML 2023*.
    - https://arxiv.org/abs/2212.04356

12. Privacy-Preserving On-Device Speech Recognition Using Vosk with Domain-Specific Language Models (2025). ResearchGate.
    - https://www.researchgate.net/publication/391807206

13. Alpha Cephei. Vosk Offline Speech Recognition API.
    - https://alphacephei.com/vosk/

### Category C: Small-Scale NLU Models

14. Sun, Z. et al. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. *ACL 2020*.
    - https://aclanthology.org/2020.acl-main.195/
    - https://arxiv.org/abs/2004.02984

15. Sanh, V. et al. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *NeurIPS 2019 Workshop*.
    - https://arxiv.org/abs/1910.01108

16. Bujel, K. et al. (2021). Effectiveness of Pre-training for Few-shot Intent Classification. arXiv.
    - https://arxiv.org/abs/2109.05782

17. Bunk, T. et al. (2020). DIET: Dual Intent and Entity Transformer. Rasa.
    - https://rasa.com/blog/introducing-dual-intent-and-entity-transformer-diet-state-of-the-art-performance-on-a-lightweight-architecture

18. Fine-Tuning BERT Models for Intent Recognition Using a Frequency Cut-Off Strategy for Domain-Specific Vocabulary Extension (2022). *Applied Sciences*, 12(3), 1610. MDPI.
    - https://www.mdpi.com/2076-3417/12/3/1610

### Category D: Privacy-Preserving AI and Edge Computing

19. Shi, W. et al. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637-646.
    - https://ieeexplore.ieee.org/document/7488250

20. Empowering Edge Intelligence: A Comprehensive Survey on On-Device AI Models (2025). *ACM Computing Surveys*.
    - https://dl.acm.org/doi/10.1145/3724420

### Category E: Task-Oriented Dialogue Systems

21. Larson, S. et al. (2022). A Survey of Intent Classification and Slot-Filling Datasets for Task-Oriented Dialog. arXiv.
    - https://arxiv.org/abs/2207.13211
    - https://www.researchgate.net/publication/362301379

22. Improved Spoken Language Representation for Intent Understanding in a Task-Oriented Dialogue System (2022). *Sensors*, 22(4), 1509. MDPI.
    - https://www.mdpi.com/1424-8220/22/4/1509

### Sri Lanka Context

23. Connecting the Connected: How Is Sri Lanka Prepared to Respond to Digital Tourists? (2022). Springer.
    - https://link.springer.com/chapter/10.1007/978-981-16-5461-9_22

24. Smart Tourism in the Digital Age: Overcoming Barriers and Unlocking New Possibilities (2024). *Revista de Gestao*, Emerald.
    - https://www.emerald.com/rege/article/32/3/224/1271283
