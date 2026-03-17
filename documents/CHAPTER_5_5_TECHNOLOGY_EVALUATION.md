# 5.5 Technology Evaluation and Selection

The selection of technologies for each component of this research prototype was guided by five core constraints derived from the requirements established in Chapter 4: offline capability (NFR-01), privacy preservation (NFR-02), low response latency (NFR-03), low-cost hardware deployment (NFR-04), and minimal IT infrastructure overhead (NFR-08). It is important to note that these constraints reflect the dual objectives of this research: to produce a functionally valid prototype for evaluating the core research hypotheses, and to demonstrate a credible pathway to production deployment. Where prototype-specific simplifications were adopted — for example, SQLite over PostgreSQL — the corresponding production-grade alternatives are explicitly identified. Table 5.x summarises the selected technology stack, with a full comparative justification presented in the subsections that follow.

**Table 5.x: Technology Stack Summary**

| Component | Prototype Selection | Production Recommendation | Primary Reason for Prototype Choice |
|-----------|--------------------|--------------------------|------------------------------------|
| Speech Recognition | Vosk (vosk-model-small-en-in-0.4, ~36MB) | Custom Vosk language model or Whisper (server-side) | Real-time streaming; native Android SDK; fully offline; Indian English accent support |
| Intent Classification | MobileBERT TFLite (26MB) + Rule-based hybrid | Expanded dataset; multi-language models | Purpose-built for mobile; 62ms latency; no server required |
| Backend Framework | FastAPI (Python) | FastAPI with load balancing | Native async + WebSocket; Python ML ecosystem |
| Database | SQLite | PostgreSQL | Zero-configuration; no DBA required; single-file |
| Real-Time Communication | WebSocket (OkHttp / Starlette) | WebSocket with message queue (Redis) | Full-duplex; no broker dependency |
| Mobile Platform | Android Native (Kotlin/Jetpack Compose) | Android with MDM provisioning | Low-cost hardware; native ML SDK; local availability |
| Text-to-Speech | Android Native TTS | Coqui TTS (custom hotel voice) | Pre-installed; zero cost; no additional model required |
| Staff Dashboard | Web-based (HTML/CSS/JS) | React/Vue with role-based auth | Browser-accessible; zero installation; rapid iteration |
| Server Configuration | Single IP/port (SharedPreferences) | MDM-managed certificates | Sufficient for controlled prototype evaluation |

---

## 5.5.1 Speech Recognition: Vosk vs Whisper vs Cloud APIs vs CMU Sphinx

Speech-to-text is the entry point of the entire system pipeline, making it the most critical component selection. Four candidate technologies were evaluated across the dimensions most relevant to the research constraints.

**Vosk** is an open-source offline speech recognition toolkit developed by Alpha Cephei, supporting over 20 languages with models ranging from 50MB to 1.8GB (Alpha Cephei, 2023). It provides real-time streaming recognition with word-level timestamps and operates entirely on-device without any network dependency.

**Whisper**, developed by OpenAI (Radford et al., 2023), is a large-scale model trained on 680,000 hours of multilingual data, achieving state-of-the-art accuracy across 99 languages. Its model sizes range from 39M parameters (tiny, ~150MB) to 1.55B parameters (large, ~6GB) (Radford et al., 2023). While Whisper tiny can run on Android, it requires batch processing rather than real-time streaming.

**Google Cloud Speech-to-Text and Amazon Transcribe** are commercial cloud-based solutions offering high accuracy across diverse accents but requiring constant internet connectivity and per-minute API charges, which scale with usage volume. Google Cloud Speech-to-Text is billed at $0.006 per 15 seconds of audio (Google Cloud, 2024), creating ongoing operational costs that are incompatible with the research context.

**CMU Sphinx (PocketSphinx)** is an older open-source offline recognition engine designed for embedded applications with very small footprints (~30MB) (Huggins-Daines et al., 2006), but built on acoustic modelling approaches that pre-date modern deep learning and has seen near-absent ongoing development since 2019 (CMU Sphinx, 2023).

**Table 5.x: Speech Recognition Technology Comparison**

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
| Satisfies NFR-01 (Offline) | Yes | Yes | **No** | Yes |
| Satisfies NFR-02 (Privacy) | Yes | Yes | **No** | Yes |

**Prototype Selection: Vosk (vosk-model-small-en-in-0.4, ~36MB)**

Vosk was selected for three decisive reasons. First, it provides native real-time streaming recognition — the system begins processing speech while the guest is still speaking, rather than waiting for a complete utterance. This behaviour is essential for achieving the 5-second end-to-end latency target (NFR-03). Whisper processes audio in batches after recording is complete, introducing additional delay incompatible with responsive guest interaction (Radford et al., 2023).

Second, Vosk provides a native Android SDK with straightforward Java/Kotlin integration (Alpha Cephei, 2023). Whisper's Python-native architecture requires ONNX conversion and a custom inference wrapper for Android, adding complexity and potential runtime instability. During early development, the larger Vosk model (1.8GB) was tested on a budget Android tablet but caused application load times exceeding 15 seconds, leading to model size reduction. The `vosk-model-small-en-in-0.4` variant (~36MB) was ultimately selected not only for its minimal resource footprint — an order of magnitude smaller than the general US English model — but critically because it is trained on Indian English acoustic data, which provides meaningfully better recognition of South Asian accent patterns. Sri Lankan English shares substantial phonological similarities with Indian English, including vowel quality, intonation contour, and consonant articulation, making this model a stronger acoustic fit for the target deployment population than a US English model of any size.

Third, Vosk processes all audio on-device, providing an architectural guarantee that raw voice data never leaves the guest's tablet. This directly satisfies the privacy preservation requirement (NFR-02) and eliminates the cross-border data transfer concerns associated with cloud STT services.

**Production Pathway:** For a production deployment, a custom Vosk language model trained on hotel-specific vocabulary would meaningfully reduce the Word Error Rate for domain terms such as "concierge," "housekeeping," and "amenities." Alternatively, server-side Whisper (medium or large) could be deployed on the hotel server for superior accent robustness, particularly relevant for Sri Lankan English speakers, while retaining privacy by remaining within the hotel's local network.

Cloud-based solutions were excluded as they fundamentally violate the offline (NFR-01) and privacy (NFR-02) requirements. CMU Sphinx was excluded due to significantly lower recognition accuracy compared to modern neural approaches and near-absent ongoing development since 2019 (CMU Sphinx, 2023).

---

## 5.5.2 Intent Classification: MobileBERT vs DistilBERT vs Rasa DIET vs Rule-Based

Intent classification determines how a transcribed guest utterance is understood and routed to the appropriate hotel department. Four approaches were evaluated.

**Table 5.x: Intent Classification Technology Comparison**

| Criteria | MobileBERT (TFLite) | DistilBERT (TFLite) | Rasa DIET | Rule-Based Only |
|----------|:---:|:---:|:---:|:---:|
| Model Size | ~26MB | ~67MB | ~100MB+ server | <1MB |
| On-Device Inference | Yes | Yes | No (server-side) | Yes |
| Inference Latency (mobile) | ~62ms | ~150ms | N/A | <5ms |
| Classification Accuracy | High (92% on test set) | High | High | Moderate |
| Handles Natural Phrasing Variation | Yes | Yes | Yes | No |
| Requires Running Server | No | No | Yes | No |
| Native TFLite Conversion | Yes | Requires extra steps | Not supported | N/A |
| Purpose-Built for Mobile | Yes | No | No | N/A |
| Satisfies NFR-03 (Latency) | Yes | Marginal | Dependent on server | Yes |

**Prototype Selection: Hybrid MobileBERT TFLite + Rule-Based Pipeline**

MobileBERT was selected as the neural classifier because it is the only transformer model architecturally designed for resource-limited mobile devices. Trained using progressive knowledge distillation from an inverted-bottleneck BERT-LARGE teacher model (Sun et al., 2020), MobileBERT achieves 4.3x compression and 5.5x speedup over BERT-BASE with a GLUE benchmark score of 77.7 (Sun et al., 2020). Its TFLite model size of 26MB is less than half that of DistilBERT (~67MB) (Sanh et al., 2019), and its 62ms inference latency satisfies the real-time response requirement.

DistilBERT (Sanh et al., 2019), while achieving comparable accuracy, offers no latency or size advantage in a mobile deployment context. Rasa DIET (Bunk et al., 2020) was excluded because it requires a persistent Rasa server process, conflicting with the on-device processing requirement (NFR-02). A purely rule-based approach was insufficient as the sole classifier because it cannot capture the full linguistic variation of natural speech — guests express the same intent in ways that no finite keyword list can exhaustively enumerate.

However, evaluation during prototype development revealed a recurring failure mode: the neural model occasionally misclassified simple, unambiguous requests (e.g., "I need towels" classified as pillow_request with 0.72 confidence). This led to the adoption of a hybrid two-tier pipeline. Rule-based keyword matching handles unambiguous, high-frequency requests at a fixed 0.99 confidence score; MobileBERT handles linguistically complex or ambiguous utterances as fallback. A minimum confidence threshold of 0.60 rejects low-confidence classifications rather than routing incorrectly.

This hybrid approach addresses a practical limitation of deploying neural models at prototype scale: the training dataset of approximately 5,000 examples, while sufficient to demonstrate feasibility, is smaller than ideal for a purely neural classifier. The rule-based tier compensates for model uncertainty in common cases while the neural model handles language variety that rules cannot capture.

**Prototype vs Production Distinction:** The rule-based keyword dictionary in the prototype requires manual maintenance — each new phrasing variant must be explicitly added. In a production system, a larger training dataset (50,000+ examples), continuous learning from real guest interactions, and multi-language model support would reduce dependence on manually curated rules.

---

## 5.5.3 Backend Framework: FastAPI vs Flask vs Django vs Express.js

**Table 5.x: Backend Framework Comparison**

| Criteria | FastAPI | Flask | Django | Express.js |
|----------|:---:|:---:|:---:|:---:|
| Native Async Support | Yes | No | Partial (v3.1+) | Yes |
| Native WebSocket Support | Yes (Starlette) | Requires Flask-SocketIO | Requires Django Channels | Requires Socket.IO |
| Automatic API Documentation | Yes (Swagger/OpenAPI) | No | No | No |
| Python ML Ecosystem | Yes | Yes | Yes | No |
| Dependency Footprint | Light | Light | Heavy | Light |
| Request Validation | Built-in (Pydantic) | Manual | Built-in (Forms) | Manual |
| Performance (requests/sec) | High | Moderate | Moderate | High |

**Prototype Selection: FastAPI**

FastAPI (Ramírez, 2024) was selected primarily for its native asynchronous support, which is essential for simultaneously managing WebSocket connections from multiple guest room devices and the staff dashboard without blocking on I/O operations. Flask's synchronous request model would require the Flask-SocketIO extension, introducing an additional dependency and a different concurrency model. Django's full-stack architecture introduces substantial overhead — the system requires a focused API server and real-time WebSocket hub, not a complete web framework with ORM, templating, and admin interface.

Express.js was excluded because the entire ML ecosystem, including model training, evaluation, and the NLU classification pipeline, is Python-based. Introducing a Node.js backend would require maintaining two language environments and duplicating business logic such as department routing, creating unnecessary complexity.

FastAPI's Pydantic-based request validation was particularly valuable during prototype development, automatically rejecting malformed API requests and providing clear error messages during debugging. The automatic Swagger documentation at `/docs` enabled rapid testing of API endpoints without additional tooling.

**Prototype vs Production Distinction:** The prototype runs FastAPI as a single process, sufficient for a controlled research environment. A production deployment serving multiple hotel properties would introduce a load balancer (nginx), multiple FastAPI worker processes, and a Redis-backed WebSocket message queue to ensure reliability and horizontal scalability.

---

## 5.5.4 Database: SQLite vs PostgreSQL vs MySQL vs MongoDB

**Table 5.x: Database Technology Comparison**

| Criteria | SQLite | PostgreSQL | MySQL | MongoDB |
|----------|:---:|:---:|:---:|:---:|
| Server Process Required | No | Yes | Yes | Yes |
| Configuration Required | None | Moderate | Moderate | Moderate |
| Concurrent Write Performance | Limited | Excellent | Good | Good |
| Storage Footprint | Minimal (<1MB per hotel) | ~100MB+ | ~200MB+ | ~300MB+ |
| Data Model | Relational | Relational | Relational | Document |
| DBA Expertise Required | None | Yes | Yes | Yes |
| Backup Method | Copy single file | pg_dump | mysqldump | mongodump |

**Prototype Selection: SQLite**

SQLite (Hipp, 2024) was selected because the expected data volume and concurrency profile of a research prototype falls well within its capabilities. A single hotel property with 50 rooms generating an estimated 100-200 requests per day produces negligible write concurrency — far below SQLite's practical limits. SQLite requires no separate server process, no installation configuration, and no ongoing database administration, directly supporting the minimal IT infrastructure requirement (NFR-08).

The system's data is inherently relational — requests reference rooms, staff messages reference requests — which excluded document-oriented MongoDB as it would artificially complicate relationship management. PostgreSQL and MySQL were not excluded for technical deficiency but for operational complexity. Deploying either would require installation, user account configuration, network binding, and ongoing maintenance, assuming IT expertise that cannot be guaranteed in small Sri Lankan hotel environments.

**Prototype vs Production Distinction:** SQLite's limited concurrent write support becomes a genuine constraint at scale. A production deployment handling 500+ rooms across multiple properties would require migration to PostgreSQL, which offers ACID-compliant concurrent writes, connection pooling via PgBouncer, and robust backup tooling. The data schema and FastAPI CRUD operations are designed to be migration-compatible, requiring only driver and connection string changes rather than structural redesign.

---

## 5.5.5 Real-Time Communication: WebSocket vs HTTP Polling vs Server-Sent Events vs MQTT

**Table 5.x: Real-Time Communication Technology Comparison**

| Criteria | WebSocket | HTTP Polling | Server-Sent Events | MQTT |
|----------|:---:|:---:|:---:|:---:|
| Bidirectional | Yes | Simulated | No (server → client only) | Yes |
| Latency | Low | High (polling interval) | Low | Low |
| Connection Overhead | Single handshake | Repeated connections | Single connection | Single connection |
| Requires External Broker | No | No | No | Yes (e.g., Mosquitto) |
| Native Android Support | Yes (OkHttp) | Yes | Limited | Requires library |
| Battery Impact on Tablets | Low | High | Low | Low |

**Prototype Selection: WebSocket (OkHttp on Android / Starlette on server)**

The WebSocket protocol (Fette and Melnikov, 2011) provides full-duplex bidirectional communication over a single persistent TCP connection, making it the most suitable protocol for the system's communication requirements. The system requires bidirectional communication across three endpoints: guest devices submitting requests and receiving status updates, the staff dashboard receiving new requests and sending messages, and the server broadcasting changes to all connected clients. This bidirectionality explicitly excludes Server-Sent Events, which are server-to-client only.

HTTP polling was excluded because it introduces unnecessary latency — a guest would not receive a "your request is in progress" update until the next polling cycle — and imposes repeated connection overhead with significant battery impact on tablet devices. MQTT (OASIS, 2019) was excluded because it requires a separate message broker process such as Mosquitto, adding infrastructure and configuration requirements that contradict the minimal IT overhead objective (NFR-08) for no material benefit in a point-to-point LAN communication scenario.

The prototype implements two categories of WebSocket endpoints: `/ws/guest/{room_number}` for per-room guest device connections, and `/ws/dashboard` for staff dashboard connections. The server maintains these connection registries in memory, which is appropriate for a prototype but would require persistent session storage in a production deployment.

**Prototype vs Production Distinction:** The in-memory WebSocket connection registry means connections are lost on server restart. A production deployment would use a Redis-backed channel layer (similar to Django Channels) to persist connection state across server restarts and enable horizontal scaling across multiple server instances.

---

## 5.5.6 Mobile Platform: Android Native vs iOS vs Flutter vs React Native

**Table 5.x: Mobile Platform Comparison**

| Criteria | Android (Kotlin) | iOS (Swift) | Flutter | React Native |
|----------|:---:|:---:|:---:|:---:|
| Device Cost Range (Sri Lanka) | $50–$150 | $300+ (iPad) | Depends on target | Depends on target |
| Local Hardware Availability | High | Limited | Target-dependent | Target-dependent |
| Vosk SDK Support | Native | Native | Community plugin | Community plugin |
| TFLite SDK Support | Native | Core ML (conversion needed) | Community plugin | Community plugin |
| On-Device TTS | Built-in | Built-in | Plugin required | Plugin required |
| ML Inference Performance | Native JNI | Native | Near-native | JS bridge overhead |
| Jetpack Compose / Material 3 | Yes | N/A | Via Flutter widgets | N/A |

**Prototype Selection: Android Native (Kotlin with Jetpack Compose)**

Android was selected for three reasons tied directly to the deployment context. First, commodity Android tablets are available at $50–$150 across local retailers in Sri Lanka, compared to $300+ for the cheapest iPads, directly satisfying the low-cost hardware requirement (NFR-04). Android devices can also be serviced and replaced through local technicians, unlike iOS devices requiring Apple Authorised Service Providers.

Second, both Vosk (Alpha Cephei, 2023) and TensorFlow Lite (Google, 2024a) provide first-party native Android SDKs. Community-maintained Flutter and React Native plugins for these SDKs introduce version compatibility risks and performance overhead that could destabilise core system functions during research evaluation.

Third, Jetpack Compose (Google, 2024b) with Material Design 3 provides a reactive UI paradigm well-suited to the system's state-driven interface — recording status, processing state, request list updates, and WebSocket-driven status changes all map naturally to Compose's observable state model, which proved critical in resolving a significant recomposition bug identified during development (discussed in Section 5.x).

**Prototype vs Production Distinction:** The prototype uses a hardcoded room number (`ROOM_NUMBER = "101"`) and a manually configured server IP address stored in SharedPreferences. In a production deployment, room-device association would be managed through a Mobile Device Management (MDM) platform such as VMware Workspace ONE or Microsoft Intune, enabling remote provisioning of room numbers and server credentials without physical device access.

---

## 5.5.7 Text-to-Speech: Android Native TTS vs Coqui TTS vs Amazon Polly

**Table 5.x: Text-to-Speech Technology Comparison**

| Criteria | Android Native TTS | Coqui TTS | Amazon Polly |
|----------|:---:|:---:|:---:|
| Additional Storage | None (pre-installed) | ~100-500MB per model | N/A (cloud) |
| Voice Quality | Acceptable | High (natural) | High (natural) |
| Custom Voice Support | No | Yes | Yes |
| Offline Operation | Yes | Yes | No |
| Configuration Required | None | Model download + setup | API key + internet |
| Cost | Free | Free | $4 per 1M characters |

**Prototype Selection: Android Native TTS**

Android's built-in TextToSpeech engine was selected for the prototype because it is pre-installed on all Android devices, requires zero additional storage, and needs no configuration beyond language and speech rate settings. For the purpose of evaluating the system's core functionality — request classification accuracy, latency, and service routing — voice naturalness is a secondary concern, and the native TTS is adequate for conveying confirmation messages and status updates.

**Prototype vs Production Distinction:** In a production hotel deployment, voice quality directly affects guest experience. A natural, consistent voice identity would be expected. Coqui TTS (Eren et al., 2021), an open-source neural TTS engine, would allow a custom hotel assistant voice to be trained and deployed on-device or on the hotel server, maintaining offline operation while significantly improving audio quality. Amazon Polly was excluded as it requires cloud connectivity, violating NFR-01, and incurs a cost of $4 per one million characters synthesised (Amazon Web Services, 2024).

---

## 5.5.8 Summary

The technology selections across all system components reflect a deliberate balance between prototype suitability and production viability. Each technology was chosen to satisfy the research constraints of offline capability, privacy preservation, low latency, low cost, and minimal infrastructure overhead. Where prototype-appropriate simplifications were adopted — SQLite over PostgreSQL, Android Native TTS over Coqui TTS, hardcoded device configuration over MDM provisioning — the production-grade alternatives have been identified to demonstrate that the proposed architecture is not merely an academic exercise but a credible foundation for real-world hotel deployment. The following chapter presents the implementation of this technology stack and the key engineering decisions made during development.

---

## References

Alpha Cephei (2023) *Vosk offline speech recognition* [Software]. Available at: https://alphacephei.com/vosk/ (Accessed: 8 March 2026).

Amazon Web Services (2024) *Amazon Polly pricing*. Available at: https://aws.amazon.com/polly/pricing/ (Accessed: 8 March 2026).

Bunk, T., Varshneya, D., Vlasov, V. and Nichol, A. (2020) 'DIET: Lightweight language understanding for dialogue systems', *arXiv preprint arXiv:2004.09936*. Available at: https://arxiv.org/abs/2004.09936.

CMU Sphinx (2023) *PocketSphinx* [Software repository]. Available at: https://github.com/cmusphinx/pocketsphinx (Accessed: 8 March 2026).

Eren, G., Gölge, E. and the Coqui TTS Team (2021) *Coqui TTS: A deep learning toolkit for text-to-speech* [Software]. Available at: https://github.com/coqui-ai/TTS (Accessed: 8 March 2026).

Fette, I. and Melnikov, A. (2011) *The WebSocket protocol*, RFC 6455. Internet Engineering Task Force (IETF). Available at: https://datatracker.ietf.org/doc/html/rfc6455.

Google (2024a) *TensorFlow Lite: Machine learning for mobile and edge devices*. Available at: https://www.tensorflow.org/lite (Accessed: 8 March 2026).

Google (2024b) *Jetpack Compose*. Available at: https://developer.android.com/jetpack/compose (Accessed: 8 March 2026).

Google Cloud (2024) *Cloud Speech-to-Text pricing*. Available at: https://cloud.google.com/speech-to-text/pricing (Accessed: 8 March 2026).

Hipp, D.R. (2024) *SQLite* [Software]. Available at: https://www.sqlite.org (Accessed: 8 March 2026).

Huggins-Daines, D., Kumar, M., Chan, A., Black, A.W., Ravishankar, M. and Waibel, A. (2006) 'Pocketsphinx: A free, real-time continuous speech recognition system for hand-held devices', in *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2006)*, Toulouse, France, May 2006, Vol. 1, pp. I-185–I-188.

OASIS (2019) *MQTT version 5.0*. OASIS Standard. Available at: https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html.

Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I. (2023) 'Robust speech recognition via large-scale weak supervision', in *Proceedings of the 40th International Conference on Machine Learning (ICML 2023)*, PMLR 202, pp. 28492–28518.

Ramírez, S. (2024) *FastAPI* [Software]. Available at: https://fastapi.tiangolo.com (Accessed: 8 March 2026).

Sanh, V., Debut, L., Chaumond, J. and Wolf, T. (2019) 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter', *arXiv preprint arXiv:1910.01108*. Available at: https://arxiv.org/abs/1910.01108.

Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y. and Zhou, D. (2020) 'MobileBERT: a compact task-agnostic BERT for resource-limited devices', in *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 2158–2170.
