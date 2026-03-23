# CHAPTER 11: CONCLUSIONS

## 11.1 Achievements

This research set out to demonstrate that a low-cost, offline voice assistant for hospitality services is achievable using small-scale neural models on commodity hardware. The following reviews each of the six research objectives defined in Chapter 1 against what was actually delivered.

**Objective 1: Design and develop a low-cost, offline-capable voice assistant prototype for hospitality services using small-scale neural models deployed on commodity Android devices.**

A fully functional prototype was built and demonstrated across four iterative development cycles. The system runs entirely without internet connectivity — all voice input, speech recognition, and intent classification happen on the guest's device. The prototype handles the complete request lifecycle from voice input through to staff notification, service completion, and guest rating, running on a commodity Android tablet in the $50–$150 price range.

**Objective 2: Implement on-device speech recognition and intent classification using small-scale neural models (Vosk, MobileBERT) optimised for low-resource deployment.**

The `vosk-model-small-en-in-0.4` speech recognition model (~36MB, Indian English) and a fine-tuned MobileBERT model (26MB TFLite, dynamically quantised from 94MB PyTorch — a 72% reduction) were successfully deployed on-device. Both models run on a budget Android tablet without GPU requirements. The hybrid NLU pipeline — combining rule-based keyword matching with MobileBERT neural inference — was an important practical innovation that emerged from implementation: keywords handle the most common, high-confidence requests directly, while the neural model handles the wider range of phrasing, giving better overall reliability than either approach alone.

**Objective 3: Build a lightweight backend system using FastAPI and SQLite for managing service requests with real-time staff communication via WebSocket.**

A FastAPI backend with SQLite persistence and WebSocket-based real-time communication was implemented and demonstrated. The system supports bidirectional messaging between guest devices and a department-filtered staff dashboard. Incoming requests are routed to the correct department via a database-driven mapping (`intent_department_mapping` table), with a keyword-based Python fallback for edge cases. The entire backend runs on any low-cost laptop or PC — no cloud infrastructure required.

**Objective 4: Evaluate the system's speech recognition accuracy, intent classification performance, and the accuracy impact of the real STT pipeline on NLU.**

Three MobileBERT model variants were trained and evaluated on a shared 2,016-sample held-out test set. Model A (clean-text trained) achieved 98.07% accuracy on clean input, but dropped to 89.34% when given actual Vosk transcriptions — an 8.73 percentage point gap caused purely by the STT step. Model C (noise-aware, trained on the mixed paired dataset) recovered this gap entirely, reaching 99.06% accuracy on Vosk output — actually exceeding the clean baseline, representing 111.3% gap recovery. The Vosk WER across the full 10,080-utterance dataset was 11.43% overall (23.84% on sentences that were changed), with per-intent WER ranging from 6.78% (`emergency`) to 16.83% (`temperature_control`). These results confirm the core research hypothesis.

**Objective 5: Develop a hospitality-domain intent classification dataset covering 18 service request categories relevant to Sri Lankan hotel operations.**

A custom dataset of 10,080 labelled utterances was created across 18 intent categories (560 per intent) mapped to 5 hotel departments. The dataset was built through template-based generation and paraphrase augmentation, with a paired Vosk-transcribed version (vosk_transcriptions.csv) enabling the noise-aware training strategy. To the best of the author's knowledge, this is the first publicly documented intent classification dataset designed specifically for hospitality service operations.

**Objective 6: Demonstrate that privacy-preserving, offline voice-based service automation is achievable on low-cost hardware without cloud dependency.**

The prototype provides a structural privacy guarantee. All voice processing happens on the guest's device using Vosk — no audio is stored, and no data is transmitted outside the hotel's local network. This is a stronger assurance than policy-based privacy (as with cloud alternatives like Alexa for Hospitality), because the architecture has no mechanism to send audio externally. The entire system uses open-source software with no recurring licensing costs, making it a realistic deployment option for small hotels in developing economies.

### Overall Assessment

The research successfully addressed all five research gaps identified in the literature review: an integrated end-to-end offline voice assistant for hospitality; on-device NLU for hospitality-specific intent classification; privacy-preserving voice processing without cloud dependency; real-time bidirectional communication between guest devices and staff systems; and cost-effective deployment on commodity hardware. The prototype demonstrates that the combination of edge AI, small-scale neural models, and hospitality technology is a viable space for practical innovation in developing economies.

---

## 11.2 Future Work

The prototype successfully demonstrates the core concept. The following items describe the most valuable next steps, organised roughly by priority and feasibility.

### 11.2.1 Field Deployment and User Study

The most important next step is a real-world deployment. Controlled evaluation on a synthetic dataset can only go so far — the system needs to be tested with real hotel guests speaking in actual hotel rooms. This would involve deploying the prototype across a number of rooms in a partner hotel for a trial period, collecting real guest interaction data (with appropriate consent), and conducting structured interviews with staff about the system's impact on their workflow.

A field study would do several things at once: validate the speech recognition accuracy with real Sri Lankan English accents in room acoustic conditions, identify new intent categories that real guests actually use, and reveal usability issues that do not appear in controlled testing. It would also allow the 99.06% accuracy figure — currently measured on TTS-synthesised speech — to be re-evaluated under genuine production conditions.

### 11.2.2 Real Speech Data Collection and Model Retraining

The training dataset and WER measurements were based on text-to-speech audio passed through Vosk, not real human recordings. While this approach produced strong and reproducible results, the tokenizer mismatch between the Python evaluation pipeline and the Android deployment (see Section 10.3.8) means the actual on-device accuracy is lower than the reported 99.06%.

Future work should collect real voice recordings from Sri Lankan English speakers across a range of accents and acoustic conditions, use the actual Vosk transcriptions of those recordings as training data, and re-evaluate all three models. This would give a much more accurate picture of real-world performance and would likely improve the Android model's practical accuracy.

### 11.2.3 Proper WordPiece Tokenizer on Android

The current Android implementation uses a simplified word-level tokenizer rather than the full HuggingFace WordPiece tokenizer used during Python training. This means words not in the vocabulary map to the unknown token, and sub-word decomposition does not happen. Implementing a proper WordPiece tokenizer in Kotlin — or switching to TFLite's built-in tokenization support — would align on-device inference with the evaluation conditions and recover some of the accuracy gap between the reported figures and real Android performance.

### 11.2.4 Multilingual Support

The system currently supports English only. In the Sri Lankan hospitality context, some staff and guests communicate more naturally in Sinhala, Tamil, or a mix of languages. Vosk supports Sinhala as a separate model, and the MobileBERT intent classifier could be fine-tuned on multilingual data. Future work could explore:

- Sinhala and Tamil speech recognition using Vosk language model variants
- A multilingual intent classifier trained on English, Sinhala, and Tamil utterances
- Automatic language detection to route transcription to the appropriate model

This would meaningfully expand the system's practical applicability across Sri Lanka's hospitality sector.

### 11.2.5 Custom Language Model for Hospitality Vocabulary

The literature (Section 2.3.1) shows that custom Vosk language models trained on domain-specific vocabulary can achieve up to 40% WER reduction for specialist domains. A hospitality-specific language model trained on words like "concierge", "housekeeping", "amenities", and "complimentary" — and adapted for South Asian English pronunciation — would reduce the transcription errors that drive accuracy degradation in the NLU pipeline. This is particularly relevant for the high-WER intents like `temperature_control` (16.83%) and `towel_request` (16.33%), where Vosk consistently struggles with domain-specific vocabulary.

### 11.2.6 Out-of-Scope Query Detection

The current system always assigns one of the 18 hotel service intents regardless of what the guest actually says. If a guest asks "What's the weather like tomorrow?" the system will still classify it as something and potentially send a spurious request to staff. Adding an explicit `out_of_scope` intent trained on non-service utterances would allow the system to recognise when a request falls outside its scope and respond helpfully — for example, "I can help with hotel services like room service and housekeeping. For other questions, please contact the front desk." This would reduce unnecessary staff notifications and build trust in the system over time.

### 11.2.7 Wake Word Detection

The prototype requires the guest to press a microphone button to initiate a request. A hands-free wake word system (for example, "Hey Hotel") would make the interaction more natural, particularly for accessibility use cases. Future work should implement a lightweight wake word detector with an always-listening low-power mode that activates the full voice pipeline only on wake word detection, keeping false positive rates low enough that background television or conversation does not trigger unwanted requests.

### 11.2.8 Multi-Turn Dialogue and Complex Requests

The current prototype handles single-turn, single-intent commands. Real guest interactions are often more complex:

- Multi-item requests: "I'd like two towels and a bottle of water"
- Follow-up references: "Actually, make that three"
- Clarification exchanges: "Did you say room cleaning? Would you like that now or later?"
- Conditional requests: "If the restaurant is still open, I'd like to order dinner"

Handling these would require a lightweight dialogue state tracker on the device. This is a more significant engineering undertaking but would substantially increase the system's practical usefulness.

### 11.2.9 Production Infrastructure

Several design decisions were made deliberately to keep the prototype simple and focused. Moving towards production deployment would require addressing them:

| Area | Prototype | Production Recommendation |
|------|-----------|--------------------------|
| Database | SQLite | PostgreSQL for concurrent multi-room access |
| Authentication | None | JWT-based device registration and role-based staff login |
| Device provisioning | Manual room number entry | Automated QR code or admin portal registration |
| Server deployment | Single Uvicorn process | Docker + Nginx + Gunicorn with crash recovery |
| Communication security | HTTP over local Wi-Fi | HTTPS with TLS, even on local network |
| Logging | Print statements | Structured logging with severity levels and timestamps |
| Backups | Manual file copy | Automated scheduled database backups |

None of these are fundamental redesigns — the architecture is sound. They are engineering tasks that turn a working prototype into a reliable deployed service.

### 11.2.10 RAG-Based Hotel Information Assistant

The system handles service requests but cannot answer informational queries such as "What time does the restaurant close?" or "Is there a pool?" A retrieval-augmented generation (RAG) module could allow hotel-specific information — facility hours, policies, local attractions — to be stored in a local vector database and queried by the voice assistant. Requests identified as informational rather than service requests would be routed to this module, which would retrieve and return relevant answers using a small on-device language model. This would extend the system from a request-routing tool into a genuine guest information assistant.

### 11.2.11 Hotel Management System Integration

In a real hotel, the voice assistant would need to connect with existing operational systems. Property management system (PMS) integration would allow service requests to be automatically associated with guest reservation records, enabling personalised interactions and linking requests to billing. Housekeeping system integration would synchronise room cleaning requests with existing scheduling tools to avoid duplicate task assignments. Exposing a well-documented REST API for third-party systems to subscribe to voice assistant events would allow hotels to integrate the prototype with their existing technology stack without modifying its core code.

### 11.2.12 Analytics and Reporting

The prototype stores all service requests in SQLite but does not surface analytical insights from that data. A reporting dashboard showing request volumes by department, average response times, peak service hours, and guest satisfaction trends would give hotel management a concrete operational return on investment from the system — not just improved guest experience, but data-driven insight into service delivery patterns.

### 11.2.13 Offline Request Queuing

The current prototype performs speech recognition and intent classification entirely on-device, but request submission still requires an active connection to the hotel server. If the server is temporarily unreachable — due to a Wi-Fi dropout or server restart — the classified request is lost and the guest receives no confirmation. Future work should implement a local request queue on the Android device: when submission fails, the request is stored in a local SQLite database with a pending-sync status, and the guest is notified that their request has been saved and will be sent shortly. A background sync service would retry submission automatically once connectivity is restored, then clear the local queue and deliver a confirmation. This would complete the resilience benefit already present in the on-device processing architecture and directly improve guest satisfaction by ensuring no request is silently dropped due to a transient network issue.

---

## 11.3 Summary

This research demonstrated that a low-cost, offline voice assistant for hospitality services is not just theoretically possible but practically buildable using small-scale neural models on commodity hardware. The prototype addresses all five research gaps identified in the literature: end-to-end offline voice processing, on-device NLU for hospitality intent classification, privacy-by-architecture without cloud dependency, real-time guest-to-staff communication, and deployment at a price point accessible to small hotels in developing economies.

The primary research contribution beyond the working system is the three-model experimental design that quantifies the accuracy gap between clean-text NLU evaluation and real offline pipeline performance, and demonstrates that Vosk-specific noise-aware training closes it. Model A's 8.73 percentage point drop from clean to Vosk conditions — and Model C's 111.3% recovery to 99.06% on Vosk output — provides concrete, reproducible evidence for a finding that standard NLU benchmarking would miss entirely.

The most important caveat remains the synthetic foundation of the evaluation: both the WER measurements and the training noise profiles were derived from TTS audio rather than real speaker recordings. Confirming these results with actual hotel guest speech, and deploying the system in a real hotel environment, are the most critical next steps before drawing firm conclusions about production readiness. The future work outlined above charts a clear path from the current prototype towards a system that could genuinely transform guest service delivery in the Sri Lankan hospitality sector and serve as a replicable model for similar contexts elsewhere.
