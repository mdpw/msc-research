# CHAPTER 9: PROJECT MANAGEMENT

## 9.1 Introduction

This chapter covers how the project was managed from planning through to submission. It describes the schedule, how risks were identified and handled, how code and model quality were maintained, and the social, legal, ethical, and professional considerations that shaped the work. Project management for a research prototype differs from a typical software project — decisions were often driven by what the research needed to find out rather than a fixed requirements list, and the iterative development approach shaped how the schedule and risk responses were structured.

---

## 9.2 Project Schedule

### 9.2.1 Work Breakdown Structure

The project was divided into six work packages, each with specific deliverables and a defined timeframe. Table 9.1 shows the breakdown.

**Table 9.1: Work Breakdown Structure**

| Work Package | Activities | Planned Duration |
|-------------|-----------|-----------------|
| WP1: Research and Planning | Literature review, requirements gathering, hotel management interviews, guest survey, technology evaluation | Weeks 1–4 |
| WP2: Dataset Development and Model Training | Hotel intent dataset creation (10,080 utterances), Vosk transcription pairing, three-model MobileBERT training, TFLite conversion, accuracy evaluation | Weeks 4–7 |
| WP3: Android Application Development | Vosk integration, hybrid NLU pipeline, voice UI in Jetpack Compose, TTS, request submission flow, WebSocket client | Weeks 6–10 |
| WP4: Backend and Dashboard Development | FastAPI server, SQLite database design, WebSocket implementation, staff dashboard HTML/CSS/JS | Weeks 8–11 |
| WP5: Integration and Testing | End-to-end integration, system testing, NLU and WER evaluation, bug fixes | Weeks 11–14 |
| WP6: Report Writing and Submission | Dissertation writing, diagrams, appendices, proofreading, final submission | Weeks 10–18 |

**Figure 9.1: Project Gantt Chart**

*(See attached Gantt chart — Figure 9.1)*

```
Week:        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
             |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
WP1 Research [========]
WP2 Dataset        [======]
                       M2▲ (>90% NLU accuracy)
WP3 Android               [============]
                                      M3▲ (Voice-to-text working)
WP4 Backend                    [=========]
WP5 Testing                               [========]
                                              M4▲ (End-to-end prototype)
                                                     M5▲ (Evaluation done)
WP6 Writing                         [========================]
             |--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
M1▲ = Lit review done (Week 4)                       M6▲ = Submission (Week 18)
```

WP3 (Android) and WP4 (Backend) were developed in parallel from Week 8, which was an intentional decision — the Android app and FastAPI server have well-defined API contracts, so they could be built independently and integrated in WP5. Report writing (WP6) began in Week 10 and ran concurrently with implementation to ensure documentation stayed current.

### 9.2.2 Schedule Adherence

The project broadly followed the planned timeline, with two deviations worth noting.

**Deviation 1 — Model conversion took longer than expected.** Converting MobileBERT from PyTorch to TFLite was estimated at two to three days but took close to a week. The main difficulty was building a compatible BERT word-piece tokeniser in Kotlin from scratch, since the standard HuggingFace tokeniser is Python-only and cannot run on Android. The tokeniser had to correctly handle subword splitting, special tokens ([CLS], [SEP], [UNK]), padding, and truncation in a way that matched exactly what the model expected. This was tedious to verify and required careful debugging against known test inputs. The delay was absorbed within the buffer in WP3.

**Deviation 2 — Report writing started later than planned.** The original plan had writing beginning in Week 10, but WebSocket stability issues in Iteration 3 (unexpected client disconnections when the device screen turned off) demanded more focused engineering time. Writing began in earnest in Week 12. This was managed by maintaining detailed notes during implementation and increasing writing effort in the final weeks.

All planned deliverables were completed within the project timeline. The iterative development approach provided natural review points at the end of each iteration, which helped catch schedule risks before they became critical problems.

---

## 9.3 Risk Management

A risk register was created at the start of the project and reviewed at the end of each development iteration. Risks were rated by likelihood and impact (Low / Medium / High) and given a mitigation strategy before any implementation began.

### 9.3.1 Risk Register

**Table 9.2: Risk Register**

| ID | Risk | Likelihood | Impact | Mitigation Strategy | Outcome |
|----|------|-----------|--------|--------------------|----|
| R1 | Vosk speech recognition accuracy insufficient for hotel domain | Medium | High | Test multiple Vosk model variants early; implement hybrid NLU to compensate for STT errors; consider custom language model if needed | Partially materialised — the Indian English model (`vosk-model-small-en-in-0.4`) had a measured WER of 11.43% on synthesised speech. The hybrid keyword pipeline partially compensates for transcription errors on common requests. |
| R2 | MobileBERT model too large or slow for budget Android hardware | Medium | High | Profile model size and inference latency in Iteration 2; have DistilBERT as backup; apply quantisation if needed | Did not materialise — the 26MB TFLite model runs within acceptable latency on test hardware (~62ms inference). |
| R3 | TFLite conversion fails or produces incorrect outputs | Medium | Medium | Test conversion early; verify output logits against PyTorch model using shared test inputs; keep server-side inference as fallback | Partially materialised — initial conversion produced mismatched label ordering between PyTorch and TFLite outputs. Resolved by verifying the label map against both models before deployment. |
| R4 | Budget Android tablets lack sufficient RAM for Vosk + MobileBERT simultaneously | High | High | Test on lowest-cost target hardware in Iteration 1; select lightweight model variants; stagger model loading if needed | Materialised for the 1.8GB Vosk model (15+ second load times, OOM crashes). Mitigated by switching to `vosk-model-small-en-in-0.4` (~36MB), which loads in ~3 seconds with stable memory usage. |
| R5 | WebSocket connections unstable on hotel Wi-Fi | Medium | Medium | Implement exponential backoff reconnection; design system to degrade gracefully on disconnect | Materialised during Iteration 3 — client silently disconnected when device screen turned off. Mitigated by implementing automatic reconnection logic (2s → 4s → 8s → 16s → 30s capped backoff). |
| R6 | Insufficient hotel service training data available publicly | Medium | High | Create a custom dataset through template generation, paraphrase augmentation, and Vosk-noise pairing | Did not materialise — 10,080 labelled utterances were generated across 18 intent categories. Model C achieved 99.06% accuracy on the Vosk test set. |
| R7 | Unable to access hotels for requirements interviews | Low | Medium | Prepare alternative approach using published hospitality research and hotel service menus | Did not materialise — interviews were conducted as planned. |
| R8 | Project scope too ambitious for the available timeline | Medium | High | Define clear MoSCoW priorities in Chapter 4; implement Must Have requirements first; defer Could Have items to future work | Managed through prioritisation — all Must Have and Should Have requirements were implemented. Won't Have items are documented as future work in Chapter 11. |

### 9.3.2 Lessons from Materialised Risks

Three risks materialised (R1, R4, R5), and the mitigation strategy worked in each case. The most consequential was R4. The large Vosk model caused out-of-memory crashes on a budget Android tablet — but this was discovered in the first week of Iteration 1, not at the end of the project. This early discovery is exactly what iterative prototyping is designed to enable. Under a sequential waterfall approach, this constraint would only have been found during integration testing in a much later stage, potentially requiring a major redesign.

R1, the speech recognition accuracy risk, is still partially present. The 11.43% overall WER means roughly half of all utterances are transcribed with some degree of error. The hybrid keyword matching pipeline provides a partial safety net for the most common requests, and the noise-aware training in Model C absorbs much of the rest. But real-world performance with actual hotel guests speaking in rooms with background noise has not been formally measured, which remains a limitation.

---

## 9.4 Quality Management

### 9.4.1 Code Quality

The following practices were used to maintain code quality throughout development:

**Version control.** All source code was managed in a Git repository. Commit messages were kept descriptive, documenting not just what changed but why — particularly for decisions that emerged from implementation constraints, such as the switch to the smaller Vosk model or the addition of the rule-based keyword tier.

**Modular architecture.** The Android application separates concerns across dedicated classes: `AudioRecorder`, `VoskService`, `NLUService`, `ApiService`, `WebSocketService`, and `TextToSpeechService`. Each class has a single responsibility, which made individual components easier to test, debug, and replace. For example, swapping the Vosk model variant required changes only in `VoskService`, not across the entire application.

**Automatic request validation.** The FastAPI backend uses Pydantic models for all request and response bodies, providing automatic type checking at the API boundary. Malformed requests are rejected with clear error messages without writing any validation code manually.

**Naming conventions.** Kotlin conventions were followed in the Android application; PEP 8 conventions were followed in the Python backend and training scripts.

### 9.4.2 Model Quality

Quality in the NLU training pipeline was maintained through several practices:

**Stratified splitting.** The 85%/15% train/validation split was applied with stratification, ensuring all 18 intent categories were proportionally represented in both the training and validation sets. Without stratification, rare categories with fewer examples could have been underrepresented in the validation set.

**Early stopping.** Rather than training for a fixed number of epochs, early stopping with a patience of 2 epochs was applied using F1 macro as the monitoring metric. This prevented overfitting — training stopped automatically when the model stopped improving on the validation set.

**Three-model controlled comparison.** All three model variants (A, B, C) used identical hyperparameters and were evaluated on the same 2,016-sample held-out test set. This design ensures that differences in performance between models reflect training data choices, not hyperparameter differences or evaluation inconsistencies.

**Per-intent analysis.** Beyond overall accuracy, per-intent precision, recall, and F1-scores were computed for all 18 categories. This granular view revealed which categories were harder to classify — particularly those with high Vosk WER (temperature_control at 16.83%, towel_request at 16.33%) — and confirmed that Model C's improvements were consistent across all categories, not just aggregate averages.

**Confusion matrix review.** Confusion matrices were generated for all four evaluation runs. Comparing `model_a_vosk_gap.png` with `model_c_mixed.png` directly shows how noise-aware training eliminated the specific misclassification patterns that appeared when Model A faced Vosk-transcribed input.

### 9.4.3 System Quality

End-to-end system quality was validated through:

**Manual integration testing.** Each complete user flow — voice request submission, confirmation, cancellation, staff status update, staff-to-guest messaging, and service rating — was tested end-to-end on physical hardware (Android tablet, laptop server, browser dashboard).

**API endpoint testing.** The `test_api.py` script verified all REST API endpoints return correct responses, and that SQLite persistence operates correctly across requests.

**Cross-browser testing.** The staff dashboard was tested across Chrome, Firefox, and Edge on desktop and mobile to verify consistent behaviour.

**Reconnection testing.** The WebSocket reconnection logic was explicitly verified by restarting the server mid-session and observing the client reconnect automatically within expected backoff timing.

---

## 9.5 Social, Legal, Ethical and Professional Considerations

### 9.5.1 Data Protection and Privacy

Privacy is both a core motivation of this research and an ethical obligation during its conduct.

**Voice data.** The system processes all audio on the guest's Android device using Vosk. No audio is stored on the device, transmitted to the server, or sent to any external service. The audio buffer is processed in real time, converted to text, and immediately discarded. This provides an architectural privacy guarantee — voice data cannot leak because it never leaves the device. This is a fundamentally different position from cloud-based alternatives such as Alexa for Hospitality, where the privacy guarantee depends entirely on the service provider's policies.

**Request data.** Service request text, intent classifications, and status information are stored in the hotel's local SQLite database for operational purposes. This data remains within the hotel's local network. No personally identifiable guest information — names, passport numbers, payment details — is collected or processed at any point.

**Compliance.** Sri Lanka's Personal Data Protection Act No. 9 of 2022, while still in its implementation phase, establishes core principles of data minimisation and purpose limitation. This system collects only what is needed for service delivery (request text and room number) and uses it only for routing service requests to staff. The architecture is compliant with these principles by design.

### 9.5.2 Ethical Considerations

**Informed consent.** The system is entirely opt-in. Guests are not required to use the voice assistant, and traditional service request methods remain available. The microphone activates only when the guest physically presses the microphone button — there is no passive or always-on listening capability. This directly addresses the privacy concerns identified in the literature review and reinforced by the guest survey findings.

**Bias and fairness.** The Vosk model (`vosk-model-small-en-in-0.4`) was selected specifically because it is trained on Indian English acoustic data, making it a better match for South Asian accents than a US English model. Despite this, the WER results show variation across intent categories, and real-world performance for guests with stronger accents or non-standard phrasing has not been formally measured. The system does not make consequential decisions independently — all requests are routed to human staff for fulfilment regardless of classification confidence. Guests whose requests are misunderstood receive a "could not understand" response and can rephrase or use the telephone instead.

**Staff impact.** The system is designed to assist hotel staff, not replace them. Requests still require a human to fulfil — the system only automates the communication and routing step. Evidence from the hospitality literature suggests that voice assistants of this type tend to reduce repetitive administrative tasks and improve response times, which can improve working conditions rather than threaten employment.

**Research ethics.** Hotel management interviews and guest surveys were conducted with the informed consent of all participants. No vulnerable populations were involved. The research was conducted in accordance with [university name]'s ethics guidelines.

### 9.5.3 Legal Considerations

All technologies used in the prototype are open-source and free for commercial use:

**Table 9.3: Open-Source Licences**

| Technology | Licence |
|-----------|---------|
| Vosk (`vosk-model-small-en-in-0.4`) | Apache 2.0 |
| MobileBERT (HuggingFace `google/mobilebert-uncased`) | Apache 2.0 |
| TensorFlow / TensorFlow Lite | Apache 2.0 |
| FastAPI | MIT |
| SQLite | Public Domain |
| OkHttp | Apache 2.0 |
| Jetpack Compose / Android SDK | Apache 2.0 |

No proprietary software or paid API services are used anywhere in the system. A hotel deploying this solution would incur no ongoing licensing costs for software. The complete source code, trained model weights, and dataset are produced as research outputs and have no third-party IP restrictions.

The custom intent dataset and fine-tuned MobileBERT model created during this research are original contributions of the project and carry no licensing obligations for hotels adopting the system.

### 9.5.4 Professional Considerations

The project was conducted in line with the British Computer Society (BCS) Code of Conduct:

**Public interest.** The system aims to make functional AI technology accessible to small hotels in developing economies without requiring cloud infrastructure or technical expertise. Making privacy-preserving voice assistance practical at the price point of a budget Android tablet serves a genuine public interest.

**Professional competence.** Technology selections were based on systematic comparative evaluation (Chapter 5) rather than familiarity or preference. Limitations were documented honestly throughout — the key testing gap (WER measured on TTS-synthesised audio rather than real human speech) is explicitly acknowledged in Chapter 8 rather than glossed over.

**Duty to the profession.** This research contributes evidence on the practical feasibility of offline, on-device NLU using open-source edge components in a real deployment context. The three-model experimental design and the paired clean/Vosk dataset provide a replicable methodology that could be applied to other offline edge AI contexts beyond hospitality.

---

## 9.6 Summary

The project was planned around six work packages with deliberate overlap between Android and backend development, and with report writing running concurrently with implementation from mid-project onwards. The schedule held broadly, with two manageable deviations. Risk management proved effective — all three risks that materialised (large Vosk model crashing budget hardware, TFLite label ordering mismatch, WebSocket disconnections on screen-off) had mitigation strategies in place and were resolved without major rescheduling.

Quality was maintained through version control, modular code architecture, stratified and reproducible model evaluation, and hands-on integration testing on physical hardware. The system's privacy-by-design architecture — where voice data never leaves the guest device — satisfies both the research objectives and the ethical obligations of deploying a voice-enabled system in a private hotel room context. The following chapter presents a critical appraisal of the project and its outcomes.
