# CHAPTER 11: CONCLUSIONS

## 11.1 Introduction

This chapter draws together the outcomes of the project. It reviews what was achieved against each of the four research objectives, outlines the most valuable directions for future work, and closes with a reflection on the broader significance of the findings. The research set out to answer a single question: can a low-cost, fully offline voice assistant prototype, built on a standard Android device, achieve sufficient technical accuracy and performance to be a viable alternative to traditional room service communication in Sri Lankan hotels? This chapter summarises the evidence gathered and what it means for practical deployment.

---

## 11.2 Achievements

This research set out to demonstrate that a low-cost, offline voice assistant for hospitality services is achievable using small-scale neural models on commodity hardware. The following reviews each of the four research objectives defined in Chapter 1 against what was actually delivered.

**Objective 1: Design and develop a low-cost, offline voice assistant prototype deployable on commodity Android hardware.**

A fully functional prototype was built and demonstrated across four iterative development cycles. The system runs entirely without internet connectivity — all voice input, speech recognition, and intent classification happen on the guest's device. The prototype handles the complete request lifecycle from voice input through to staff notification, service completion, and guest rating, running on a commodity Android tablet in the $50–$150 price range.

**Objective 2: Build a hospitality-domain dataset and train a noise-aware NLU model.**

A custom dataset of 10,080 labelled utterances was created across 18 intent categories (560 per intent) mapped to 5 hotel departments, built through template-based generation and paraphrase augmentation. A paired Vosk-transcribed version (`vosk_transcriptions.csv`) was produced to enable noise-aware training. The `vosk-model-small-en-in-0.4` speech recognition model (~36MB, Indian English) and a fine-tuned MobileBERT model (26MB TFLite, dynamically quantised from 94MB PyTorch — a 72% reduction) were successfully deployed on-device. The hybrid NLU pipeline — combining rule-based keyword matching with MobileBERT neural inference — was an important practical innovation that emerged from implementation: keywords handle the most common, high-confidence requests directly, while the neural model handles the wider range of phrasing. To the best of the author's knowledge, this is the first publicly documented intent classification dataset designed specifically for hospitality service operations.

**Objective 3: Build a lightweight backend with real-time staff communication.**

A FastAPI backend with SQLite persistence and WebSocket-based real-time communication was implemented and demonstrated. The system supports bidirectional messaging between guest devices and a department-filtered staff dashboard. Incoming requests are routed to the correct department via a database-driven mapping (`intent_department_mapping` table), with a keyword-based Python fallback for edge cases. The entire backend runs on any low-cost laptop or PC — no cloud infrastructure required.

**Objective 4: Evaluate system accuracy, latency, and cost-effectiveness, and demonstrate privacy-preserving offline operation.**

Three MobileBERT model variants were trained and evaluated on a shared 2,016-sample held-out test set. Model A (clean-text trained) achieved 98.07% accuracy on clean input, but dropped to 89.34% on actual Vosk transcriptions — an 8.73 percentage point gap caused purely by the STT step. Model C (noise-aware) recovered this gap entirely, reaching 99.06% on Vosk output, representing 111.3% gap recovery. The Vosk WER across the full dataset was 11.43% overall, with per-intent WER ranging from 6.78% (`emergency`) to 16.83% (`temperature_control`). End-to-end latency reached P95 2,827ms — within the 5-second NFR target. The prototype also provides a structural privacy guarantee: all voice processing happens on the guest's device, no audio is stored, and no data is transmitted outside the hotel's local network. The entire system uses open-source software with no recurring licensing costs, making it a realistic deployment option for small hotels in developing economies.

### Overall Assessment

The research successfully addressed all four research gaps identified in the literature review: an integrated end-to-end offline voice assistant for hospitality; on-device NLU for hospitality-specific intent classification within a real STT pipeline; privacy-preserving, cost-effective deployment on commodity hardware without cloud dependency; and a system designed for the specific operational conditions of hotels in developing economies. The prototype demonstrates that the combination of edge AI, small-scale neural models, and hospitality technology is a viable space for practical development in developing economies.

---

## 11.3 Future Work

The prototype successfully demonstrates the core concept. The following items describe the most valuable next steps.

### 11.3.1 Field Deployment and User Study

The most important next step is a real-world deployment. Controlled evaluation on a synthetic dataset can only go so far — the system needs to be tested with real hotel guests speaking in actual hotel rooms. This would involve deploying the prototype across a number of rooms in a partner hotel for a trial period, collecting real guest interaction data (with appropriate consent), and conducting structured interviews with staff about the system’s impact on their workflow.

A field study would do several things at once: validate the speech recognition accuracy with real Sri Lankan English accents in room acoustic conditions, identify new intent categories that real guests actually use, and reveal usability issues that do not appear in controlled testing. It would also allow the 99.06% accuracy figure — currently measured on TTS-synthesised speech — to be re-evaluated under genuine production conditions.

### 11.3.2 Real Speech Data Collection and Model Retraining

The training dataset and WER measurements were based on text-to-speech audio passed through Vosk, not real human recordings. While this approach produced strong and reproducible results, the tokenizer mismatch between the Python evaluation pipeline and the Android deployment (see Section 10.4.8) means the actual on-device accuracy is lower than the reported 99.06%.

Future work should collect real voice recordings from Sri Lankan English speakers across a range of accents and acoustic conditions, use the actual Vosk transcriptions of those recordings as training data, and re-evaluate all three models. This would give a much more accurate picture of real-world performance and would likely improve the Android model’s practical accuracy.

### 11.3.3 Proper WordPiece Tokenizer on Android

The current Android implementation uses a simplified word-level tokenizer rather than the full HuggingFace WordPiece tokenizer used during Python training. This means words not in the vocabulary map to the unknown token, and sub-word decomposition does not happen. Implementing a proper WordPiece tokenizer in Kotlin — or switching to TFLite’s built-in tokenization support — would align on-device inference with the evaluation conditions and recover some of the accuracy gap between the reported figures and real Android performance.

### 11.3.4 Multilingual Support

The system currently supports English only. In the Sri Lankan hospitality context, some staff and guests communicate more naturally in Sinhala, Tamil, or a mix of languages. Vosk supports Sinhala as a separate model, and the MobileBERT intent classifier could be fine-tuned on multilingual data. Future work could explore:

- Sinhala and Tamil speech recognition using Vosk language model variants
- A multilingual intent classifier trained on English, Sinhala, and Tamil utterances
- Automatic language detection to route transcription to the appropriate model

This would meaningfully expand the system’s practical applicability across Sri Lanka’s hospitality sector.

### 11.3.5 Further NLU and Dialogue Improvements

Beyond the core limitations above, several improvements would strengthen the NLU layer: a custom Vosk language model trained on hospitality vocabulary (targeting high-WER intents such as `temperature_control` at 16.83% and `towel_request` at 16.33%), an explicit `out_of_scope` intent to prevent spurious staff notifications when guests ask non-service questions, and a lightweight dialogue state tracker to handle multi-turn requests such as "I’d like two towels and a bottle of water."

---

## 11.4 Summary

This research demonstrated that a low-cost, offline voice assistant for hospitality services is not just theoretically possible but practically buildable using small-scale neural models on commodity hardware. The prototype addresses all four research gaps identified in the literature: end-to-end offline voice processing with on-device NLU in a real STT pipeline, privacy-by-architecture without cloud dependency, cost-effective deployment on commodity hardware, and a solution designed for the practical constraints of hotels in developing economies.

The research demonstrates that the answer to the core question is yes. The prototype achieves sufficient technical accuracy and performance across all measured dimensions: NLU accuracy reaches 99.06% on real pipeline input (Model C), STT word error rate is acceptable for the hospitality domain, end-to-end latency falls within practical bounds, and the total hardware cost sits in the $50–$150 range. Together, these results make a credible case for viability as an alternative to manual room service communication. An important supporting finding is the three-model experimental design that quantifies the accuracy gap between clean-text NLU evaluation and real offline pipeline conditions, and demonstrates that noise-aware training closes it. Model A's 8.73 percentage point drop from clean to Vosk conditions — and Model C's 111.3% recovery to 99.06% on Vosk output — provides concrete, reproducible evidence for why noise-aware training is necessary for a reliable production pipeline.

The most important caveat remains the synthetic foundation of the evaluation: both the WER measurements and the training noise profiles were derived from TTS audio rather than real speaker recordings. Confirming these results with actual hotel guest speech, and deploying the system in a real hotel environment, are the most critical next steps before drawing firm conclusions about production readiness. The future work outlined above charts a clear path from the current prototype towards a system that could genuinely transform guest service delivery in the Sri Lankan hospitality sector and serve as a replicable model for similar contexts elsewhere. Chapter 12 reflects on the personal learning experience behind this work.
