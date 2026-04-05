# CHAPTER 11: CONCLUSIONS

## 11.1 Introduction

This chapter draws together the outcomes of the project. It reviews what was achieved against each of the four research objectives, outlines the most valuable directions for future work, and closes with a reflection on the broader significance of the findings.

---

## 11.2 Achievements

**Objective 1: Design and develop a low-cost, offline voice assistant prototype deployable on commodity Android hardware.**

A fully functional prototype was built across four iterative development cycles. The system runs without internet connectivity — speech recognition and intent classification happen on the guest's device. It handles the complete request lifecycle from voice input to staff notification, service completion, and guest rating, on a commodity Android tablet in the $50–$150 range.

**Objective 2: Build a hospitality-domain dataset and train a noise-aware NLU model.**

A custom dataset of 10,080 labelled utterances was created across 18 intent categories (560 per intent) mapped to 5 hotel departments. A paired Vosk-transcribed version was produced to enable noise-aware training. The `vosk-model-small-en-in-0.4` model (~36MB, Indian English) and a fine-tuned MobileBERT (26MB TFLite, quantised from 94MB PyTorch — 72% reduction) were deployed on-device via a hybrid NLU pipeline: keyword matching handles high-confidence requests, MobileBERT handles broader phrasing. To the best of the author's knowledge, this is the first publicly documented intent classification dataset for hospitality service operations.

**Objective 3: Build a lightweight backend with real-time staff communication.**

A FastAPI backend with SQLite persistence and WebSocket-based communication was implemented. Requests are routed to the correct department via the `intent_department_mapping` table. The staff dashboard supports bidirectional messaging and department filtering. The entire backend runs on any low-cost laptop — no cloud infrastructure required.

**Objective 4: Evaluate system accuracy, latency, and cost-effectiveness, and demonstrate privacy-preserving offline operation.**

Three MobileBERT variants were evaluated on a shared 2,016-sample held-out test set. Model A dropped from 98.07% on clean text to 89.34% on Vosk transcriptions — an 8.73 percentage point gap from the STT step alone. Model C (noise-aware) recovered to 99.06% on Vosk output (111.3% gap recovery). WER was 11.43% overall, ranging from 6.78% (`emergency`) to 16.83% (`temperature_control`). P95 latency was 2,827ms — within the 5-second target. All voice processing is on-device, no audio leaves the hotel network, and the full stack is open-source with no recurring licensing costs.

### Overall Assessment

The research addressed all four gaps identified in the literature: end-to-end offline voice processing with on-device NLU in a real STT pipeline; privacy-preserving deployment without cloud dependency; cost-effective commodity hardware; and a solution designed for the operational constraints of hotels in developing economies.

---

## 11.3 Future Work

The prototype successfully demonstrates the core concept. The following items describe the most valuable next steps.

### 11.3.1 Field Deployment and User Study

The most important next step is deploying in a real hotel. A field study would validate speech recognition with real Sri Lankan English accents in room acoustic conditions, identify intent categories that real guests actually use, and surface usability issues that controlled testing cannot reveal. It would also allow the 99.06% accuracy figure — currently measured on TTS-synthesised speech — to be re-evaluated under genuine production conditions.

### 11.3.2 Real Speech Data Collection and Model Retraining

Training data and WER measurements were based on TTS audio, not real recordings. The tokenizer mismatch on Android (see Section 10.4.3) means actual on-device accuracy is lower than 99.06%. Collecting real voice recordings from Sri Lankan English speakers, using their Vosk transcriptions as training data, and re-evaluating all three models would give a more accurate picture of real-world performance.

### 11.3.3 Proper WordPiece Tokenizer on Android

The Android implementation uses a simplified word-level tokenizer rather than the HuggingFace WordPiece tokenizer used during training. Implementing a proper WordPiece tokenizer in Kotlin — or switching to TFLite’s built-in tokenization support — would align on-device inference with the evaluation conditions and recover some of the accuracy gap.

### 11.3.4 Multilingual Support

The system supports English only. In Sri Lanka, guests and staff may communicate in Sinhala, Tamil, or a code-switched mix. Future work could add Sinhala and Tamil STT using Vosk language model variants, a multilingual intent classifier, and automatic language detection. This would meaningfully expand applicability across Sri Lanka’s hospitality sector.

### 11.3.5 Further NLU and Dialogue Improvements

Beyond the core limitations above, several improvements would strengthen the NLU layer: a custom Vosk language model trained on hospitality vocabulary (targeting high-WER intents such as `temperature_control` at 16.83% and `towel_request` at 16.33%), an explicit `out_of_scope` intent to prevent spurious staff notifications when guests ask non-service questions, and a lightweight dialogue state tracker to handle multi-turn requests such as "I’d like two towels and a bottle of water."

---

## 11.4 Summary

The answer to the core research question is yes. The prototype achieves sufficient technical accuracy and performance across all measured dimensions: 99.06% NLU accuracy on real Vosk output (Model C), 11.43% WER acceptable for the hospitality domain, P95 latency of 2,827ms within the 5-second target, and hardware cost in the $50–$150 range. The three-model experiment provides the key supporting evidence — Model A's 8.73 percentage point drop from clean to Vosk conditions, and Model C's 111.3% recovery, make a concrete case for noise-aware training as a requirement for reliable production performance.

The main caveat is the synthetic evaluation foundation: WER measurements and training noise profiles came from TTS audio, not real recordings. Confirming these results with actual hotel guest speech, and deploying in a real hotel, are the most critical next steps. Chapter 12 reflects on the personal learning experience behind this work.
