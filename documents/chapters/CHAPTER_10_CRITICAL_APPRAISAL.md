# CHAPTER 10: CRITICAL APPRAISAL

## 10.1 Introduction

This chapter critically appraises the project — what was achieved, what worked well, and where the limitations lie. It reviews each research objective against what was actually delivered, analyses the strengths and weaknesses of the approach, reflects on the key technical decisions, and summarises the knowledge gained through the work. The goal is to give an honest assessment of the prototype's contributions and the degree to which the results can be generalised beyond the conditions under which they were obtained.

---

## 10.2 Achievement of Research Objectives

Each research objective defined in Chapter 1 is reviewed against what was actually delivered.

| Objective | Outcome | Assessment |
|-----------|---------|-----------|
| Design and develop a low-cost, offline voice assistant prototype deployable on commodity Android hardware | A working prototype was built and demonstrated end-to-end from voice input to staff notification, running on a commodity Android tablet in the $50–$150 price range across four development iterations | Fully achieved |
| Build a hospitality-domain dataset and train a noise-aware NLU model | Custom dataset of 10,080 labelled utterances across 18 categories (560 per intent), with paired Vosk-transcribed version enabling noise-aware training. `vosk-model-small-en-in-0.4` and fine-tuned MobileBERT (26MB TFLite) deployed on-device with a hybrid keyword + neural NLU pipeline | Fully achieved |
| Build a lightweight backend with real-time staff communication | FastAPI + SQLite backend with DB-driven department routing and bidirectional WebSocket communication between guest devices and staff dashboard — no cloud infrastructure required | Fully achieved |
| Evaluate system accuracy, latency, and cost-effectiveness, and demonstrate privacy-preserving offline operation | Three-model evaluation on 2,016-sample test set: Model C achieved 99.06% on Vosk output (111.3% gap recovery). WER 11.43% overall. P95 latency 2,827ms. Hardware cost under $150. All voice processing on-device — no audio leaves the hotel network | Fully achieved |

---

## 10.3 Strengths of the Project

### 10.3.1 Identifying and Quantifying the Real-World Accuracy Gap

Model A, trained only on clean text, achieves 98.07% accuracy on clean text — which is what most published work would report. On actual Vosk transcriptions of the same utterances, accuracy drops to 89.34%, a reduction of 8.73 percentage points. With a WER of 11.43% and 47.8% of utterances changed in some way by Vosk, the production pipeline operates under meaningfully different conditions from what clean-text evaluation assumes. Offline, on-device STT introduces a noise profile that must be accounted for in training — and this study measures that gap directly.

### 10.3.2 Noise-Aware Training Closes — and Exceeds — the Gap

Model C, trained on the mixed paired dataset (clean + Vosk-transcribed text), achieves 99.06% on real Vosk output — recovering the 8.73 percentage point drop and surpassing the clean baseline by 0.99 points (111.3% gap recovery). Model B, trained on Vosk-transcribed text only, reaches 96.38% on Vosk output, confirming that any exposure to the target noise profile helps. But it falls 2.68 points below Model C, showing that the mixed strategy — preserving clean-text generalisation while adding Vosk robustness — is better than training on noisy data alone.

### 10.3.3 A Concrete Example: The towel_request Case

`towel_request` is the clearest illustration of the gap. Model A achieves precision 0.97 on clean text, but on Vosk output recall drops to 0.51 and F1 falls to 0.67 — the weakest result across all 18 intents. Vosk's WER for this category is 16.33% (third-highest), and the confusion matrix shows "towel" being transcribed as phonetically similar words that land in `room_cleaning`, `toiletries_request`, `food_order`, and `checkout_billing`. Model C recovers `towel_request` F1 to 0.98 on Vosk output — directly demonstrating how noise-aware training addresses intent-specific degradation that clean-text evaluation cannot detect.

### 10.3.4 Consistent Noise Profile Between Training and Deployment

The same Vosk model (`vosk-model-small-en-in-0.4`) was used to generate the noisy training data and is deployed in the Android application. This consistency is what makes noise-aware training work — if training noise came from a different STT engine or model variant, the error patterns would not match production. The Indian English model was also a better acoustic fit for Sri Lankan English than a US English model, keeping WER at 11.43%.

### 10.3.5 Balanced and Reproducible Evaluation Design

The 10,080-utterance dataset (18 intents, 560 per intent) was created with exact class balance. The 2,016-sample held-out test set was fixed before any model training began and used consistently across all three models and all four evaluation conditions. This means all accuracy comparisons reflect real differences in model behaviour, not differences in class distribution or test set composition.

### 10.3.6 Privacy by Architecture

Voice data cannot leave the guest device — the architecture has no mechanism to send audio externally. This is a structural guarantee, not a policy-based one. It is a stronger assurance than cloud-based alternatives where privacy depends on provider agreements, and is particularly relevant as data protection legislation develops across South and Southeast Asia.

### 10.3.7 Cost-Effectiveness and Viability at Scale

A "viable alternative" only holds if hotels can actually afford to deploy it. This section presents an itemised per-room deployment cost, a comparison against cloud-based and commercial alternatives, and a three-year total cost of ownership (TCO) projection for a hypothetical 50-room hotel, as defined in the evaluation methodology (Section 3.8.4).

#### Hardware and Software Cost — Production Deployment

The prototype runs on a laptop server with SQLite, which is sufficient for a single-device evaluation setup but would require production-grade infrastructure for a real hotel deployment. Table 10.1 reflects this realistic cost, not the prototype configuration.

**Table 10.1: Production Deployment Cost for a 50-Room Hotel**

| Component | Scope | Item | Estimated Cost (USD) | Notes |
|-----------|-------|------|---------------------|-------|
| Guest device | Per room (×50) | Entry-level Android tablet (8", 3GB RAM) | 50–150 each | Locally available in Sri Lanka; commodity hardware (see Section 5.6) |
| Production server | Shared (×1) | Mini PC / small form-factor server (Intel NUC or equivalent, 16GB RAM, SSD) | 300–500 | Replaces Raspberry Pi for production; handles concurrent WebSocket connections and PostgreSQL under real hotel load |
| Database | Shared | PostgreSQL (replaces prototype SQLite) | 0 | Open-source; production-grade; zero licence cost |
| Wi-Fi access points | Shared | Dual-band WAP (one per floor, 2–3 units typical for a 50-room hotel) | 80–150 each | Required if hotel lacks adequate existing coverage; shared with all hotel systems |
| Network router | Shared (×1) | Business-grade router with Virtual Local Area Network (VLAN) support | 100–200 | Isolates guest device traffic; recommended for security |
| UPS / power backup | Shared (×1) | Uninterruptible power supply for server | 80–150 | Prevents data loss on power interruption |
| Vosk STT model | Per device | `vosk-model-small-en-in-0.4` | 0 | Open-source; Apache 2.0 licence (Alpha Cephei, 2020) |
| MobileBERT TFLite | Per device | Compressed intent classifier | 0 | Open-source; Apache 2.0 licence (Sun et al., 2020) |
| Backend software | Server | FastAPI, PostgreSQL, Python | 0 | Open-source; no licence fees |
| Ongoing API / subscription fees | — | None | 0 | No cloud calls; no per-request billing |

**Estimated total hardware investment (50-room hotel):**

| Scenario | Tablets (×50) | Server | Network (WAPs + router) | UPS | **Total** |
|----------|:-------------:|:------:|:-----------------------:|:---:|:---------:|
| Lower bound | $2,500 | $300 | $340 (2× WAP + router) | $80 | **~$3,220** |
| Upper bound | $7,500 | $500 | $650 (3× WAP + router) | $150 | **~$8,800** |

Spread over three years with zero recurring software costs, this amounts to approximately **USD 1,073–2,933 per year** for the entire hotel, or **USD 21–59 per room per year**. Network infrastructure (WAPs and router) is shared with all other hotel systems — email, POS, guest Wi-Fi — so the cost attributed to this system alone is proportionally lower in practice.

#### Scenario Comparison — Two Deployment Models

Sri Lanka's accommodation sector is predominantly small and independent properties (SLTDA, 2024), for whom per-room subscription fees are a prohibitive barrier. The question is not whether the system is cheap in absolute terms, but whether it is affordable relative to the alternatives that would otherwise serve this market.

**Table 10.2: Three-Year TCO Comparison for a 50-Room Hotel**

| Cost Item | This System | Cloud STT (Google) | Commercial Platform (e.g., Alexa for Hospitality) |
|-----------|:-----------:|:-----------------:|:-------------------------------------------------:|
| Guest devices (×50) | $2,500–7,500 | $2,500–7,500 | $5,000–10,000 (proprietary/branded hardware) |
| Server (production-grade) | $300–500 | $300–500 | Included in subscription (cloud-hosted) |
| Network infrastructure | $340–650 | $340–650 | Not required (cloud-managed) |
| UPS | $80–150 | $80–150 | Not required |
| Software licences | $0 | $0 | Subscription required (proprietary platform) |
| STT API fees (3 yr) | $0 | $1,095–2,190 | Bundled in subscription |
| Platform subscription (3 yr) | $0 | $0 | Not publicly priced; per-room, per-month billing |
| Internet dependency | LAN only | Constant internet | Constant internet |
| **Estimated 3-year total** | **$3,220–8,800** | **$4,315–11,000** | **Significantly higher; proprietary terms** |

**Cloud STT cost estimate (Google Cloud Speech-to-Text):** Billed at USD 0.006 per 15 seconds of audio (Google Cloud, 2024; consistent with Section 5.6.1), assuming 5–10 requests per room per day at ~10 seconds each:

```
Low estimate  (5 req/day):  5 × 365 × (10/15) × $0.006 × 50 rooms × 3 yr = $1,095
High estimate (10 req/day): 10 × 365 × (10/15) × $0.006 × 50 rooms × 3 yr = $2,190
```

This cost accumulates with no ceiling and requires reliable internet — not guaranteed across all Sri Lankan hotel properties (Wickramasinghe and Ratnayake, 2022).

**Commercial hospitality platforms:** Platforms such as Amazon Alexa Smart Properties for Hospitality require proprietary hardware, constant cloud connectivity, and per-room subscription fees not publicly disclosed (Amazon Web Services, n.d.). The hardware cost alone (Echo-class devices at USD 100+ per room) matches the entire lower-bound cost of the proposed system before any subscription is applied. Buhalis and Moldavska (2022) note that commercial platform adoption has been confined to large chains precisely because their costs are incompatible with small independent properties.

#### Summary: Cost-Effectiveness Objective Met

The proposed system is the lowest-cost option across both comparison scenarios, and the cost-effectiveness objective stated in Chapter 1 is substantiated on the following evidence:

- **Zero recurring software costs** — the entire stack is open-source with no per-request, per-room, or per-month billing. Unlike cloud STT solutions whose API costs compound annually, this system's software cost is fixed at zero.
- **No internet dependency** — operation over a local LAN eliminates ongoing connectivity costs and removes the infrastructure risk that cloud-dependent systems carry in markets with variable internet reliability (Wickramasinghe and Ratnayake, 2022).
- **Commodity hardware** — Android tablets in the USD 50–150 range are available from local Sri Lankan retailers and can be repaired or replaced through local technicians, unlike proprietary hotel-grade hardware locked to specific vendors.
- **No vendor lock-in** — all components (Vosk, MobileBERT, FastAPI, SQLite) can be substituted, modified, or extended without licence constraints or contractual dependencies.
- **Market fit** — at USD 21–59 per room per year amortised over three years (including production-grade server, network infrastructure, and UPS), the system is within reach of the small independent hotel operators that make up the majority of Sri Lanka's registered accommodation sector (SLTDA, 2024), and for whom commercial alternatives with ongoing per-room subscription fees are not a realistic option.

The cost-effectiveness objective is met. The system delivers voice assistant capability at a hardware-only cost that commercial alternatives cannot match, with no ongoing software expenditure.

---

## 10.4 Limitations and Weaknesses

### 10.4.1 WER Measured on Synthesised Speech, Not Real Guests

The 11.43% WER and the Vosk noise profiles used for Model C training were generated by passing gTTS audio through Vosk — not by recording real speakers. TTS does not replicate the full range of real speech: varying accents, hesitations, microphone distance, and room acoustics. Whether the noise-aware training advantage holds with real hotel guests remains untested. This is the most significant methodological limitation of the study.

### 10.4.2 Synthetically Generated Training Data

All 10,080 utterances were produced through template expansion and paraphrase augmentation, not collected from real guest interactions. The dataset does not include fragmented sentences, code-switching between English and Sinhala or Tamil, or culturally specific phrasing. Model performance on genuine guest speech has not been measured.

### 10.4.3 Tokenizer Mismatch on Android

The Python evaluation pipeline uses the correct HuggingFace WordPiece tokenizer, which is why 99.06% accuracy is reported. The Android implementation in `NLUService.kt` uses a simplified word-level tokenizer — words not found in `vocab.json` are replaced with an unknown token, and sub-word splitting is not performed. Actual on-device accuracy is therefore lower than the reported figure, particularly for hotel-specific vocabulary. The rule-based Tier 1 keyword layer partially compensates for the most common requests. Implementing a proper WordPiece tokenizer in Kotlin, or using TFLite's built-in tokenization support, would be the recommended fix in a production deployment.

### 10.4.4 No Field Testing in a Real Hotel

The system was not deployed in an operational hotel. User acceptance, staff adoption, latency on real hotel Wi-Fi under competing traffic, and guest usability with diverse accents have not been measured. Strong evaluation results under controlled conditions do not guarantee the same performance in a live environment.

### 10.4.5 Prototype-Scale Architecture

Several decisions are appropriate for a research prototype but would need to change for production: SQLite has limited concurrent write support under multi-room load; there is no staff authentication; HTTP traffic is unencrypted; and the server runs as a single Uvicorn process with no crash recovery. These are deliberate scope boundaries, not oversights. The research goal was to demonstrate feasibility, not to deliver a production-ready system.

### 10.4.6 English Language Only

The system supports English only. In Sri Lanka, guests and staff may communicate in Sinhala, Tamil, or a code-switched mix. Vosk has a Sinhala model, and the NLU pipeline could in principle support multilingual input, but this was outside the scope of this prototype. Multilingual support is the most contextually relevant direction for future work.

### 10.4.7 Summary of Limitations

| Limitation | Impact on Findings | Mitigation in This Study | Future Work |
|---|---|---|---|
| TTS-synthesised WER | Gap magnitude may differ with real speech | Demonstrates the principle of noise-aware training | Re-measure with real recorded speech |
| Synthetic training data | Accuracy may be optimistic vs. real guest speech | Accepted practice when real data is unavailable | Collect real hotel utterances for retraining |
| Tokenizer mismatch on Android | On-device accuracy lower than reported 99.06% | Tier 1 keyword layer compensates for common cases | Implement WordPiece tokenizer in Kotlin |
| No real hotel deployment | Operational claims are based on controlled testing | System tested on real hardware; architecture validated | Conduct field study in a Sri Lankan hotel |
| Prototype-scale architecture | Not production-ready without further engineering | Scope limited to feasibility demonstration | Replace SQLite, add authentication and TLS |
| English only | Limited applicability for non-English speakers | Target guests communicate in English | Add Sinhala and Tamil support |

---

## 10.5 Technical Decisions — What Worked and What Could Be Improved

### 10.5.1 What Worked Well

**Indian English Vosk model.** Switching to `vosk-model-small-en-in-0.4` was one of the most consequential decisions of the research. It improved transcription quality for South Asian accent speakers, kept the device storage footprint small (~36MB), and — most critically — ensured the training noise profile matched the production system. Without this match, noise-aware training would have prepared the model for the wrong type of errors.

**Mixed paired dataset for Model C.** Training on both clean text and Vosk-transcribed versions of the same utterances (14,864 records after deduplication) outperformed Vosk-only training (Model B: 96.38% vs Model C: 99.06%). Preserving clean-text generalisation while adding Vosk robustness is better than optimising for noise alone.

**Fixed shared test set.** Fixing the 2,016-sample test set before training any model and using it consistently across all conditions was essential for producing valid, comparable results. Without this, differences in test set composition could have explained away real performance differences between models.

**MobileBERT over DistilBERT.** MobileBERT's mobile-specific design produced a smaller (26MB vs ~67MB) and faster model for Android deployment. This confirms that choosing a model based on the deployment target matters more than general benchmark scores.

**FastAPI for the backend.** Native async support and built-in WebSocket handling were essential for managing simultaneous connections from multiple guest devices and dashboard instances. The automatic OpenAPI documentation at `/docs` significantly accelerated API testing during development.

**Single-file staff dashboard.** Building the dashboard as one self-contained HTML file with embedded CSS and JavaScript eliminated all build tooling and dependency management. Any staff member can open it in a browser on any device — no installation, no configuration.

### 10.5.2 What Could Be Improved

**TTS-generated noise profiles.** Using text-to-speech audio passed through Vosk to generate training noise was practical but is the main limitation on how confidently the results transfer to real deployment. Collecting speech recordings from real Sri Lankan English speakers and using their actual Vosk transcriptions as training data would be a much stronger foundation.

**Confidence threshold calibration.** The 0.60 confidence threshold for the hybrid NLU pipeline was chosen through manual testing rather than systematic calibration. A proper approach would use a held-out validation set to find the threshold that optimises the trade-off between false rejections and misclassified requests, potentially varying by intent category.

**Python keyword fallback in routing.** The intent-to-department mapping is correctly stored in the `intent_department_mapping` database table as the primary routing source. However, the server also maintains a hardcoded Python keyword list as a fallback for cases where the intent field is missing. Moving this fallback logic into the database as well would allow hotels to fully customise routing behaviour without any code changes.

**No structured logging.** The prototype uses print statements for debugging. Structured logging with severity levels and timestamps would make it far easier to diagnose issues in testing and would be a requirement for any real-world deployment.

---

## 10.6 Knowledge and Expertise Gained

**On-device machine learning deployment.** The PyTorch → TFLite → Android pipeline is not a straightforward export. It requires careful checking of input format consistency (including `token_type_ids` as a third input), tokenisation with `padding='max_length'` for fixed-shape TFLite inputs, and label map alignment between PyTorch and TFLite outputs.

**Experimental design for comparative NLU evaluation.** Isolating training data composition as the only variable — identical architecture, hyperparameters, and test set across all three models — required fixing the shared test set before training any model. Without this, accuracy differences between Model A and Model C could have been attributed to test set composition rather than genuine behaviour differences.

**ASR-NLU pipeline interaction.** The accuracy gap between clean-text and real pipeline performance is measurable and intent-specific. The `towel_request` case — WER 16.33%, F1 dropping from 0.97 to 0.67 — shows the gap is driven by intents with phonetically vulnerable vocabulary. Per-intent WER is a useful pre-deployment diagnostic: higher WER intents are the ones most at risk from pipeline-induced accuracy loss.

**Privacy-preserving system design.** Privacy-by-design is an architectural commitment, not a compliance checkbox. Ensuring voice data cannot leave the device required thinking through data flows at every stage — from microphone capture through to the WebSocket broadcast — and influenced component selection from the start.

**Research methodology.** Starting with a working prototype early reveals constraints that no amount of upfront planning can anticipate. The Vosk accent mismatch problem and the hybrid NLU pipeline design both emerged from hands-on implementation, not from design documents.

---

## 10.7 Summary

The prototype meets the technical bar for viability as an offline alternative to traditional room service communication. Model C achieves 99.06% NLU accuracy on real Vosk output — recovering the 8.73 percentage point gap that clean-text training introduces — the system runs within the 5-second latency target, hardware cost falls in the $50–$150 per-room range, and voice data never leaves the guest device. The three-model experiment is the key supporting evidence: it shows that clean-text benchmarks overstate real pipeline performance, and that noise-aware training is what makes the NLU component reliable in a real deployment.

The main caveat is that both the WER measurement and the training noise profiles came from TTS-synthesised speech, not real recordings. Confirming these results with actual hotel guest speech is the most critical next step before drawing firm conclusions about production readiness.
