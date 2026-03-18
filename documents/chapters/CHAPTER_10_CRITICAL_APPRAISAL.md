# CHAPTER 10: CRITICAL APPRAISAL

## 10.1 Achievement of Research Objectives

Each research objective defined in Chapter 1 is reviewed against what was actually delivered.

| Objective | Outcome | Assessment |
|-----------|---------|-----------|
| Design and develop a low-cost, offline-capable voice assistant prototype for hospitality services using small-scale neural models | A working prototype was built and demonstrated end-to-end from voice input to staff notification | Fully achieved |
| Implement on-device speech recognition and intent classification using small-scale neural models (Vosk, MobileBERT) | Both models run entirely on-device on a commodity Android tablet without internet connectivity. `vosk-model-small-en-in-0.4` (Indian English) was selected specifically for South Asian accent compatibility | Fully achieved |
| Build a lightweight backend system using FastAPI and SQLite with real-time staff communication via WebSocket | Backend handles request persistence, department routing (DB-driven with keyword fallback), and bidirectional WebSocket communication | Fully achieved |
| Develop a hospitality-domain intent classification dataset covering 18 service request categories | Custom dataset of 10,080 labelled utterances created across 18 categories (560 per intent), with paired clean and Vosk-transcribed versions enabling noise-aware training | Fully achieved |
| Evaluate speech recognition accuracy, intent classification performance, and the accuracy impact of the real STT pipeline on NLU | Three-model evaluation completed on a 2,016-sample shared held-out test set. Model A (clean-trained) dropped from 98.07% on clean text to 89.34% on Vosk output (8.73 pp gap). Model C (noise-aware) recovered to 99.06% on Vosk output — exceeding the clean baseline | Fully achieved |
| Demonstrate that privacy-preserving, offline voice-based service automation is achievable on low-cost hardware | All voice processing occurs on-device; no audio or transcript data leaves the hotel network | Fully achieved |

---

## 10.2 Strengths of the Project

### 10.2.1 Identifying and Quantifying the Real-World Accuracy Gap

The most important contribution of this research is measuring something that standard NLU benchmarking does not reveal. A MobileBERT model trained only on clean text (Model A) achieves 98.07% accuracy when tested on clean text — which is what most published work would report as the model's performance. When the same model is evaluated on actual Vosk transcriptions of the same utterances, accuracy drops to 89.34%, a reduction of 8.73 percentage points. This degradation is not random noise. It is a predictable consequence of Vosk's characteristic transcription errors — phonetically similar substitutions, insertions, and deletions — changing the vocabulary the clean-trained model learned to associate with specific intents.

With an overall WER of 11.43% across the dataset, and 47.8% of utterances changed in some way by Vosk, the production voice pipeline operates in meaningfully different conditions from what clean-text evaluation assumes. This finding addresses a genuine gap: offline, on-device deployments using open-source edge STT introduce a noise profile that differs from both clean text and cloud STT error patterns, and this profile needs to be accounted for in training to achieve reliable real-world performance.

### 10.2.2 Noise-Aware Training Closes — and Exceeds — the Gap

Model C, trained on the mixed paired dataset (clean + Vosk-transcribed text), achieves 99.06% accuracy on real Vosk output — not only recovering the 8.73 percentage point drop but surpassing the clean baseline by 0.99 percentage points. The gap recovery of 111.3% shows that the noise-aware model generalises better across both conditions than the clean-only model does on its best condition.

Model B, trained only on Vosk-transcribed text, also outperforms the clean-trained baseline on Vosk output (96.38% vs 89.34%), which confirms that any exposure to the target noise profile helps. However, Model B is 2.68 percentage points below Model C, which shows that the mixed training strategy — preserving clean-text generalisation while adding Vosk robustness — is better than training on noisy data alone.

These results are the main empirical finding: Vosk-specific noise-aware training using a paired clean and noisy dataset is an effective and practical way to close the accuracy gap between clean-text NLU benchmarks and real offline pipeline performance.

### 10.2.3 A Concrete Example: The towel_request Case

The most visible illustration of Vosk-induced accuracy degradation is the `towel_request` category. In clean-text evaluation, Model A performs well (precision 0.97). On Vosk output, recall drops to 0.51 and F1 falls to 0.67 — the weakest result across all 18 intents in all evaluation conditions. Vosk's WER for towel_request utterances is 16.33% (third-highest across all intents), and the confusion matrix shows the word "towel" being consistently transcribed as phonetically similar alternatives that fall into the vocabulary of other intents: `room_cleaning`, `toiletries_request`, `food_order`, and `checkout_billing`.

Model C recovers `towel_request` F1 to 0.98 on Vosk output. This one intent illustrates the broader pattern clearly: Vosk's phonetic substitutions create intent-specific accuracy drops invisible under clean-text evaluation, and noise-aware training directly addresses them.

### 10.2.4 Consistent Noise Profile Between Training and Deployment

One of the more important design decisions in this research was using the same Vosk model (`vosk-model-small-en-in-0.4`) to generate the noisy training data that is deployed in the Android application. This consistency is what makes noise-aware training work. If the training noise was generated by a different STT engine or a different Vosk variant, the error patterns in training would not match the errors the model encounters in production. The Indian English model was also a better acoustic fit for Sri Lankan English than a US English model, which kept the WER to 11.43% rather than significantly higher.

### 10.2.5 Balanced and Reproducible Evaluation Design

The 10,080-utterance dataset (18 intents, 560 per intent) was created with exact class balance. The 2,016-sample held-out test set was fixed before any model training began and used consistently across all three models and all four evaluation conditions. This means all accuracy comparisons reflect real differences in model behaviour, not differences in class distribution or test set composition.

### 10.2.6 Privacy by Architecture

Unlike cloud-based solutions where privacy depends on provider policies and legal agreements, this prototype provides a structural privacy guarantee. Voice data cannot leave the guest device because the architecture has no mechanism to send audio externally. This is a stronger assurance than any policy-based approach and is particularly relevant as data protection legislation continues to develop across South and Southeast Asia.

### 10.2.7 Genuinely Low-Cost Deployment

The prototype runs on an Android tablet (USD 50–150), any existing PC as a server, and a fully open-source software stack with no recurring subscription fees. This is a realistic deployment model for small hotels in Sri Lanka. Commercial alternatives like Alexa for Hospitality require proprietary hardware and ongoing AWS subscription costs that are incompatible with the target market.

---

## 10.3 Limitations and Weaknesses

### 10.3.1 WER Derived from TTS-Synthesised Speech, Not Real Speakers

The WER of 11.43% and the Vosk noise profiles used for training were produced by passing text-to-speech audio (gTTS, `tld='co.in'`) through the Vosk model, not by recording real hotel guests. TTS-to-Vosk transcription produces realistic phonetic errors, but it does not capture the full range of acoustic variation from real speakers: non-native pronunciation patterns, hesitations, varying microphone distances, room acoustics, and accent strength differences across individuals.

This is the most significant methodological limitation of the study. The noise-aware training approach is sound and the results are strong, but they were obtained under controlled synthetic conditions. The actual WER in a real hotel deployment — with guests of diverse nationalities — would likely be different, and whether Model C's advantage transfers to those conditions remains unknown.

### 10.3.2 Synthetically Generated Training Data

The 10,080-utterance dataset was generated synthetically through template expansion and paraphrase augmentation, not collected from real hotel guest interactions. Synthetic data may not capture the full range of real speech, including:

- Incomplete or fragmented sentences common in natural spoken requests
- Code-switching between English and Sinhala or Tamil
- Indirect or culturally specific ways of making requests
- Natural hesitations, fillers, and self-corrections

The model's performance on genuinely real guest utterances — both in accuracy and in handling unexpected phrasings — has not been tested.

### 10.3.3 Residual Weakness in misc_request

Even with noise-aware training, `misc_request` is the weakest intent for Model C, with F1 of 0.96 on Vosk output compared to 1.00 for several other intents. The confusion matrix shows some `misc_request` instances being misclassified as `blanket_request` and `pillow_request`. This is an inherent challenge with having a general catch-all category that by design overlaps semantically with other classes. Neither training strategy fully resolves this ambiguity, and it would persist with more data.

### 10.3.4 No Out-of-Scope Query Handling

The system assumes all voice input from guests is a hotel service request. It has no mechanism to distinguish valid service requests ("Can I have a bottle of water?") from unrelated utterances ("What is the weather like tomorrow?"). The 18-class classifier always assigns one of the 18 labels regardless of input type. The 0.60 confidence threshold provides some protection, but no specific mechanism was designed or evaluated for detecting out-of-scope queries. In real deployment, unrecognised inputs that pass the threshold would generate unnecessary staff notifications, which could erode staff trust in the system over time.

### 10.3.5 Limited Intent Scope

The prototype handles 18 single-turn, single-intent voice commands. Real hotel operations involve interactions the current system cannot handle:

- Multi-item requests ("I'd like two towels and a pillow")
- Conditional requests ("If the restaurant is still open, I'd like to order dinner")
- Follow-up references ("Actually, make that three instead of two")
- Conversational queries ("What time does the pool close?")

Multi-turn dialogue, entity extraction beyond simple quantities, and complex request parsing are outside the current prototype's scope.

### 10.3.6 Prototype-Scale Architecture

Several design decisions are reasonable for a single-hotel research prototype but would not scale without changes:

| Prototype Decision | Production Limitation |
|-------------------|-----------------------|
| SQLite database | Limited concurrent write support under high multi-room load |
| No staff authentication | Any device on the hotel Wi-Fi can access the dashboard and API |
| Manual room number configuration | Each device must be set up individually; no central provisioning |
| Single Uvicorn process | No crash recovery, process management, or load balancing |
| HTTP without TLS | Local network traffic is unencrypted |

These are deliberate scope decisions, not oversights. The research goal was to demonstrate feasibility, not to deliver a production-ready product.

### 10.3.7 No Field Testing in a Real Hotel

The system was not deployed in an actual hotel for operational testing. User acceptance, staff adoption, end-to-end latency on real hotel Wi-Fi with competing traffic, and guest usability with real accents have not been measured. High classification accuracy under controlled evaluation does not guarantee the same experience in a live hotel environment.

### 10.3.8 Tokenizer Mismatch Between Evaluation and Android Deployment

MobileBERT uses a WordPiece tokenizer, where words are split into sub-word units — for example, "housekeeping" becomes ["house", "##keeping"]. The Python evaluation pipeline (Steps 3 and 4) uses the correct HuggingFace MobileBERT tokenizer, which is why the reported 99.06% accuracy is achieved.

The Android implementation in `NLUService.kt`, however, uses a simplified word-level tokenizer that splits text by whitespace and looks up each whole word in a `vocab.json` file. Any word not found in the vocabulary is replaced with the unknown token, and sub-word tokenization is not performed. As a result, the actual intent classification accuracy on the Android device will be lower than 99.06% — particularly for less common words and hotel-specific terminology that Vosk transcribes into forms requiring sub-word decomposition.

This was a practical trade-off made to avoid implementing a full WordPiece tokenizer in Kotlin. The rule-based keyword matching layer (Tier 1) partially compensates by handling the most common hotel service requests before the MobileBERT model is invoked. Implementing a proper WordPiece tokenizer on Android, or using TFLite's built-in tokenization support, would be the recommended fix in a production deployment.

### 10.3.9 Single STT Engine Evaluated

This research uses Vosk as the only offline STT engine. The core finding — that there is an accuracy gap between clean-text NLU and Vosk-pipeline NLU, and that noise-aware training closes this gap — is demonstrated specifically for Vosk's transcription error patterns.

Other offline STT engines such as Whisper (tiny variant) or CMU Sphinx would produce different transcription errors with different characteristics. The size of the accuracy gap and the degree to which noise-aware training recovers it would likely differ for other STT engines. The general principle — that NLU models should be trained on STT-transcribed data to perform well in a real pipeline — is expected to hold across engines, but this has not been empirically validated in this study. Future work could replicate the three-model pipeline with Whisper tiny as the STT engine to test whether the findings generalise.

### 10.3.10 English Language Only

The system currently supports English only. In the Sri Lankan hospitality context, some staff and guests may communicate more naturally in Sinhala, Tamil, or a code-switched mix of languages. Vosk supports Sinhala as a separate model, and the NLU model could in principle be fine-tuned on Sinhala or multilingual data, but this was outside the scope of this research. Supporting multiple languages is identified as a direction for future work.

### 10.3.11 Summary of Limitations

| Limitation | Impact on Findings | Mitigation in This Study | Future Work |
|---|---|---|---|
| Synthetic dataset | Accuracy may be optimistic vs. real guest speech | Accepted practice when real data unavailable | Collect real hotel utterances for retraining |
| TTS-generated Vosk transcriptions | Gap magnitude may differ with real speech | Demonstrates principle of noise-aware training | Re-measure with real recorded speech |
| Tokenizer mismatch on Android | Android accuracy lower than reported 99.06% | Rule-based Tier 1 compensates for common cases | Implement WordPiece tokenizer in Kotlin |
| Single STT engine (Vosk only) | Findings may not generalise to other STT engines | Vosk is the most practical offline choice | Replicate with Whisper tiny |
| No real hotel deployment | Operational claims are theoretical | System tested on real device, architecture validated | Conduct field study in a Sri Lankan hotel |
| English only | Limited applicability for non-English speakers | Target user group communicates in English | Add Sinhala/Tamil support |

---

## 10.4 Technical Decisions — What Worked and What Could Be Improved

### 10.4.1 What Worked Well

**Indian English Vosk model.** Switching to `vosk-model-small-en-in-0.4` was one of the most consequential decisions of the research. It improved transcription quality for South Asian accent speakers, kept the device storage footprint small (~36MB), and — most critically — ensured the training noise profile matched the production system. Without this match, noise-aware training would have prepared the model for the wrong type of errors.

**Mixed paired dataset for Model C.** Training on both clean text and Vosk-transcribed versions of the same utterances (14,864 records after deduplication) outperformed Vosk-only training (Model B: 96.38% vs Model C: 99.06%). Preserving clean-text generalisation while adding Vosk robustness is better than optimising for noise alone.

**Fixed shared test set.** Fixing the 2,016-sample test set before training any model and using it consistently across all conditions was essential for producing valid, comparable results. Without this, differences in test set composition could have explained away real performance differences between models.

**MobileBERT over DistilBERT.** MobileBERT's mobile-specific design produced a smaller (26MB vs ~67MB) and faster model for Android deployment. This confirms that choosing a model based on the deployment target matters more than general benchmark scores.

**FastAPI for the backend.** Native async support and built-in WebSocket handling were essential for managing simultaneous connections from multiple guest devices and dashboard instances. The automatic OpenAPI documentation at `/docs` significantly accelerated API testing during development.

**Single-file staff dashboard.** Building the dashboard as one self-contained HTML file with embedded CSS and JavaScript eliminated all build tooling and dependency management. Any staff member can open it in a browser on any device — no installation, no configuration.

### 10.4.2 What Could Be Improved

**TTS-generated noise profiles.** Using text-to-speech audio passed through Vosk to generate training noise was practical but is the main limitation on how confidently the results transfer to real deployment. Collecting speech recordings from real Sri Lankan English speakers and using their actual Vosk transcriptions as training data would be a much stronger foundation.

**Confidence threshold calibration.** The 0.60 confidence threshold for the hybrid NLU pipeline was chosen through manual testing rather than systematic calibration. A proper approach would use a held-out validation set to find the threshold that optimises the trade-off between false rejections and misclassified requests, potentially varying by intent category.

**Python keyword fallback in routing.** The intent-to-department mapping is correctly stored in the `intent_department_mapping` database table as the primary routing source. However, the server also maintains a hardcoded Python keyword list as a fallback for cases where the intent field is missing. Moving this fallback logic into the database as well would allow hotels to fully customise routing behaviour without any code changes.

**No structured logging.** The prototype uses print statements for debugging. Structured logging with severity levels and timestamps would make it far easier to diagnose issues in testing and would be a requirement for any real-world deployment.

---

## 10.5 Knowledge and Expertise Gained

**On-device machine learning deployment.** Working through the full PyTorch → TFLite → Android pipeline gave a clear understanding of the practical challenges of edge ML deployment. The most useful lesson was that model conversion is not a straightforward export step — it requires careful checking of input/output format consistency (particularly the requirement to include `token_type_ids` as a third input), tokenisation compatibility with `padding='max_length'` for fixed-shape TFLite inputs, and label map alignment between the PyTorch training output and the TFLite inference output.

**Experimental design for comparative NLU evaluation.** Designing an experiment that isolates training data composition as the only variable — identical model architecture, identical hyperparameters, identical test set — required careful planning. The most important decision was fixing the shared test set before training any model. Without this, the gap between Model A and Model C could have been attributed to differences in what samples they were evaluated on rather than genuine differences in behaviour.

**ASR-NLU pipeline interaction.** This research showed concretely that the accuracy gap between clean-text NLU performance and real pipeline performance is measurable and intent-specific. The `towel_request` case — WER 16.33%, F1 dropping from 0.97 to 0.67 under clean-trained evaluation — shows that this gap is not evenly distributed and is driven by specific intents with phonetically vulnerable vocabulary. This is a practically useful insight for any offline voice system design.

**Per-intent WER as a diagnostic tool.** The variation in WER across intents (6.78% for `emergency` to 16.83% for `temperature_control`) shows that Vosk's difficulty varies systematically with the vocabulary in each category. Intents with higher WER are also the ones most likely to show accuracy degradation when the model is trained only on clean text. This relationship provides a useful pre-deployment diagnostic: compute per-intent WER first to identify which categories are most at risk from pipeline-induced accuracy loss.

**Privacy-preserving system design.** Designing a system where voice data privacy is guaranteed by architecture rather than by policy required thinking carefully about data flows at every level. This experience reinforced that privacy-by-design is not an afterthought or a compliance checkbox — it is an architectural commitment that influences component selection and data flow design from the start.

**Research methodology.** The Design Science Research approach combined with iterative prototyping worked well for a project involving substantial technical uncertainty. The most valuable lesson: starting with a working prototype early reveals constraints that no amount of upfront planning can anticipate. Both the Vosk accent mismatch problem and the hybrid NLU pipeline design emerged from hands-on implementation, not from design documents.

---

## 10.6 Summary

The prototype successfully demonstrates that a low-cost, offline voice assistant for hospitality services is achievable using small-scale neural models on commodity Android hardware. Beyond the working system, the primary research contribution is measuring the accuracy gap between clean-text NLU evaluation and real offline pipeline performance, and showing that Vosk-specific noise-aware training closes it. Model A's 8.73 percentage point drop from clean text (98.07%) to Vosk output (89.34%) confirms that standard clean-text benchmarks overstate the actual performance of NLU models in offline STT pipelines. Model C's recovery to 99.06% — a 111.3% gap recovery that exceeds the clean-text baseline — demonstrates that paired noise-aware training is an effective and practical solution.

The most important caveat is that both the WER measurement and the training noise profiles were derived from TTS-synthesised speech rather than real speaker recordings. Confirming these results with actual hotel guest speech is the most critical next step before drawing firm conclusions about production readiness. Deploying the system in a real hotel and evaluating it on real guest utterances is the most important direction for future research.
