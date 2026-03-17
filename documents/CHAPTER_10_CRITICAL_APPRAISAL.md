# Chapter 10: Critical Appraisal

This chapter provides an honest evaluation of the project. It looks at what was achieved, what worked well, what did not work as well as expected, and what was learned throughout the research process.

---

## 10.1 Achievement of Research Objectives

Each research objective defined in Chapter 1 is reviewed against what was actually delivered.

| Objective | Outcome | Assessment |
|---|---|---|
| Design and develop a low-cost, offline-capable voice assistant prototype for hospitality services using small-scale neural models | A working prototype was developed and demonstrated end-to-end functionality from voice input to staff notification | Fully achieved |
| Implement on-device speech recognition and intent classification using small-scale neural models (Vosk, MobileBERT) | Both models run entirely on-device on a commodity Android tablet without internet connectivity. The Indian English Vosk model (`vosk-model-small-en-in-0.4`) was selected to better match Sri Lankan accent characteristics | Fully achieved |
| Build a lightweight backend system using FastAPI and SQLite with real-time staff communication via WebSocket | Backend handles request persistence, department routing, and bidirectional WebSocket communication | Fully achieved |
| Develop a hospitality-domain intent classification dataset covering 18 service request categories | Custom dataset of 10,080 labelled utterances created across 18 categories (560 per intent), with paired clean and Vosk-transcribed versions enabling noise-aware training | Fully achieved |
| Evaluate speech recognition accuracy, intent classification performance, and the accuracy impact of the real STT pipeline on NLU | Three-model evaluation completed on a held-out test set of 2,016 paired utterances. Model A (clean-trained) achieved 98.07% on clean text but dropped to 89.34% on real Vosk output (8.73 percentage point gap). Model C (noise-aware trained) recovered to 99.06% on Vosk output — exceeding the clean baseline | Fully achieved |
| Demonstrate that privacy-preserving, offline voice-based service automation is achievable on low-cost hardware | All voice processing occurs on-device; no data leaves the hotel network | Fully achieved |

---

## 10.2 Strengths of the Project

### 10.2.1 Identifying and Quantifying the Real-World Accuracy Gap

The most important contribution of this research is measuring an accuracy gap that standard NLU benchmarking does not reveal. A MobileBERT model trained only on clean text (Model A) achieves 98.07% accuracy when tested on clean text — which is what most published work would report as the model's performance. However, when the same model is tested on actual Vosk transcriptions of the same utterances, accuracy drops to 89.34%, a reduction of 8.73 percentage points.

This is not a random effect. It happens because Vosk makes characteristic transcription errors — phonetically similar substitutions, insertions, and deletions — that change the vocabulary the clean-trained model learned to associate with specific intents. With an overall Word Error Rate (WER) of 11.43% across the dataset and 47.8% of utterances changed in some way by Vosk, the production voice pipeline operates in very different conditions from what clean-text evaluation assumes.

This finding addresses a real gap in the literature: offline, on-device deployments using open-source edge STT introduce a noise profile that is different from both clean text and cloud STT error patterns, and this profile must be taken into account during training to achieve reliable real-world performance.

### 10.2.2 Noise-Aware Training Closes — and Exceeds — the Gap

Model C, trained on the mixed paired dataset (clean + Vosk-transcribed text), achieves 99.06% accuracy on real Vosk output. This not only recovers the 8.73 percentage point drop but actually exceeds the clean baseline (98.07%) by 0.99 percentage points. The gap recovery of 111.3% shows that the noise-aware model generalises better across both clean and noisy inputs than the clean-only model does on its preferred condition.

Model B, trained only on Vosk-transcribed text, also performs much better than the clean-trained baseline on Vosk output (96.38% vs 89.34%), which confirms that any exposure to the target noise profile during training helps. However, Model B performs 2.68 percentage points lower than Model C, which shows that the mixed training strategy — preserving clean-text generalisation while adding Vosk robustness — is better than training on noisy data alone.

These results are the main empirical finding of the research: Vosk-specific noise-aware training using a paired clean and noisy dataset is an effective and practical way to close the accuracy gap between clean-text NLU benchmarks and real offline pipeline performance.

### 10.2.3 The towel_request Case: A Concrete Example of the Gap

The most visible example of Vosk-induced accuracy degradation is in the `towel_request` category. In the clean-text evaluation, Model A performs well for towel_request (precision 0.97). But when tested on Vosk output, recall drops to 0.51, giving an F1 of 0.67 — the weakest result across all 18 intents and all evaluation conditions. The Vosk WER for towel_request (16.33%) is the third-highest across all intents. Looking at the confusion matrix, Vosk consistently changes the word "towel" into phonetically similar but different words, causing misclassification into `room_cleaning`, `toiletries_request`, `food_order`, and `checkout_billing`.

Model C almost completely fixes this, bringing `towel_request` F1 back to 0.98 on Vosk output. This single intent shows the broader pattern clearly: Vosk's phonetic substitutions cause intent-specific accuracy drops that are not visible under clean-text evaluation, and noise-aware training directly addresses them.

### 10.2.4 Accent-Appropriate Speech Recognition

One deliberate improvement made during the research was selecting `vosk-model-small-en-in-0.4`, the Vosk Indian English model, instead of a general American English model. This model is more suitable for the South Asian English accent that is common among Sri Lankan hotel guests and staff, and it produced a dataset WER of 11.43%, which is reasonable for a lightweight on-device model in a constrained vocabulary domain. Importantly, the same Vosk model used in the deployed Android application was also used to generate the noisy training data for Models B and C. This consistency — making sure the training noise profile matches the production noise profile — is essential for noise-aware training to work properly in deployment.

### 10.2.5 Balanced and Reproducible Evaluation Design

The intent classification dataset of 10,080 utterances (18 intents, 560 per intent) was created with equal class balance. The held-out test set of 2,016 utterances (112 per intent) was fixed before any model training began and used consistently across all three models and all four evaluation conditions. This design means that all accuracy comparisons reflect real differences in model behaviour rather than differences in class distribution or test set overlap.

### 10.2.6 Privacy by Architecture

Unlike cloud-based solutions where privacy depends on the provider's policies and legal agreements, this prototype provides a structural privacy guarantee. Voice data cannot leave the guest device because the system architecture has no mechanism to send audio externally. This is a stronger privacy assurance than any policy-based approach and is particularly relevant as data protection regulations continue to develop across South and Southeast Asia.

### 10.2.7 Genuinely Low-Cost Deployment Model

The prototype runs on hardware and software that are either free or available locally at low cost. An Android tablet (USD 50–150), a basic server (any existing PC), and a fully open-source software stack with no recurring subscription fees make this a realistic option for small hotels in Sri Lanka. This is very different from commercial solutions such as Alexa for Hospitality, which require proprietary hardware and ongoing AWS subscription costs.

---

## 10.3 Limitations and Weaknesses

### 10.3.1 WER Derived from TTS-Synthesised Speech, Not Real Speakers

The WER of 11.43% and the Vosk noise profiles used for training were produced by passing text-to-speech synthesised audio through the Vosk model, not by recording real hotel guests. While TTS-to-Vosk transcription produces realistic phonetic substitution errors, it does not capture the full range of acoustic variation from real speakers: non-native pronunciation patterns, hesitations, varying microphone distances, room acoustics, and differences in accent strength between speakers.

This is the most significant methodological limitation of the study. The noise-aware training approach is sound and the results are strong, but they were obtained under controlled synthetic conditions. The actual WER and error distribution in a real hotel deployment — with diverse guests of different nationalities — would likely be different, and whether Model C's advantage transfers to those conditions is unknown. The `wer_report.txt` file explicitly notes: "Actual WER on real speech may differ."

### 10.3.2 Synthetically Generated Training Data

The intent classification dataset of 10,080 utterances was generated synthetically through template expansion and paraphrase augmentation, not collected from real hotel guest interactions. Synthetic data may not capture the full range of real guest speech, including:

- Incomplete or fragmented sentences that are common in natural spoken requests
- Code-switching between English and Sinhala or Tamil
- Indirect or culturally specific ways of making requests
- Natural hesitations, fillers, and self-corrections

The model's performance on real guest utterances — in terms of both accuracy and handling of unexpected phrasings — has not been tested.

### 10.3.3 Residual Weakness in misc_request

Even with noise-aware training, `misc_request` is the weakest intent for Model C, with an F1 of 0.96 on Vosk output compared to 1.00 for seven other intents. The confusion matrix shows that Model C misclassifies some `misc_request` instances as `blanket_request` and `pillow_request`. This happens because `misc_request` is a catch-all category that by design overlaps semantically with other categories, and neither training strategy fully resolves this ambiguity. This is an inherent challenge with having a general fallback intent class.

### 10.3.4 No Out-of-Scope Query Handling

The prototype assumes that all voice input from the guest is a hotel service request. It does not distinguish between valid service requests (e.g., "Can I have a bottle of water?") and general questions or unrelated utterances (e.g., "Can I do water rafting?" or "What is the weather like tomorrow?"). The 18-class classifier will always assign one of the 18 defined labels regardless of whether the input is actually a hotel service request. The 0.60 confidence threshold gives some protection, but no specific mechanism was designed or evaluated for detecting out-of-scope queries. In a real deployment, out-of-scope inputs would generate unnecessary staff notifications over time, which could reduce staff confidence in the system.

### 10.3.5 Limited Intent Scope

The prototype supports 18 intent categories across five departments. Real hotel operations involve many more types of guest interactions that the current system cannot handle:

- Multi-item requests ("I'd like two towels and a pillow")
- Conditional requests ("If the restaurant is still open, I'd like to order dinner")
- Follow-up references ("Actually, make that three instead of two")
- Conversational queries ("What time does the pool close?")

The system handles single-turn, single-intent voice commands only. Multi-turn dialogue, entity extraction, and complex request parsing are beyond what the current prototype can do.

### 10.3.6 Single-Hotel Prototype Scale

The prototype was designed for a single-hotel deployment with one room device. Several design decisions that are reasonable at this scale would not work in a larger deployment:

| Prototype Decision | Production Limitation |
|---|---|
| SQLite database | Cannot handle concurrent writes from multiple staff dashboards efficiently |
| No authentication | Any device on the Wi-Fi network can access the API and dashboard |
| Hardcoded room number | Each device must be manually configured |
| Single Uvicorn process | No process management, crash recovery, or load balancing |
| HTTP without TLS | Communication on the local network is unencrypted |

These are deliberate prototype scope decisions. The research goal was to show that offline voice-based service automation is feasible, not to build a production-ready product.

### 10.3.7 No Field Testing in an Actual Hotel

The system was not deployed in a real hotel for operational testing. User acceptance, staff adoption, and real-world performance — including end-to-end latency on hotel Wi-Fi networks with varying signal strength and competing traffic — have not been tested. The high intent classification accuracy under controlled evaluation conditions does not guarantee the same performance or usability in a live hotel environment.

---

## 10.4 Technical Decisions — What Worked and What Could Be Improved

### 10.4.1 Decisions That Worked Well

**Indian English Vosk model.** Switching to `vosk-model-small-en-in-0.4` was one of the most important decisions in the research. It improved transcription quality for South Asian accent speakers, kept the device footprint small, and — most importantly — made sure the training noise profile matched the production system. Without this match, noise-aware training would have prepared the model for the wrong type of noise.

**Mixed paired dataset for Model C.** Training on both clean text and Vosk-transcribed versions of the same utterances performed better than training on Vosk-only data (Model B: 96.38% vs Model C: 99.06%). The mixed approach keeps generalisation to clean text while adding robustness to Vosk's specific error patterns, producing a model that outperforms the clean baseline even on Vosk output.

**Fixed shared test set across all four evaluation conditions.** Fixing the test set before training any model and using it consistently across all three models and all four evaluation conditions was essential for making the accuracy gap measurable and the model comparisons valid. Without this, different models could have been tested on different samples, which would have made the comparison results unreliable.

**MobileBERT over DistilBERT.** MobileBERT's mobile-specific design produced a smaller and faster model for the Android deployment, which shows that choosing a model based on the deployment target is more important than choosing based on general benchmark scores alone.

**FastAPI for the backend.** FastAPI's built-in async support and WebSocket handling worked well for this project. The automatic OpenAPI documentation also made it easier to test API endpoints during development without needing separate testing tools.

**Single-file staff dashboard.** Building the dashboard as a single HTML file with embedded CSS and JavaScript removed the need for any build tools or dependency management, making it easy to deploy and maintain without any IT support.

### 10.4.2 Decisions That Could Be Improved

**TTS-generated noise profiles.** Using text-to-speech audio passed through Vosk to generate training noise was a practical decision given the scope of this research, but it is the main weakness in terms of how well the results transfer to real deployment. A better approach would be to collect speech recordings from real Sri Lankan English speakers and use their actual Vosk transcriptions as training data.

**Confidence threshold set by trial and error.** The 0.60 confidence threshold for the hybrid NLU pipeline was chosen through manual testing rather than through a systematic calibration process. A more reliable approach would use a validation set with a defined acceptable false-rejection rate to find the threshold that best suits the hotel operational context.

**Hardcoded department routing.** The intent-to-department mapping is defined as a Python dictionary in the server code. Storing this in a database table would allow hotel administrators to adjust routing without needing to change the code. This was not prioritised because the focus was on the core research functionality.

**No structured logging.** The prototype uses basic print statements during development. Proper structured logging would have made it easier to track and diagnose issues during testing and would be needed for any real-world deployment.

---

## 10.5 Knowledge and Expertise Gained

**On-device machine learning deployment.** Working through the full pipeline — from model training in PyTorch, through TFLite conversion, to deployment on Android — gave a clear understanding of the practical challenges of edge ML deployment. The most useful lesson was that model conversion is not a straightforward export step. It requires careful checking of input/output formats, tokenisation consistency (for example, the `padding='max_length'` requirement for TFLite), and numerical precision across frameworks.

**Experimental design for comparative NLU evaluation.** Designing an experiment that isolates training data composition as the only variable — while keeping model architecture, hyperparameters, and test set the same — required careful planning. The decision to fix the shared test set before training any model was critical to getting valid and interpretable results.

**ASR-NLU pipeline interaction.** This research clearly showed that the gap between clean-text NLU performance and real pipeline performance is real, measurable, and different for each intent. The `towel_request` case — where a 16.33% WER caused F1 to drop to 0.67 under clean-trained evaluation — shows that this gap is not spread evenly and is driven by specific intents with phonetically vulnerable vocabulary. Understanding this relationship is directly relevant to any offline voice processing system.

**Per-intent WER as a diagnostic tool.** The WER analysis shows that Vosk's transcription difficulty varies significantly across intents — from 6.78% for `emergency` to 16.83% for `temperature_control`. Intents with higher WER are also the ones most likely to show accuracy degradation when training only on clean text. This is a useful practical insight: before deploying any on-device STT+NLU system, computing per-intent WER on a representative set of utterances helps identify which categories are most at risk from pipeline-induced accuracy loss.

**Privacy-preserving system architecture.** Designing a system where privacy is guaranteed by the architecture itself — rather than by policies or agreements — required thinking carefully about data flows at every level of the system. This experience showed that privacy-by-design is an architectural decision that needs to be made from the start, not added later.

**Research methodology.** The Design Science Research approach combined with iterative prototyping worked well for this project, which involved a lot of technical uncertainty. The most valuable methodological lesson was that starting with a working prototype early reveals real constraints that cannot be predicted from planning alone — both the Vosk accent mismatch problem and the hybrid NLU pipeline design came out of hands-on implementation rather than upfront design.

---

## 10.6 Summary

The prototype successfully shows that a low-cost, offline voice assistant for hospitality services is achievable using small-scale neural models on commodity Android hardware. The main research contribution — beyond the working system itself — is measuring the accuracy gap between clean-text NLU evaluation and real offline pipeline performance, and showing that Vosk-specific noise-aware training closes that gap. Model A's 8.73 percentage point accuracy drop from clean text (98.07%) to Vosk output (89.34%) confirms that standard clean-text benchmarks overstate the actual performance of NLU models in offline STT pipelines. Model C's recovery to 99.06% on Vosk output — a 111.3% gap recovery that exceeds the clean-text baseline — shows that paired noise-aware training is an effective solution.

The main limitation that affects how these results should be interpreted is that both the WER measurement and the training noise were derived from TTS-synthesised speech rather than real speaker recordings. The practical size of the accuracy gap and the effectiveness of the noise-aware fix would need to be confirmed with real hotel guest speech before making firm conclusions about production deployment readiness. Deploying the system in an actual hotel and evaluating it on real guest utterances is the most important next step for future research.
