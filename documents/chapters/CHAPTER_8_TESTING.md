# CHAPTER 8: TESTING

## 8.1 Testing Approach

The testing strategy for this project was shaped by what actually needed to be verified, not by a predefined framework. The system has three distinct concerns that each require a different approach:

1. **NLU model accuracy** — does the model correctly classify hotel service requests, and does it degrade when it receives Vosk-transcribed text instead of clean text? This is the core research question and needed the most rigorous, controlled testing.

2. **Speech recognition accuracy** — how much does Vosk distort guest utterances, and which types of requests are most affected?

3. **System functionality** — does the backend API behave correctly, and does the end-to-end pipeline from voice input to staff notification actually work?

The NLU evaluation was treated as the primary testing concern because it directly answers the research question. The system and API testing was more pragmatic — enough to verify that the prototype works reliably for demonstration and evaluation, without investing in formal test suites that would be disproportionate for a research prototype.

Automated unit and integration tests were not written for the Android application beyond the basic context verification included by Android Studio by default. The reasoning was practical: the app was developed iteratively and tested live on a physical device throughout development. The hybrid NLU pipeline, WebSocket reconnection, and voice confirmation flow were all validated through hands-on testing rather than automated instrumentation. This is a known limitation of the prototype and is discussed in Section 8.7.

---

## 8.2 NLU Model Testing

### 8.2.1 Test Design

The core research question asks whether a model trained on clean text loses accuracy when it receives Vosk-transcribed input — and whether noise-aware training fixes it. Testing this required a controlled comparison across three model variants on a shared, held-out test set.

The test set used for all model evaluations is a 20% stratified hold-out from `vosk_transcriptions.csv` — 2,016 samples covering all 18 intent categories. Because the same test set is used for all three models, results are directly comparable. The test set contains both a `clean_text` column and a `vosk_text` column for each sample, which allows the same data to be evaluated under both input conditions.

Four evaluation runs were performed:

| Evaluation | Model | Input | Purpose |
|-----------|-------|-------|---------|
| Run 1 | Model A | Clean text | Upper-bound baseline — best possible accuracy |
| Run 2 | Model A | Vosk-transcribed text | Reveals the accuracy gap introduced by STT |
| Run 3 | Model B | Vosk-transcribed text | Shows what Vosk-only training achieves |
| Run 4 | Model C | Vosk-transcribed text | The proposed fix — noise-aware training |

All evaluations were run using `step4_evaluate.py`, which computes accuracy, F1 macro, F1 weighted, per-intent precision/recall/F1, and generates confusion matrix heatmaps for each run.

### 8.2.2 Results

**Table 8.1: NLU Model Evaluation Results (2,016-sample test set)**

| Model | Training Data | Test Accuracy | F1 Macro | F1 Weighted |
|-------|--------------|--------------|---------|------------|
| Model A | Clean text only | 98.07% (clean input) | 0.9805 | 0.9805 |
| Model A | Clean text only | 89.34% (Vosk input) | 0.8908 | 0.8908 |
| Model B | Vosk-transcribed only | 96.38% (Vosk input) | 0.9636 | 0.9636 |
| **Model C** | **Mixed (clean + Vosk)** | **99.06% (Vosk input)** | **0.9905** | **0.9905** |

The results confirm the hypothesis. Model A achieves 98.07% on clean text but drops to 89.34% when given Vosk-transcribed input — an 8.73 percentage point degradation caused purely by the STT step. This is the research gap.

Model C, trained on the mixed paired dataset, achieves 99.06% on the same Vosk-transcribed test set — actually surpassing Model A's clean-text performance. The gap is not just closed but reversed: the noise-aware training approach achieves 111.3% gap recovery.

**Figure 8.1: Accuracy Gap and Recovery (see confusion_matrices/)**

The confusion matrices generated during evaluation provide a more granular view. The `model_a_vosk_gap.png` heatmap shows the main misclassifications that appear when Model A faces Vosk output — notably confusion between intents with overlapping phonology such as `towel_request` / `toiletries_request` and `temperature_control` / `do_not_disturb`. The `model_c_mixed.png` heatmap shows that nearly all of these off-diagonal entries collapse in Model C.

### 8.2.3 Per-Intent Results (Model C on Vosk)

**Table 8.2: Model C Per-Intent Performance (selected intents)**

| Intent | Precision | Recall | F1-Score |
|--------|-----------|--------|---------|
| emergency | 1.0000 | 1.0000 | 1.0000 |
| checkout_billing | 1.0000 | 0.9911 | 0.9955 |
| concierge_taxi | 0.9911 | 1.0000 | 0.9955 |
| blanket_request | 0.9821 | 0.9821 | 0.9821 |
| temperature_control | 0.9804 | 0.9804 | 0.9804 |
| towel_request | 0.9732 | 0.9821 | 0.9776 |

All 18 intent categories achieved F1-scores above 0.97. The weakest performers were the intents with the highest WER in the Vosk transcription step (temperature_control and towel_request), which is consistent with the expectation that intents prone to transcription errors would be harder to classify correctly even after noise-aware training.

---

## 8.3 Speech Recognition (WER) Testing

### 8.3.1 What Was Tested and Why

The WER evaluation served two purposes. First, it quantified how much Vosk actually distorts hotel-domain utterances — a number needed to contextualise the NLU accuracy gap. Second, it identified which intent categories are most affected by transcription errors, which informed the interpretation of the NLU results.

The WER was measured on the `vosk_transcriptions.csv` dataset — the 10,080 paired utterances generated in Step 2 of the research pipeline. Each utterance was passed through gTTS (Google Text-to-Speech, `tld='co.in'` for Indian English pronunciation), converted to 16kHz mono WAV using ffmpeg, and transcribed by `vosk-model-small-en-in-0.4`. The Vosk output was compared against the original clean text to compute WER and CER.

The `step6_wer_analysis.py` script computed the metrics using the `jiwer` library:

```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

### 8.3.2 Results

**Table 8.3: Overall WER Statistics (10,080 utterances, 72,133 reference words)**

| Metric | Value |
|--------|-------|
| Overall Word Error Rate (WER) | 11.43% |
| Character Error Rate (CER) | 4.61% |
| WER on changed sentences only | 23.84% |
| Sentences changed by Vosk | 4,819 / 10,080 (47.8%) |
| Sentences unchanged | 5,261 / 10,080 (52.2%) |
| Average words per utterance | 7.16 |

The 11.43% overall WER looks reasonable for an on-device model. However, the more revealing figure is the 23.84% WER on sentences that were actually changed — over half of all utterances came through unmodified, which pulls the headline number down. In practice, nearly half of all guest utterances will experience some transcription distortion.

**Table 8.4: Per-Intent WER (highest to lowest)**

| Intent | WER | Sentences Changed |
|--------|-----|------------------|
| temperature_control | 16.83% | 63.0% |
| do_not_disturb | 16.77% | 52.7% |
| towel_request | 16.33% | 66.2% |
| toiletries_request | 15.81% | 61.6% |
| laundry_service | 14.52% | 57.5% |
| concierge_general | 13.97% | 55.4% |
| maintenance | 13.21% | 52.3% |
| room_cleaning | 12.88% | 49.1% |
| food_order | 12.10% | 47.7% |
| checkout_billing | 11.54% | 44.6% |
| concierge_taxi | 10.98% | 43.6% |
| pillow_request | 10.43% | 41.8% |
| blanket_request | 10.12% | 40.5% |
| wake_up_call | 9.87% | 39.2% |
| lighting_control | 9.54% | 38.1% |
| noise_complaint | 8.76% | 34.4% |
| misc_request | 8.21% | 30.5% |
| emergency | 6.78% | 26.6% |

The pattern makes sense. `temperature_control` and `towel_request` have high WER because they contain words that Vosk consistently struggles with: "thermostat", "towels", "blanket", "toiletries". `emergency` has the lowest WER because its keywords ("help", "emergency", "doctor") are short, high-frequency words that Vosk handles reliably.

**Note on WER evaluation scope:** These WER measurements were computed on TTS-synthesised audio (text converted to speech using gTTS, then transcribed by Vosk) rather than real recorded human speech. This was a practical necessity — collecting hundreds of real voice recordings from participants with South Asian accents was outside the scope of this research. The TTS-based WER provides a controlled, reproducible baseline, but real-world WER with actual hotel guests would likely be higher, particularly for guests with stronger accents or in rooms with background noise.

---

## 8.4 Backend API Testing

The backend API was tested using `test_api.py`, a manual test script that sends HTTP requests to the running server and verifies the responses. Four tests were defined:

**Table 8.5: Backend API Tests**

| Test | Endpoint | Action | Expected Result |
|------|----------|--------|----------------|
| 1 | `POST /api/submit-request` | Submit a guest request with room number, request text, and intent | Returns success, request ID, and assigned department |
| 2 | `GET /api/requests` | Retrieve all requests | Returns list including the submitted request from Test 1 |
| 3 | `POST /api/update-status` | Update the request status to `in_progress` | Returns success, status change confirmed |
| 4 | `GET /api/requests` | Retrieve all requests again | Confirms the status update from Test 3 persisted in SQLite |

All four tests passed. The tests confirm the core request lifecycle — submission, retrieval, and status update — and that the SQLite persistence layer is working correctly.

The test script runs against a locally running server and was used throughout development to verify that endpoint changes did not break existing behaviour. It is not a pytest-based automated suite, but it was sufficient for the prototype's purposes.

The WebSocket endpoints were tested manually during integration testing — starting the server, connecting a guest device and a browser dashboard, submitting requests through the Android app, and verifying that the dashboard updated in real time without page refresh.

---

## 8.5 End-to-End Integration Testing

Integration testing was performed manually by running the complete system — Android app on a physical tablet, FastAPI server on a laptop, staff dashboard in a browser — connected over a local Wi-Fi network.

**Scenarios tested:**

| Scenario | Outcome |
|----------|---------|
| Guest speaks a request → system classifies → staff dashboard updates in real time | Passed |
| Staff marks request "In Progress" → guest device announces via TTS | Passed |
| Staff marks request "Complete" → guest can submit star rating | Passed |
| Staff sends message → guest device reads it aloud via TTS | Passed |
| Guest says "cancel order [X]" → request cancelled → dashboard updates | Passed |
| Request routed to wrong department → staff transfers to correct department | Passed |
| Server restart → guest device reconnects automatically via exponential backoff | Passed |
| Low-confidence classification (below 0.60 threshold) → "could not understand" response | Passed |
| Multiple staff dashboards open simultaneously → all update in real time | Passed |
| Keyword match (Tier 1) → bypasses neural model, returns 0.99 confidence | Passed |

No critical integration failures were found during this phase. The exponential backoff reconnection worked as expected — after a simulated server restart, the Android client reconnected within the first or second retry attempt on a stable local network.

---

## 8.6 What Was Not Tested

Being transparent about the gaps in testing is important for understanding the scope of these results.

**Latency measurement:** The methodology (Chapter 3) identifies end-to-end latency as an evaluation dimension. However, no automated latency instrumentation was implemented in the codebase. Subjective observation during integration testing suggested the pipeline (voice input to TTS confirmation) completes within 3–5 seconds on a budget Android tablet, which is within the 5-second target (NFR-03). A formal, timestamped latency measurement across multiple test requests was not completed within the project timeline and remains as future evaluation work.

**Real speech WER:** The WER reported in Section 8.3 was computed on TTS-synthesised audio. A proper WER evaluation would use actual voice recordings from Sri Lankan English speakers in a hotel-room-like environment. Without this, the real-world speech recognition performance of the system is estimated rather than measured.

**Multi-room load testing:** The WebSocket server was tested with a small number of simultaneous connections (one guest device, one or two dashboard browsers). Behaviour under realistic hotel load — say, 30 rooms submitting requests simultaneously during peak hours — has not been tested. SQLite's concurrent write limitations would likely become visible at that scale, as noted in the design chapter.

**Security testing:** The staff dashboard has no authentication. The API endpoints have no input sanitisation beyond Pydantic validation. No security testing was conducted, consistent with the prototype's scope.

---

## 8.7 Testing Summary

**Table 8.6: Testing Coverage Summary**

| Area | Test Method | Status |
|------|------------|--------|
| NLU model accuracy (3-model comparison) | Automated evaluation script (step4_evaluate.py), 2,016-sample test set | Completed — strong results |
| Speech recognition WER | Automated WER computation (step6_wer_analysis.py), 10,080 TTS-Vosk pairs | Completed — 11.43% overall WER |
| Backend API functionality | Manual test script (test_api.py), 4 endpoint tests | Passed |
| End-to-end system integration | Manual testing on physical hardware | Passed all scenarios |
| System latency | Not formally measured | Outstanding |
| Real speech WER with human participants | Not conducted | Outstanding |
| Multi-room concurrent load | Not tested | Outstanding |
| Security / authentication | Out of scope for prototype | Not tested |

The testing that matters most for this research — the controlled NLU model comparison — was conducted rigorously with a proper held-out test set, consistent metrics across all model variants, and full per-intent breakdowns. The results clearly demonstrate the accuracy gap introduced by on-device STT and the effectiveness of noise-aware training in closing it. The system-level testing was adequate to demonstrate a working prototype. The outstanding items are realistic limitations of a research prototype rather than oversights, and the most critical ones are identified as future work.
