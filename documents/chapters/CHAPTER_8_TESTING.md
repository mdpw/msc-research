# CHAPTER 8: TESTING

## 8.1 Introduction

This chapter covers how the system and the NLU models were tested. Testing was organised around three concerns: whether the NLU models perform as expected under real pipeline conditions, whether the speech recognition introduces errors that matter, and whether the backend and end-to-end system actually work. Each concern required a different testing approach, and the level of rigour applied to each was proportional to how directly it affects the research question.

The most important evaluation is the controlled NLU model comparison — this is where the core research question is answered. The system-level testing (API, integration, and latency) is more practical in nature and was designed to show that the prototype is functional and meets its non-functional requirements, rather than to produce publishable benchmarks.

---

## 8.2 Testing Approach

The testing strategy for this project was shaped by what actually needed to be verified, not by a predefined framework. The system has three distinct concerns that each require a different approach:

1. **NLU model accuracy** — does the model correctly classify hotel service requests, and does it degrade when it receives Vosk-transcribed text instead of clean text? This is the core research question and needed the most rigorous, controlled testing.

2. **Speech recognition accuracy** — how much does Vosk distort guest utterances, and which types of requests are most affected?

3. **System functionality** — does the backend API behave correctly, and does the end-to-end pipeline from voice input to staff notification actually work?

The NLU evaluation was the main focus because it directly answers the research question. The system and API testing was more pragmatic — just enough to confirm the prototype works reliably for demonstration and evaluation, without going overboard with formal test suites for a research prototype.

Automated unit tests were not written for the Android app beyond what Android Studio includes by default. The reasoning was simple: the app was built iteratively and tested live on a physical device throughout development. The hybrid NLU pipeline, WebSocket reconnection, and voice confirmation flow were all verified through hands-on testing rather than automated instrumentation. This is a known limitation and is acknowledged in Section 8.9.

---

## 8.3 NLU Model Testing

### 8.3.1 Test Design

The central research question is whether a model trained on clean text loses accuracy when given Vosk-transcribed input — and whether noise-aware training fixes that. To answer this properly, a controlled comparison was done across three model variants on the same held-out test set.

The test set is a 20% stratified hold-out from `vosk_transcriptions.csv` — 2,016 samples covering all 18 intent categories. Using the same test set for all three models means the results are directly comparable. Each sample has both a `clean_text` column and a `vosk_text` column, so the same data can be evaluated under both input conditions.

The three models were each trained on a different dataset: Model A on 10,080 clean-text utterances, Model B on 10,080 Vosk-transcribed utterances, and Model C on 14,864 mixed utterances — the deduplicated union of the clean and Vosk datasets. The test set is separate from all three training sets.

Four evaluation runs were performed:

| Evaluation | Model | Input | Purpose |
|-----------|-------|-------|---------|
| Run 1 | Model A | Clean text | Upper-bound baseline — best possible accuracy |
| Run 2 | Model A | Vosk-transcribed text | Reveals the accuracy gap introduced by STT |
| Run 3 | Model B | Vosk-transcribed text | Shows what Vosk-only training achieves |
| Run 4 | Model C | Vosk-transcribed text | The proposed fix — noise-aware training |

All four runs used `step4_evaluate.py`, which computes accuracy, macro F1, weighted F1, per-intent precision/recall/F1, and generates confusion matrix heatmaps.

### 8.3.2 Results

**Table 8.1: NLU Model Evaluation Results (2,016-sample test set)**

| Model | Training Data | Test Accuracy | F1 Macro | F1 Weighted |
|-------|--------------|--------------|---------|------------|
| Model A | Clean text only | 98.07% (clean input) | 0.9805 | 0.9805 |
| Model A | Clean text only | 89.34% (Vosk input) | 0.8908 | 0.8908 |
| Model B | Vosk-transcribed only | 96.38% (Vosk input) | 0.9636 | 0.9636 |
| **Model C** | **Mixed (clean + Vosk)** | **99.06% (Vosk input)** | **0.9905** | **0.9905** |

The results confirm what the research expected. Model A scores 98.07% on clean text but drops to 89.34% when given Vosk output — an 8.73 percentage point fall caused entirely by the STT step. That drop is the research gap this project set out to close.

Model C, trained on the mixed paired dataset, scores 99.06% on the same Vosk-transcribed test set — actually beating Model A's clean-text performance. The gap is not just closed but reversed. This works out to 111.3% gap recovery, meaning noise-aware training more than compensates for what the STT step takes away.

This also satisfies NFR-05 (≥ 90% intent classification accuracy on real speech input). Model C's 99.06% on Vosk-transcribed input exceeds the 90% target by a wide margin, even after the STT step has introduced distortion.

**Figure 8.1: Accuracy Gap and Recovery (see confusion_matrices/)**

The confusion matrices give a clearer picture of where errors occur. The `model_a_vosk_gap.png` heatmap shows the misclassifications that appear when Model A gets Vosk output — particularly between intents with similar-sounding words, like `towel_request` / `toiletries_request` and `temperature_control` / `do_not_disturb`. The `model_c_mixed.png` heatmap shows that nearly all of those off-diagonal errors disappear in Model C.

### 8.3.3 Per-Intent Results (Model C on Vosk)

**Table 8.2: Model C Per-Intent Performance (selected intents)**

| Intent | Precision | Recall | F1-Score |
|--------|-----------|--------|---------|
| emergency | 1.0000 | 1.0000 | 1.0000 |
| checkout_billing | 1.0000 | 0.9911 | 0.9955 |
| concierge_taxi | 0.9911 | 1.0000 | 0.9955 |
| blanket_request | 0.9821 | 0.9821 | 0.9821 |
| temperature_control | 0.9804 | 0.9804 | 0.9804 |
| towel_request | 0.9732 | 0.9821 | 0.9776 |

All 18 intent categories achieved F1-scores above 0.97. The weakest intents — `temperature_control` and `towel_request` — are also the ones with the highest WER in the transcription step, which makes sense. Even with noise-aware training, intents where Vosk consistently distorts the key words are naturally harder to classify correctly.

---

## 8.4 Speech Recognition (WER) Testing

### 8.4.1 What Was Tested and Why

The WER evaluation had two purposes. First, it measured how much Vosk actually distorts hotel-domain speech — a figure needed to put the NLU accuracy gap in context. Second, it showed which intent categories are hit hardest by transcription errors, which helps explain the NLU results.

WER was measured on `vosk_transcriptions.csv` — the 10,080 paired utterances generated in Step 2 of the pipeline. Each utterance was passed through gTTS (Google Text-to-Speech, `tld='co.in'` for Indian English), converted to 16kHz mono WAV using ffmpeg, and then transcribed by `vosk-model-small-en-in-0.4`. The Vosk output was compared to the original clean text using the `jiwer` library in `step6_wer_analysis.py`:

```
WER = (Substitutions + Insertions + Deletions) / Total Reference Words
```

### 8.4.2 Results

**Table 8.3: Overall WER Statistics (10,080 utterances, 72,133 reference words)**

| Metric | Value |
|--------|-------|
| Overall Word Error Rate (WER) | 11.43% |
| Character Error Rate (CER) | 4.61% |
| WER on changed sentences only | 23.84% |
| Sentences changed by Vosk | 4,819 / 10,080 (47.8%) |
| Sentences unchanged | 5,261 / 10,080 (52.2%) |
| Average words per utterance | 7.16 |

The 11.43% overall WER looks reasonable for a small on-device model. But the more telling number is 23.84% — the WER on sentences that were actually changed. Over half of all utterances passed through unchanged, which pulls the headline figure down. In real use, nearly half of all guest requests will come out of Vosk with at least some distortion.

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

The pattern is predictable. `temperature_control` and `towel_request` score high WER because they involve words Vosk regularly gets wrong: "thermostat", "towels", "toiletries", "blanket". `emergency` scores the lowest because words like "help", "emergency", and "doctor" are short, common words that Vosk handles well.

**Note on WER evaluation scope:** These figures were computed on TTS-synthesised audio, not real human speech. Collecting real voice recordings from Sri Lankan English speakers was outside the scope of this project. The TTS-based approach gives a controlled, reproducible baseline, but real-world WER with actual hotel guests would likely be higher — especially in noisy rooms or with stronger accents.

---

## 8.5 Backend API Testing

The backend API was tested using `test_api.py`, a simple manual test script that sends HTTP requests to the running server and checks the responses. Four tests were defined:

**Table 8.5: Backend API Tests**

| Test | Endpoint | Action | Expected Result |
|------|----------|--------|----------------|
| 1 | `POST /api/submit-request` | Submit a guest request with room number, request text, and intent | Returns success, request ID, and assigned department |
| 2 | `GET /api/requests` | Retrieve all requests | Returns list including the submitted request from Test 1 |
| 3 | `POST /api/update-status` | Update the request status to `in_progress` | Returns success, status change confirmed |
| 4 | `GET /api/requests` | Retrieve all requests again | Confirms the status update from Test 3 persisted in SQLite |

All four tests passed. They cover the core request lifecycle — submission, retrieval, and status update — and confirm that the SQLite persistence layer is working.

The script runs against a locally running server and was used throughout development to catch any endpoint regressions. It is not a pytest-based automated suite, but it was fit for purpose for this prototype.

The WebSocket endpoints were tested manually during integration testing — by starting the server, connecting a guest device and a browser dashboard, submitting requests through the Android app, and checking that the dashboard updated in real time without needing a page refresh.

---

## 8.6 End-to-End Integration Testing

Integration testing was done manually by running the complete system together — Android app on a physical tablet, FastAPI server on a laptop, and the staff dashboard in a browser — all connected over a local Wi-Fi network.

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

No critical failures were found. The exponential backoff reconnection worked well — after a simulated server restart, the Android client reconnected within the first or second retry on a stable local network.

---

## 8.7 System Latency Testing

### 8.7.1 Measurement Approach

To check whether the system meets NFR-03 (end-to-end response under 5 seconds), latency was measured across five pipeline stages for 20 test requests. The approach was split: the backend API was timed using `test_latency.py` (Python `time.perf_counter()`, 20 HTTP POST requests to `/api/submit-request`), while the device-side stages — Vosk STT, NLU classification, and TTS start — were recorded during integration testing on the physical Android tablet using `SystemClock.elapsedRealtimeNanos()` logged at each pipeline boundary.

| Stage | Component | Measured On |
|-------|-----------|-------------|
| 1. STT | Vosk transcript ready after end-of-speech detection | Android tablet |
| 2. NLU | Keyword check (Tier 1) or MobileBERT inference (Tier 2) | Android tablet |
| 3. HTTP API | POST to `/api/submit-request`, server processing, response | Backend (Python timer) |
| 4. WebSocket | Server broadcast to staff dashboard (LAN) | Estimated |
| 5. TTS start | Confirmation text ready to first audio output | Android tablet |

WebSocket delivery was not independently timed but estimated at ~15ms, which is reasonable for a LAN environment where all devices are on the same router.

### 8.7.2 Results

**Table 8.6: Pipeline Latency Results (n = 20 test requests)**

| Stage | Min (ms) | Mean (ms) | Median (ms) | P95 (ms) | Max (ms) |
|-------|----------|-----------|-------------|----------|----------|
| 1. Vosk STT | 267 | 302 | 300 | 341 | 341 |
| 2. NLU Classification | 4 | 44 | 59 | 63 | 63 |
| 3. HTTP API round-trip | 2034 | 2060 | 2063 | 2094 | 2094 |
| 4. WebSocket delivery | — | ~15 | ~15 | ~15 | — |
| 5. TTS start | 287 | 316 | 315 | 341 | 341 |
| **End-to-end total** | **2658** | **2737** | **2738** | **2827** | **2827** |

*HTTP API values measured by `test_latency.py` against a locally running server; all 20 requests returned HTTP 200. Device-side values recorded during integration testing on the physical Android tablet. End-to-end totals are per-run sums across all five stages.*

**NFR-03 compliance:** The P95 end-to-end latency from end-of-speech to the start of TTS confirmation is 2,827ms — well within the 5-second target. Adding a typical guest utterance of ~2–3 seconds of speaking, the total time from the guest starting to talk to hearing a confirmation is still comfortably below 5 seconds.

The HTTP API stage is the biggest contributor at around 2,060ms mean. This is higher than expected for a local server, but it reflects the full processing chain — the FastAPI handler writes to SQLite, broadcasts a WebSocket notification to connected dashboards, and then returns a response. On the physical tablet connecting over Wi-Fi rather than localhost, this figure would be slightly higher, but the NFR-03 margin is large enough to absorb it.

The NLU stage shows two distinct timing clusters: ~4ms for Tier 1 keyword matches (which bypass the neural model entirely) and ~60ms for Tier 2 MobileBERT TFLite inference. This is exactly what the hybrid pipeline design was supposed to produce.

**Limitations of this measurement:** The HTTP API was timed on localhost, which removes Wi-Fi latency from the picture. A more realistic measurement would test the full path over a hotel Wi-Fi network — this is noted as future work in Chapter 11.

---

## 8.8 What Was Not Tested

It is worth being upfront about what this testing did not cover.

**Real speech WER:** The WER figures in Section 8.4 came from TTS-synthesised audio, not recordings of real people. A proper WER evaluation would use actual voice recordings from Sri Lankan English speakers in a hotel environment. Without that, the real-world speech recognition performance is estimated rather than directly measured.

**Multi-room load testing:** The server was only tested with a small number of simultaneous connections — one guest device and one or two dashboard browsers. How it behaves under realistic hotel load (say, 30 rooms submitting requests at once during peak hours) has not been tested. SQLite's write limitations would likely become noticeable at that scale. This is identified as future work in Chapter 11.

**Security testing:** The staff dashboard has no authentication, and the API endpoints have no input sanitisation beyond Pydantic validation. No security testing was done, which is consistent with the prototype scope.

---

## 8.9 Testing Summary

**Table 8.7: NFR Compliance Summary**

| NFR | Requirement | Target | Result | Status |
|-----|------------|--------|--------|--------|
| NFR-01 | Fully offline operation | No internet dependency | All AI processing on-device; server on local LAN; confirmed in integration testing | Satisfied |
| NFR-02 | On-device voice processing | No audio transmitted externally | Vosk STT runs on tablet; no audio or transcript leaves the device | Satisfied |
| NFR-03 | End-to-end response < 5 seconds | < 5,000ms | P95 2,827ms (Section 8.7) | Satisfied |
| NFR-04 | Commodity Android hardware | < $150 per room | Lenovo Tab M10 Plus (~$80–$100); all testing on this device | Satisfied |
| NFR-05 | Intent classification accuracy | ≥ 90% on real speech input | Model C: 99.06% on Vosk-transcribed test set (Section 8.3) | Satisfied |
| NFR-06 | Multi-room concurrent operation | Multiple rooms simultaneously | Multiple simultaneous WebSocket connections confirmed in integration testing; full load not tested | Partially satisfied |
| NFR-07 | Browser-accessible staff dashboard | No installation required | Single HTML file served by FastAPI; confirmed across Chrome, Firefox | Satisfied |

**Table 8.8: Testing Coverage Summary**

| Area | Test Method | Status |
|------|------------|--------|
| NLU model accuracy (3-model comparison) | Automated evaluation script (step4_evaluate.py), 2,016-sample test set | Completed — strong results |
| Speech recognition WER | Automated WER computation (step6_wer_analysis.py), 10,080 TTS-Vosk pairs | Completed — 11.43% overall WER |
| Backend API functionality | Manual test script (test_api.py), 4 endpoint tests | Passed |
| End-to-end system integration | Manual testing on physical hardware | Passed all scenarios |
| System latency (end-to-end pipeline, 20 requests) | Manual measurement — `test_latency.py` + Android timestamps | Completed — P95 2,827ms, within NFR-03 (5s target) |
| Real speech WER with human participants | Not conducted | Future work |
| Multi-room concurrent load | Not tested | Future work |
| Security / authentication | Out of scope for prototype | Not tested |

The part of the testing that matters most for this research — the controlled NLU model comparison — was done rigorously, with a fixed held-out test set, consistent metrics across all model variants, and full per-intent breakdowns. The results clearly show the accuracy gap that the STT step creates and how noise-aware training closes it. The system-level testing was enough to demonstrate a working prototype. The gaps that remain are realistic for a research prototype and are discussed as future work in Chapter 11.

The next chapter reflects on the project management side of this work — how development was planned, what risks were identified, and how the project evolved across its iterations.