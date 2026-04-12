# CHAPTER 2: LITERATURE REVIEW

## 2.1 Introduction

Voice-based AI assistants in hotel rooms have received growing attention from both researchers and the industry. Buhalis and Moldavska (2021, 2022) found that in-room voice AI is used for room service ordering, environment controls, wake-up calls, and general queries — and that guests consistently report benefits including convenience, faster service, and reduced front desk workload. A 2025 study in the Journal of Theoretical and Applied Electronic Commerce Research (JTAECR), using structural equation modelling on 529 survey responses, confirmed that voice assistant interactivity and responsiveness positively influence guest evaluations and hotel reputation. Amazon's Alexa for Hospitality, deployed at Marriott, Hilton, and Best Western since 2018, is the most widely used commercial implementation.

In Sri Lanka, tourism is a significant part of the national economy. The country attracted over 2 million international tourists in 2024 across cultural, wellness, and eco-tourism segments (SLTDA, 2024). A 2022 Springer study on digital readiness in Sri Lanka's tourism sector found that while leading hotel chains have adopted basic tools like online booking systems, AI-powered guest services remain very limited. This gap represents a clear opportunity for an affordable, offline voice assistant suited to the Sri Lankan context.

---

## 2.2 Challenges Hotel Businesses Face in Sri Lanka

Sri Lankan hotels face several practical challenges that prevent adoption of mainstream commercial voice assistant solutions.

**Internet connectivity dependency and cost.** Cloud voice assistants like Alexa for Hospitality rely entirely on a live AWS connection — if the internet drops, the system stops working. While Sri Lanka's connectivity has improved (Starlink satellite internet launched in 2023, reaching rural tourism areas such as hill country resorts and the Arugam Bay and Pasikuda coasts), internet remains a managed cost rather than a free utility. A fully offline system removes this dependency entirely.

**High cost of commercial solutions.** Alexa for Hospitality requires proprietary Echo hardware (USD 99–200 per device) plus AWS subscription fees that grow with usage. For a mid-range Sri Lankan hotel deploying across 50 rooms, the combined cost is not financially sustainable. Buhalis and Moldavska (2022) noted this barrier but did not suggest a more affordable alternative.

**Accent and language mismatch.** Commercial speech recognition systems are mainly trained on North American and British English. Studies have found around 30% accuracy loss on non-native English accents — a significant problem in Sri Lanka where both guests and staff typically speak South Asian English. India is consistently Sri Lanka's largest source of tourists (416,974 arrivals in 2024, approximately 20% of all international visitors — SLTDA, 2024), making South Asian accent robustness a direct operational requirement.

**Limited IT expertise.** The 2022 Springer study found that most Sri Lankan hotel operators lack in-house technical staff capable of setting up and maintaining cloud-connected AI systems, creating vendor dependency and adding cost and downtime risk.

**Operational inefficiency.** Without automated routing, every guest request — whether intended for housekeeping, maintenance, or the kitchen — passes through the front desk first. This creates bottlenecks during busy periods and increases miscommunication risk. Hwang and Erdem (2025) confirmed that voice assistants reduce repetitive front desk calls, but only when the system is well-designed for the operational context.

**Privacy risks.** Buhalis and Moldavska (2022) found 67% of guests uncomfortable with cloud-connected microphones in hotel rooms and documented risks of third-party skills accessing guest audio without consent. Hotels serving European guests face GDPR exposure, and this risk grows as data protection legislation expands across Asia.

---

## 2.3 Impact of the Challenges

These challenges have direct consequences for guest experience, operational efficiency, and market competitiveness.

Slow service delivery reduces guest satisfaction. Research consistently identifies response speed as one of the most important factors in hotel guest satisfaction (Buhalis & Moldavska, 2022), and the 2025 JTAECR study confirmed that hotels with responsive voice systems receive higher guest evaluation scores than those relying solely on telephone-based request handling.

Without automated routing, every request passes through the front desk before reaching the right department — delaying resolution and removing any management visibility into service performance or recurring problems.

As global booking platforms increasingly list smart-room features as searchable amenities, hotels without technology-enhanced services risk losing higher-spending international guests. Combined with expanding data protection legislation across Asia and Europe, cloud-based deployments that send guest audio to overseas servers carry growing legal exposure even without an actual breach.

---

## 2.4 Solutions Proposed in Existing Research and Industry

Researchers and technology companies have proposed various solutions to address voice-driven hotel service delivery. These can be grouped into three categories: commercial cloud platforms, offline speech recognition, and compact NLU models for on-device processing.

### 2.4.1 Commercial Cloud-Based Voice Assistants

Amazon's Alexa for Hospitality is the most widely deployed commercial solution. The platform allows hotels to manage devices at a property level, create custom voice skills, and includes some privacy features such as a microphone disconnect button. Hwang and Erdem (2025) studied how this system affects hotel staff workload and found it can reduce repetitive tasks and improve response times. The 2025 JTAECR study also confirmed that voice assistant features such as interactivity and information provision positively affect hotel reputation through improved guest evaluations.

### 2.4.2 Edge-Based and Offline Speech Recognition

Vosk (Alpha Cephei) is an open-source offline speech recognition toolkit supporting over 20 languages with model sizes ranging from 50MB to 1.8GB. Korkmaz et al. (2025) built a prototype edge-based STT system using Vosk on a Raspberry Pi and showed it can provide real-time, low-latency speech processing without cloud connectivity. Chantrapornchai and Suchato (2022) similarly demonstrated that offline speech recognition on a Raspberry Pi is practical for command-style IoT control tasks.

Radford et al. (2023) introduced Whisper, a large model trained on 680,000 hours of multilingual audio that handles diverse accents and background noise well. However, its computational requirements — especially for the medium and large variants — make it unsuitable for running on commodity mobile hardware.

### 2.4.3 Compact NLU Models for On-Device Intent Classification

BERT (Devlin et al., 2019) established bidirectional pre-training as the standard for natural language understanding, but its 110 million parameters (~440MB) make it too large for mobile deployment. Several compressed alternatives have been developed:

Sanh et al. (2019) introduced DistilBERT, which retains 97% of BERT's language understanding at 40% smaller size and 60% faster inference using knowledge distillation. Sun et al. (2020) proposed MobileBERT, designed specifically for mobile devices, achieving 4.3× compression and 5.5× faster inference versus BERT-BASE with only 0.6 points lower on the GLUE benchmark — running at 62ms per inference on a Google Pixel 4.

For server-side NLU, Bunk et al. (2020) introduced Rasa DIET, which handles both intent classification and entity recognition in a single model while training six times faster than fine-tuned BERT. Bujel et al. (2021) showed that fine-tuning BERT on as few as 1,000 domain-specific examples produces better results than general pre-trained models for specialised intent classification.

### 2.4.4 Privacy-Preserving Edge Computing

Shi et al. (2016) established the concept of edge computing in their widely-cited IEEE IoT Journal paper, showing how processing data locally reduces latency, improves privacy, and decreases bandwidth usage. Korkmaz et al. (2025) demonstrated that fully offline on-device models can reach competitive accuracy for domain-specific tasks while ensuring voice data never leaves the user's device — something cloud systems cannot guarantee regardless of their privacy policies.

---

## 2.5 How Existing Solutions Have Helped Address the Challenges

Existing solutions have made progress in addressing individual parts of the problem, and there is good evidence from the literature to support their effectiveness.

**Cloud-based commercial voice assistants** have clearly shown that hotel guests find value in voice interaction. Buhalis and Moldavska (2021) found that the benefits of hotel voice assistants consistently outweigh the drawbacks for both guests and hotel operators. Hwang and Erdem (2025) reported concrete improvements: fewer repetitive calls to front desk staff, faster information delivery, and a more modern perception of the hotel's service. The 2025 JTAECR study also confirmed that interactive voice assistants positively affect hotel reputation scores.

**Edge-based speech recognition** research has shown that offline STT is technically possible on standard hardware. Korkmaz et al. (2025) demonstrated that Vosk-based edge STT can match cloud solutions in terms of response speed for short command-style inputs. Custom domain language models for Vosk have been shown to reduce Word Error Rate by up to 40% compared to generic offline models (Alpha Cephei, 2023), which substantially closes the accuracy gap with cloud-based alternatives. Chantrapornchai and Suchato (2022) confirmed that Raspberry Pi devices can handle real-time offline speech recognition for command recognition tasks.

**Compact NLU models** have shown that on-device natural language understanding is feasible. MobileBERT's 62ms inference latency on a Google Pixel 4 (Sun et al., 2020) shows that transformer-based intent classification is practical for real-time use on mobile devices. DistilBERT (Sanh et al., 2019) demonstrated that knowledge distillation can produce smaller models without a large drop in accuracy. Wang et al. (2025) survey on-device AI optimisation techniques, including parallel decoder architectures that achieve significant inference speedups with substantially fewer parameters.

**Privacy-preserving design** has also been technically validated. Korkmaz et al. (2025) showed that fully local processing can reach competitive accuracy while eliminating the data exposure risk that comes with cloud-based systems.

However, while each of these solutions addresses one aspect of the problem — whether it is offline STT, on-device NLU, or privacy — none of them combines all of these aspects into a complete, integrated system designed specifically for hospitality, running on affordable hardware, and tested in a developing-country context.

---

## 2.6 Knowledge Contribution of This Research and Model Performance

### 2.6.1 Identified Research Gaps

Table 2.1 compares existing solutions across seven important dimensions to show where this research contributes.

**Table 2.1: Comparison Matrix of Existing Solutions**

| Solution | Offline Capable | On-Device Processing | Hospitality Domain | Low-Cost Hardware | Privacy Preserving | Staff Integration | End-to-End System |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Buhalis & Moldavska (2021, 2022) | No | No | Yes | No | No | No | No |
| Alexa for Hospitality | No | No | Yes | No | Partial | Yes | Yes |
| Korkmaz et al. (2025) – Edge STT | Yes | Yes | No | Yes | Yes | No | No |
| Custom Vosk LM (Alpha Cephei, 2023) | Yes | Yes | No | Yes | Yes | No | No |
| Chantrapornchai & Suchato (2022) | Yes | Yes | No | Yes | Yes | No | No |
| Whisper (Radford et al., 2023) | Partial | No | No | No | Partial | No | No |
| MobileBERT (Sun et al., 2020) | Yes | Yes | No | Yes | Yes | No | No |
| DistilBERT (Sanh et al., 2019) | Yes | Partial | No | Partial | Yes | No | No |
| Rasa DIET (Bunk et al., 2020) | Partial | No | No | No | Partial | No | Partial |
| **This Research** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

As shown in the table, none of the existing solutions satisfies all seven dimensions at the same time. Four research gaps are identified:

- **Gap 1 — STT Noise Impact on NLU (Core Research Gap):** No published research has measured how much accuracy is lost when offline STT transcription noise (specifically from Vosk) is passed to a downstream NLU intent classifier, and no research has shown that training the NLU model on STT-transcribed data can recover this accuracy. NLU models are typically trained and evaluated on clean text, but in real offline deployments the model receives noisy, error-prone STT output. This mismatch between training conditions and deployment conditions has not been studied or addressed in the hospitality domain. This is the primary research gap that this study addresses.

- **Gap 2 — On-Device NLU for Hospitality in a Real STT Pipeline:** No published research evaluates fine-tuned compact transformer models (MobileBERT/TFLite) on a hospitality-specific intent dataset within a real offline STT pipeline. Existing NLU research either uses clean text inputs or operates in a server-based environment, neither of which reflects how such a model actually performs when receiving Vosk-transcribed speech on a mobile device.

- **Gap 3 — Integrated Offline Hospitality System:** No existing work combines offline STT, on-device NLU, privacy-preserving local processing, and real-time staff dashboard integration into a single end-to-end system designed for hospitality. Individual components have been studied in isolation, but no research has integrated them into a complete, working hotel voice assistant that runs without cloud connectivity and routes requests to department-specific staff in real time.

- **Gap 4 — Developing Economy Context:** No such system has been designed, built, and evaluated for the specific operational and economic conditions of hotels in developing economies such as Sri Lanka, where low hardware cost, local serviceability, and freedom from recurring cloud fees are essential requirements for practical adoption.

### 2.6.2 Empirical Approach to Addressing Gap 1

To address Gap 1, three MobileBERT models were trained on a custom hospitality intent dataset covering 18 intent categories — using clean text only (Model A), Vosk-transcribed text only (Model B), and a mixed clean+Vosk dataset (Model C). All three were converted to TensorFlow Lite and evaluated on the same held-out test set of Vosk-transcribed utterances. The full results, including accuracy figures, confusion matrices, and per-intent breakdowns, are presented in Chapter 8.

---

## 2.7 Technology Tools and Their Suitability

Each technology was selected based on specific performance, cost, and deployment requirements identified in Sections 2.2 and 2.3.

### 2.7.1 Vosk for On-Device Speech-to-Text

Vosk was chosen because it works fully offline and the `vosk-model-small-en-in-0.4` variant (Indian English, ~50MB) is suited to the South Asian English accents of Sri Lankan hotel guests and staff. At 50MB it fits comfortably on a commodity Android tablet. Korkmaz et al. (2025) showed that Vosk achieves comparable latency to cloud STT for short command-style inputs, and custom domain language models can reduce Word Error Rate by up to 40% (Alpha Cephei, 2023), providing a clear path for further accuracy improvement.

### 2.7.2 MobileBERT and TensorFlow Lite for On-Device NLU

MobileBERT was chosen over DistilBERT and Rasa DIET because it was designed specifically for mobile inference. It converts to TFLite with a 26MB footprint and runs at 62ms per inference on a Pixel 4. DistilBERT has not been optimised for TFLite on Android; Rasa DIET requires a server process, which conflicts with the on-device requirement. Critically, the noise-aware training approach used in this research (Model C) achieves 99.06% accuracy on real Vosk output — substantially better than standard clean-text training produces (89.34% on Vosk input).

### 2.7.3 Android Tablets for Hardware Deployment

Android tablets were chosen over Amazon Echo or Raspberry Pi for three reasons: **cost** (USD 50–150 in Sri Lanka vs USD 200+ for smart speakers), **availability** (local technicians can source and repair them without specialist knowledge or importing parts), and **platform support** (well-established TFLite, Vosk, and WebSocket tooling). The 2022 Springer study on Sri Lankan digital tourism found that locally serviceable hardware is a key requirement for technology adoption in this market.

### 2.7.4 FastAPI and WebSocket for Backend Communication

FastAPI handles both HTTP and WebSocket communication efficiently with low overhead, suitable for running on standard hotel hardware within a local network. WebSockets provide real-time two-way communication between guest devices and the staff dashboard — a capability not offered by Alexa for Hospitality. Running the backend on hotel hardware means no recurring hosting fees and all guest data stays on-premises, consistent with the edge computing privacy model described by Shi et al. (2016).

### 2.7.5 Summary of Tool Suitability

Table 2.2 maps each selected tool to the challenges it addresses.

**Table 2.2: Challenge-to-Tool Mapping**

| Challenge (Section 2.2) | Selected Tool | Research Gap Addressed |
|---|---|---|
| Internet dependency and cost | Vosk offline STT | Gap 3 — fully offline, no cloud dependency |
| High commercial platform cost | Android tablets + open-source stack | Gap 4 — USD 50–150 hardware, zero licensing fees |
| Non-native accent degradation | Vosk Indian English model | Gap 1 — acoustically suited to South Asian English |
| STT noise degrading NLU accuracy | Noise-aware MobileBERT training | Gap 1 — core research contribution |
| Limited IT expertise | Android + FastAPI | Gap 4 — widely used platforms, locally serviceable |
| Operational inefficiency | WebSocket real-time routing | Gap 3 — department-specific request routing |
| Privacy and data exposure | On-device processing | Gap 3 — all voice data stays within hotel LAN |

---

## 2.8 Summary

This chapter reviewed the existing literature and identified the research gaps this study addresses. Section 2.2 described the main barriers preventing Sri Lankan hotels from adopting mainstream voice assistant solutions — internet dependency, high cost, accent mismatch, limited IT expertise, and privacy risk. Section 2.3 showed how these barriers affect guest experience, operational efficiency, and market competitiveness.

Sections 2.4 and 2.5 reviewed existing solutions: commercial platforms like Alexa for Hospitality, offline STT research using Vosk, and compact NLU models like MobileBERT and DistilBERT. While each addresses part of the problem, none provides a complete integrated offline hospitality system.

Section 2.6 identified four research gaps and presented empirical results. The key finding is an 8.73 percentage point NLU accuracy drop when a model trained on clean text is tested on real Vosk output — the first documented measurement of this gap in the hospitality domain. Noise-aware training (Model C) recovers +9.72 percentage points, reaching 99.06% accuracy on Vosk output. Section 2.7 explained why each technology was selected, including the Indian English Vosk model suited to the South Asian English accent of the target users.

The following chapter presents the research methodology used to design, build, and evaluate the complete system.
