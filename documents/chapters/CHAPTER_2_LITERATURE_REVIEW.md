# CHAPTER 2: LITERATURE REVIEW

## 2.1 Background: The Hotel Industry and the Rise of Voice Technology

The global hotel industry is one of the largest service sectors in the world, generating over USD 1.5 trillion in revenue annually and employing more than 300 million people across its supply chains (World Travel and Tourism Council, 2023). Guest experience is one of the most important factors that determines a hotel's success, and in recent years hotels have been investing in digital technologies such as mobile check-in systems, smart room controls, and AI-powered voice assistants to improve the quality of their services.

Voice-based AI assistants in hotel rooms have received growing attention from both researchers and the industry. Buhalis and Moldavska (2021) conducted a qualitative study through 28 semi-structured interviews with hospitality technology providers and hotel guests to examine how in-room voice AI assistants affect hotel services and guest experiences. The study found that voice assistants were used for room environment controls (lighting, temperature), room service ordering, wake-up alarms, and general information queries. Guests reported benefits including convenience, accessibility, and faster service. Buhalis and Moldavska (2022) further published a comprehensive analysis in the Journal of Hospitality and Tourism Technology, confirming that voice assistants can reduce the burden on front desk staff and allow guests to access services without making phone calls or visiting reception.

Amazon's Alexa for Hospitality (now Alexa Smart Properties), launched in 2018, became the most widely deployed commercial voice assistant in the hotel industry. Major hotel chains including Marriott International, Hilton, and Best Western deployed Amazon Echo devices in guest rooms with customised skills for room service ordering, housekeeping requests, and facility information (Marriott International, 2018). A 2025 study published in the Journal of Theoretical and Applied Electronic Commerce Research used structural equation modelling on 529 survey responses and found that voice assistant attributes — including connectivity, information provision, and interactivity — positively influence guest evaluations and online hotel reputation.

In Sri Lanka, the tourism industry is an important part of the national economy. Sri Lanka's tourism sector is projected to generate USD 8 billion in revenue by 2028 (Revista de Gestao, 2024), with the country attracting more international tourists each year who are interested in cultural, wellness, and eco-tourism experiences. A 2022 study published by Springer looked at the digital readiness of Sri Lanka's four major tourism sectors — hoteliers, restaurateurs, airlines, and tour operators — and found that while leading hotel chains have adopted basic digital tools such as online booking systems and mobile applications, the use of advanced AI-powered guest services is still very limited. This shows that there is a significant opportunity to introduce affordable and practical AI solutions in Sri Lanka's hotel industry.

---

## 2.2 Challenges Hotel Businesses Face in Sri Lanka

Sri Lankan hotels face several practical challenges that make it difficult to adopt mainstream commercial voice assistant solutions.

**Internet connectivity dependency and cost** is a major concern for cloud-based systems. While Sri Lanka's connectivity has improved — the launch of Starlink satellite internet in 2023 extended broadband access to rural tourism areas such as hill country properties in Kandy and Nuwara Eliya and coastal resorts in Arugam Bay and Pasikuda — internet connectivity is still a managed cost and not a free utility. Cloud-based voice assistants like Alexa for Hospitality rely entirely on a live connection to Amazon Web Services (AWS) to process speech. If the internet connection is disrupted due to a network outage, bandwidth issue, or service interruption, the system stops working. More importantly, requiring Starlink or dedicated broadband as part of a hotel voice assistant setup adds another recurring cost on top of existing cloud subscription and hardware fees, which makes the overall solution less affordable for small and mid-range Sri Lankan hotels. A fully offline system removes this dependency and guarantees operation regardless of internet availability.

**High cost of commercial solutions** is another barrier. The Alexa for Hospitality model requires proprietary Echo hardware (typically USD 99–200 per device) along with AWS cloud subscription fees that increase with usage. For a mid-range Sri Lankan hotel deploying devices in 50 rooms, the combined hardware and ongoing cloud costs are not financially sustainable. Buhalis and Moldavska (2022) noted this cost problem but did not suggest a more affordable alternative.

**Accent and language mismatch** causes accuracy problems. Commercial voice recognition systems are mainly trained on North American and British English data. Studies have found around 30% accuracy loss when processing non-native English accents, which is a significant problem in Sri Lanka where both guests and hotel staff often speak with a South Asian English accent. This is especially relevant in the Sri Lankan hotel context because India is consistently Sri Lanka's largest source of international tourists — accounting for 416,974 arrivals in 2024, approximately 20% of all international visitors (Sri Lanka Tourism Development Authority, 2024). The majority of these guests speak Indian English, making South Asian accent robustness a direct operational requirement rather than a theoretical concern.

**Limited IT expertise** makes it harder for Sri Lankan hotels to adopt complex technology. The 2022 Springer study found that most Sri Lankan hotel operators do not have in-house technical staff who can set up, maintain, or fix cloud-connected AI systems. This means they have to depend on external vendors, which adds both cost and delay.

**Operational inefficiency** in guest services is a common problem. Without voice technology, guests have to call the front desk for all requests — room service, housekeeping, maintenance, and general queries. This puts a lot of pressure on front desk staff, especially during busy periods, and increases the chance of miscommunication. Hwang and Erdem (2025) found that voice assistants can reduce the number of repetitive calls to front desk staff, but noted that the system needs to be well-designed and properly implemented for these improvements to actually happen.

**Privacy risks in cloud-connected deployments** are also a concern. Research at Purdue University (2018) identified specific security vulnerabilities in hotel Alexa deployments, including the risk that third-party skills could listen to guest conversations through the Alexa Skills Kit. Hotel guests may unknowingly share sensitive information — such as credit card details, travel plans, or personal conversations — with cloud-connected devices in their rooms. This creates a risk for hotels, particularly those that serve European guests who have GDPR data protection rights.

---

## 2.3 Impact of the Challenges

The challenges described in Section 2.2 have direct negative effects on guest experience, hotel operations, and the hotel's ability to compete in the market.

**Guest experience** is affected when service delivery is slow. Research has consistently shown that how quickly a hotel responds to guest requests is one of the most important factors in guest satisfaction (Buhalis & Moldavska, 2022). When guests struggle to make requests and staff are busy managing phone calls, response times become longer, guests become dissatisfied, and this often leads to negative online reviews. A 2025 study on voice assistant attributes and hotel reputation found that hotels using responsive voice systems received higher guest evaluation scores compared to those relying only on telephone-based request handling.

**Operational efficiency** suffers when all guest requests go through a single telephone line to the front desk, especially during busy times like morning check-outs or evening arrivals. Without any automated routing, a request that should go directly to housekeeping, maintenance, or the kitchen first has to pass through the front desk, causing delays and increasing the chance of errors. There is also no way for managers to see how quickly requests are being handled or to identify recurring service problems.

**Market competitiveness** is also affected. Global hotel booking platforms are starting to list AI amenities and smart room features as searchable options. Hotels that do not offer technology-enhanced services may find it harder to attract international guests, who tend to be higher-spending visitors and are more likely to expect modern in-room technology.

**Privacy and regulatory risk** is growing. As data protection laws expand across Asia and more European tourists visit Sri Lanka, hotels that use cloud-based systems which send guest voice data to overseas servers face increasing legal risk, even if no actual data breach takes place.

---

## 2.4 Solutions Proposed in Existing Research and Industry

Researchers and technology companies have proposed various solutions to address the challenges of voice-driven hotel service delivery. These can be grouped into three categories: commercial cloud platforms, offline speech recognition systems, and compact NLU models for on-device processing.

### 2.4.1 Commercial Cloud-Based Voice Assistants

Amazon's Alexa for Hospitality is the most widely used commercial solution for hotel voice assistance. The platform allows hotels to manage devices at a property level, create custom voice skills, and includes some privacy features such as a microphone disconnect button. Hwang and Erdem (2025) studied how this system affects the work of hotel staff and found it can reduce repetitive tasks and improve response times. The 2025 JTAECR study also confirmed that voice assistant features such as interactivity and information provision have a positive effect on hotel reputation through improved guest evaluations.

### 2.4.2 Edge-Based and Offline Speech Recognition

Researchers have developed offline speech recognition alternatives to address the limitations of cloud-based systems. Vosk, developed by Alpha Cephei, is an open-source offline speech recognition toolkit that supports over 20 languages with model sizes ranging from 50MB to 1.8GB. Korkmaz et al. (2025) built a prototype edge-based speech-to-text system using Vosk and showed that it could provide real-time, low-latency speech processing on constrained hardware such as a Raspberry Pi, without any cloud connectivity. Chantrapornchai and Suchato (2022) also explored offline speech recognition on Raspberry Pi for IoT control applications and demonstrated that edge-based speech recognition is practical for simple command-style tasks.

A 2024 benchmarking study compared multiple speech recognition frameworks — including Vosk, Whisper, and several commercial tools — across accuracy, latency, and resource usage. Vosk was found to offer the best balance of accuracy and resource efficiency for real-time edge deployment. Radford et al. (2023) introduced Whisper, a large model trained on 680,000 hours of multilingual audio data that achieves high accuracy and handles different accents and background noise well. However, Whisper's high computational requirements — especially for the medium and large model sizes — make it unsuitable for running directly on commodity mobile hardware.

### 2.4.3 Compact NLU Models for On-Device Intent Classification

BERT (Devlin et al., 2019) was a major advancement in natural language understanding by using bidirectional pre-training. However, BERT-BASE has 110 million parameters and is approximately 440MB in size, which is too large for mobile deployment. Several smaller models have been developed to address this:

Sanh et al. (2019) introduced DistilBERT, which keeps 97% of BERT's language understanding ability while being 40% smaller and 60% faster through a process called knowledge distillation. Sun et al. (2020) proposed MobileBERT, a compact version of BERT specifically designed for mobile devices. MobileBERT achieves 4.3× model compression and 5.5× faster inference compared to BERT-BASE, while scoring only 0.6 points lower on the GLUE benchmark. On a Google Pixel 4, MobileBERT runs in 62 milliseconds per inference.

For server-side NLU, Bunk et al. (2020) introduced DIET (Dual Intent and Entity Transformer) in the Rasa framework, which handles both intent classification and entity recognition in a single model. It outperforms fine-tuned BERT on several benchmarks while training six times faster. Bujel et al. (2021) showed that fine-tuning BERT on as few as 1,000 domain-specific examples can produce better results than general pre-trained models for intent classification in specialised domains.

### 2.4.4 Privacy-Preserving Edge Computing

Shi et al. (2016) established the concept of edge computing in their widely-cited IEEE IoT Journal paper, showing how processing data locally, close to its source, can reduce latency, improve privacy, and decrease bandwidth usage. This approach means that sensitive voice data does not need to be sent to external cloud servers. A 2025 study on privacy-preserving on-device speech recognition using Vosk with custom domain language models demonstrated that offline models can reach close to cloud-level accuracy for domain-specific tasks, while ensuring that voice data never leaves the user's device — something cloud systems cannot guarantee regardless of their privacy policies.

---

## 2.5 How Existing Solutions Have Helped Address the Challenges

Existing solutions have made progress in addressing individual parts of the problem, and there is good evidence from the literature to support their effectiveness.

**Cloud-based commercial voice assistants** have clearly shown that hotel guests find value in voice interaction. Buhalis and Moldavska (2021) found that the benefits of hotel voice assistants consistently outweigh the drawbacks for both guests and hotel operators. Hwang and Erdem (2025) reported concrete improvements: fewer repetitive calls to front desk staff, faster information delivery, and a more modern perception of the hotel's service. The 2025 JTAECR study also confirmed that interactive voice assistants positively affect hotel reputation scores.

**Edge-based speech recognition** research has shown that offline STT is technically possible on standard hardware. Korkmaz et al. (2025) demonstrated that Vosk-based edge STT can match cloud solutions in terms of response speed for short command-style inputs. Custom domain language models for Vosk have been shown to reduce Word Error Rate by up to 40% compared to generic offline models (2025), which substantially closes the accuracy gap with cloud-based alternatives. Chantrapornchai and Suchato (2022) confirmed that Raspberry Pi devices can handle real-time offline speech recognition for command recognition tasks.

**Compact NLU models** have shown that on-device natural language understanding is feasible. MobileBERT's 62ms inference latency on a Google Pixel 4 (Sun et al., 2020) shows that transformer-based intent classification is practical for real-time use on mobile devices. DistilBERT (Sanh et al., 2019) demonstrated that knowledge distillation can produce smaller models without a large drop in accuracy. The AUTODIAL framework (2025) achieved 3–6× faster inference with 11× fewer parameters using a parallel decoder architecture for dialogue state tracking and intent prediction.

**Privacy-preserving design** has also been technically validated. The 2025 on-device Vosk study showed that fully local processing can reach competitive accuracy while eliminating the data exposure risk that comes with cloud-based systems.

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
| Custom Vosk LM (2025) | Yes | Yes | No | Yes | Yes | No | No |
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

### 2.6.2 Model Performance and Empirical Contribution

This research makes an empirical contribution by building, training, and evaluating three MobileBERT models on a custom hospitality intent classification dataset. The clean training dataset contains **10,080 labelled utterances** covering **18 hospitality-specific intent categories** such as room service ordering, housekeeping requests, maintenance reporting, concierge queries, and wake-up calls. To address Gap 6, an additional **10,080 Vosk-transcribed versions** of the same utterances were generated, creating a paired dataset. Three models were trained:

- **Model A** — trained on clean text only (baseline)
- **Model B** — trained on Vosk-transcribed text only
- **Model C** — trained on a mixed clean+Vosk dataset (14,863 sentences)

All three models were converted to TensorFlow Lite format, producing a **25.1MB on-device model**, and evaluated on the same held-out test set of 2,016 Vosk-transcribed utterances.

**Table 2.3: Three-Model Evaluation Results (n = 2,016 test samples)**

| Model | Training Data | Test Input | Accuracy | F1 Macro |
|-------|--------------|------------|----------|----------|
| A — baseline | Clean text | Clean text | 98.07% | 0.9805 |
| A — baseline | Clean text | Vosk output | 89.34% | 0.8908 |
| B | Vosk only | Vosk output | 96.38% | 0.9636 |
| **C — deployed** | **Clean + Vosk mixed** | **Vosk output** | **99.06%** | **0.9905** |

The main finding of this research is the **8.73 percentage point accuracy drop** when Model A is tested on Vosk-transcribed text (89.34%) compared to clean text (98.07%). This drop is caused by transcription noise from the Vosk STT engine and would not be seen if the model were only evaluated on clean text — as is common in standard NLU benchmarks. This means that reported NLU accuracy in research is often higher than what would actually be observed in a real offline deployment.

Noise-aware training (Model C) recovers **+9.72 percentage points**, achieving 99.06% accuracy on Vosk output — a **111.3% gap recovery**. This shows that the accuracy drop caused by STT noise is not a fundamental limitation of compact transformer models. It is caused by a mismatch between training data and real deployment conditions, and can be effectively fixed by including Vosk-transcribed text in the training data.

These results represent the first documented measurement of Vosk STT noise impact on hospitality NLU, and the first demonstration that noise-aware MobileBERT training can close this gap in a real offline deployment.

---

## 2.7 Technology Tools and Their Suitability

Each technology used in this research was selected based on specific performance, cost, and deployment requirements that come from the challenges identified in Sections 2.2 and 2.3.

### 2.7.1 Vosk for On-Device Speech-to-Text

Vosk (Alpha Cephei) was chosen as the speech recognition engine for the Android application for several reasons. First, it works fully offline, which means it does not require an internet connection for any of its processing — this is an important requirement given that the system should work without depending on internet availability (Section 2.2). Second, the `vosk-model-small-en-in-0.4` model variant (Indian English, approximately 50MB) is small enough to be included on a commodity Android tablet and is better suited to the South Asian English accent spoken by both guests and hotel staff in Sri Lanka compared to US or British English model variants. Third, Korkmaz et al. (2025) showed that Vosk-based edge STT can achieve latency similar to cloud solutions for short command-style utterances. Fourth, a 2024 benchmarking study found that Vosk offers the best balance of accuracy and resource efficiency among offline ASR frameworks. Custom domain language models for Vosk have also been shown to reduce Word Error Rate by up to 40% (2025), which provides a clear path for further improvement as the dataset grows.

### 2.7.2 MobileBERT and TensorFlow Lite for On-Device NLU

MobileBERT (Sun et al., 2020) was chosen as the intent classification model instead of DistilBERT or Rasa DIET. DistilBERT is not specifically designed for mobile inference and has not been tested with TFLite on Android. Rasa DIET needs to run as a server process, which does not fit the requirement for on-device processing. MobileBERT was designed specifically for resource-limited mobile devices and achieves 62ms inference latency on a Pixel 4, which is fast enough for real-time interaction. Converting the model to TFLite produces a **25.1MB model** that can be stored and run on a standard Android tablet.

More importantly, the training approach used in this research directly addresses Gap 1 (Section 2.6.1). Instead of only using clean text for training — which is the standard approach — Model C was trained on a mixed dataset of 14,863 clean and Vosk-transcribed utterances. As shown in Section 2.6.2, this noise-aware training achieves **99.06% accuracy** and **0.9905 F1 macro** on Vosk-transcribed test inputs, recovering 111.3% of the STT-induced accuracy gap. This shows that including realistic STT output in the training data leads to better real-world performance than training on clean text alone, which has practical implications for any offline NLU system that uses an imperfect STT engine.

### 2.7.3 Android Tablets for Hardware Deployment

Android tablets were chosen as the guest-room hardware instead of specialised devices such as Amazon Echo or Raspberry Pi. There are three main reasons for this. First, **cost**: Android tablets are available in Sri Lanka for approximately USD 50–150, which is much lower than the USD 200+ cost of smart speakers, and avoids the additional complexity and custom housing costs of Raspberry Pi setups. Second, **availability**: local technicians in Sri Lanka can purchase, repair, and replace Android tablets without specialist knowledge or needing to import parts from overseas. Third, **platform support**: Android has a well-established development environment for TFLite inference, Vosk integration, and WebSocket communication, which reduces development effort and makes the system easier to maintain. The 2022 Springer study on Sri Lankan digital tourism found that locally available and easily serviceable hardware is a key requirement for technology adoption in this market.

### 2.7.4 FastAPI and WebSocket for Backend Communication

A Python FastAPI server with WebSocket support was chosen for the hotel backend. FastAPI handles both HTTP and WebSocket communication efficiently with low overhead, making it suitable for running on a standard laptop or server within the hotel's local network. WebSockets allow real-time, two-way communication between guest devices and the staff dashboard — a feature that is not provided by existing solutions including Alexa for Hospitality. This directly addresses Gap 3 identified in Section 2.6.1. Running the backend on hotel hardware rather than in the cloud means there are no recurring hosting fees and all guest data stays within the hotel premises, which is consistent with the edge computing approach to data privacy described by Shi et al. (2016).

### 2.7.5 Summary of Tool Suitability

Table 2.2 shows how each selected tool addresses the challenges identified in Section 2.2.

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

This chapter reviewed the existing literature and identified the research gaps that this study addresses. Section 2.1 provided background on the hotel industry and the growing use of voice AI in hospitality, including the situation in Sri Lanka. Section 2.2 described the main challenges that prevent Sri Lankan hotels from adopting mainstream voice assistant solutions, including internet dependency, high cost, accent mismatch, limited IT expertise, and privacy risks. Section 2.3 explained how these challenges affect guest experience, operational efficiency, and market competitiveness.

Sections 2.4 and 2.5 reviewed existing solutions from both research and industry. These include commercial platforms like Alexa for Hospitality, offline STT research using Vosk, and compact NLU models like MobileBERT and DistilBERT. While each of these solutions addresses part of the problem, none provides a complete, integrated, offline system for hospitality.

Section 2.6 identified four research gaps and presented the empirical results of this study. The most important finding (Gap 1) is an 8.73 percentage point accuracy drop when a standard NLU model trained on clean text is tested on real Vosk output — the first documented measurement of this gap in the hospitality domain. Noise-aware training (Model C) recovers +9.72 percentage points, achieving 99.06% accuracy on Vosk output, which represents a 111.3% gap recovery. Section 2.7 explained why each technology was selected for this research, including the choice of the Indian English Vosk model which is better suited to the South Asian English accent of the target users.

The following chapter presents the methodology used to design, build, and evaluate the complete system.
