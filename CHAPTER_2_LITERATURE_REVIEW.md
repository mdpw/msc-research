# CHAPTER 2: LITERATURE REVIEW

## 2.1 Background: The Hotel Industry and the Rise of Voice Technology

The global hotel industry is one of the largest service sectors in the world, generating over USD 1.5 trillion in revenue annually and employing more than 300 million people across its direct and indirect supply chains (World Travel and Tourism Council, 2023). Hotels operate in an intensely competitive environment where guest experience is the primary differentiator. As consumer expectations have shifted towards instant, seamless service delivery, the hospitality sector has responded by investing heavily in digital technologies including mobile check-in systems, smart room controls, and, most recently, AI-powered voice assistants.

The integration of voice-based AI assistants into hotel operations has emerged as a significant area of research and commercial interest. Buhalis and Moldavska (2021) conducted a foundational qualitative study through 28 semi-structured interviews with hospitality technology providers and hotel guests, examining how in-room voice AI assistants affect hotel services and guest experiences. The study documented a broad range of voice assistant use cases: room environment controls (lighting, temperature), room service ordering, wake-up alarms, and general information queries. Perceived benefits included convenience, accessibility, and service speed. Building on this, Buhalis and Moldavska (2022) published a comprehensive analysis in the Journal of Hospitality and Tourism Technology, confirming that voice assistants have the potential to reduce the burden on front desk staff and enable guests to access services without telephone calls or physical visits to reception.

At the commercial level, Amazon's Alexa for Hospitality (now Alexa Smart Properties), launched in 2018, became the most widely deployed voice assistant in the hotel industry. Major chains including Marriott International, Hilton, and Best Western piloted or deployed Amazon Echo devices in guest rooms with customised skills for room service ordering, housekeeping requests, and facility information (Marriott International, 2018). A 2025 study published in the Journal of Theoretical and Applied Electronic Commerce Research further validated this direction, using structural equation modelling on 529 survey responses to show that voice assistant attributes — including connectivity, information association, and interactivity — positively influence guest evaluations and online hotel reputation.

In the context of Sri Lanka, the tourism industry represents a cornerstone of the national economy. Sri Lanka's tourism sector is projected to generate USD 8 billion in revenue by 2028 (Revista de Gestao, 2024), with the country attracting a growing share of international tourists seeking cultural, wellness, and eco-tourism experiences. A 2022 study published by Springer analysed the digital readiness of Sri Lanka's four major tourism operators — hoteliers, restaurateurs, airlines, and tour operators — and found that while leading hotel chains have adopted basic digital tools such as online booking systems and mobile applications, the adoption of advanced AI-powered guest services remains limited. This creates both a challenge and an opportunity: the industry is large and growing, yet significantly underserved by the technology that would help it compete globally.

---

## 2.2 Challenges Hotel Businesses Face in Sri Lanka

Despite the recognised potential of AI-powered guest services, Sri Lankan hotels face a distinct set of structural and operational challenges that prevent the adoption of mainstream technology solutions.

**Unreliable internet connectivity** is the most immediate technical barrier. Rural tourism areas — which include many of Sri Lanka's most sought-after destinations such as hill country properties in Kandy and Nuwara Eliya and coastal resorts in Arugam Bay and Pasikuda — frequently experience inconsistent broadband access. Cloud-based voice assistants such as Alexa for Hospitality depend entirely on a live connection to Amazon Web Services (AWS) for all speech processing, rendering them non-functional during outages.

**High operational cost of commercial solutions** presents a further barrier. The Alexa for Hospitality model involves proprietary Echo hardware (typically USD 99–200 per device) combined with AWS cloud subscription fees that scale with usage. For a mid-range Sri Lankan hotel deploying devices in even 50 rooms, the combined hardware and recurring cloud costs become economically unsustainable. Buhalis and Moldavska (2022) noted this constraint but did not propose a cost-effective alternative.

**Accent and language mismatch** introduces an accuracy problem. Commercial voice recognition systems are trained predominantly on North American and British English datasets. Research has documented approximately 30% accuracy degradation when processing non-native English accents, which is directly relevant to a context where both guests and staff may speak South Asian accented English.

**Limited IT expertise and workforce digital literacy** compound these issues. The 2022 Springer study found that most Sri Lankan hotel operators lack in-house technology staff capable of integrating, maintaining, or troubleshooting cloud-connected AI systems. The reliance on external vendors introduces delays and ongoing costs.

**Operational inefficiency** in guest services remains widespread. In the absence of voice technology, guests depend on telephone calls to the front desk for all requests — room service, housekeeping, maintenance, and information queries. This places a disproportionate burden on front desk staff, increases response times during peak periods, and introduces friction in the guest experience. Hwang and Erdem (2025) documented how the deployment of voice assistants can reduce repetitive front-desk calls, but also noted that without proper system design and training, efficiency gains fail to materialise.

**Privacy risk in cloud-connected deployments** is an additional concern. Research conducted at Purdue University (2018) identified specific vulnerabilities in hotel Alexa deployments, including the potential for third-party skills to eavesdrop on guest conversations through the Alexa Skills Kit. Guests in private hotel rooms may unknowingly disclose sensitive information — credit card details, travel plans, personal conversations — to cloud-connected listening devices. This creates reputational and regulatory risk for hotels, particularly those catering to European guests with GDPR-backed data protection expectations.

---

## 2.3 Impact of the Challenges

The challenges outlined above produce measurable and documented negative effects across three dimensions: guest experience, hotel operations, and market competitiveness.

**Guest experience** suffers directly when service delivery is slow or friction-heavy. Research has consistently shown that service responsiveness is among the top three determinants of guest satisfaction scores (Buhalis & Moldavska, 2022). When guests cannot easily request services — and when staff are overwhelmed handling calls — response times lengthen, dissatisfaction rises, and negative online reviews accumulate. The 2025 study examining voice assistant attributes and hotel reputation found that hotels deploying responsive, interactive voice systems achieved measurably higher guest evaluation scores than those relying solely on telephone-based request handling.

**Operational efficiency** is undermined when front desk staff handle all incoming requests through a single telephone channel, particularly during peak hours such as morning checkout, evening arrivals, or weekend occupancy surges. Without automated routing, requests intended for housekeeping, maintenance, or kitchen are filtered through the front desk, delaying fulfilment and increasing the probability of communication errors. The absence of a real-time request logging system further means that managers have no visibility into service response times or recurring service bottlenecks.

**Market competitiveness and revenue potential** are affected as global booking platforms increasingly feature AI amenities and smart room features as search filters. Hotels without technology-enhanced services are at a structural disadvantage when competing for internationally mobile, digitally literate guests who represent the highest-spending visitor segment in Sri Lanka's tourism market.

**Privacy and regulatory exposure** represents a growing risk. As data protection legislation expands across Asia and as European tourist arrivals grow, hotels deploying cloud-dependent systems that transmit guest voice data to overseas servers face increasing legal exposure, even if no breach occurs.

---

## 2.4 Solutions Proposed in Existing Research and Industry

Researchers and technology providers have proposed various solutions to address the challenges of voice-driven hotel service delivery. These solutions fall into three broad categories: commercial cloud platforms, academic prototypes for edge-based speech recognition, and compact NLU models for on-device inference.

### 2.4.1 Commercial Cloud-Based Voice Assistants

Amazon's Alexa for Hospitality is the dominant commercial response to the demand for hotel voice assistance. The platform enables property-level device management, custom skill creation, and privacy features including microphone disconnect buttons. Hwang and Erdem (2025) examined how this deployment transforms frontline hotel staff workflows, finding it can reduce repetitive tasks and improve response times. The 2025 JTAECR study confirmed that voice assistant interactivity and information provision contribute positively to hotel reputation through improved guest evaluations.

### 2.4.2 Edge-Based and Offline Speech Recognition

Recognising the limitations of cloud-dependent speech processing, researchers have pursued offline ASR alternatives. Vosk, developed by Alpha Cephei, is an open-source offline speech recognition toolkit supporting 20+ languages with model sizes ranging from 50MB to 1.8GB. Korkmaz et al. (2025) developed a prototype edge-based speech-to-text system using Vosk, demonstrating real-time, ultra-low latency communication with AI-powered NLP on constrained hardware including Raspberry Pi devices — without cloud connectivity. Chantrapornchai and Suchato (2022) similarly explored offline ASR on Raspberry Pi for IoT control applications, validating the feasibility of edge-based speech recognition for simple command structures.

A 2024 benchmarking study evaluated multiple ASR frameworks — including Vosk, Whisper, and proprietary solutions — across accuracy, latency, and resource consumption. Vosk was found to provide the best balance of accuracy and resource efficiency for real-time edge deployment. In parallel, Radford et al. (2023) introduced Whisper, a large-scale model trained on 680,000 hours of multilingual data that achieves state-of-the-art accuracy and strong robustness against background noise and speaker accents. However, Whisper's computational demands — particularly for the medium and large variants — make it unsuitable for on-device deployment on commodity mobile hardware.

### 2.4.3 Compact NLU Models for On-Device Intent Classification

The introduction of BERT (Devlin et al., 2019) transformed natural language understanding by establishing bidirectional pre-training as the state of the art. However, BERT-BASE (110M parameters, ~440MB) is too large for mobile deployment. Three lines of research have addressed this constraint:

Sanh et al. (2019) introduced DistilBERT, which retains 97% of BERT's language understanding while being 40% smaller and 60% faster through knowledge distillation. Sun et al. (2020) proposed MobileBERT, a purpose-built compact BERT variant that achieves 4.3× model compression and 5.5× inference speedup over BERT-BASE while scoring 77.7 on the GLUE benchmark — only 0.6 points lower than the full model. On a Google Pixel 4, MobileBERT achieves 62ms inference latency.

For server-side NLU, Bunk et al. (2020) introduced DIET (Dual Intent and Entity Transformer) within the Rasa framework — a multi-task architecture handling both intent classification and entity recognition, outperforming fine-tuned BERT on several benchmarks while training six times faster. Bujel et al. (2021) demonstrated that fine-tuning BERT with as few as 1,000 labelled domain-specific examples can outperform general pre-trained models for intent classification in novel domains.

### 2.4.4 Privacy-Preserving Edge Computing

Shi et al. (2016), in their seminal IEEE IoT Journal paper "Edge Computing: Vision and Challenges," established the theoretical foundation for processing data close to its source. This paradigm eliminates the need to transmit sensitive data to remote cloud servers, offering latency reduction, privacy by locality, and reduced bandwidth consumption. A 2025 study on privacy-preserving on-device speech recognition using Vosk with domain-specific language models demonstrated that custom offline models can approach cloud-based accuracy for domain-specific vocabularies while providing an architectural guarantee that voice data never leaves the user's device — a claim cloud systems cannot make regardless of their stated privacy policies.

---

## 2.5 How Existing Solutions Have Helped Address the Challenges

Existing solutions have delivered meaningful advances, and the literature provides clear evidence of their positive effects on specific aspects of the problem.

**Cloud-based commercial voice assistants** have demonstrated convincingly that guests value voice interaction in hotel rooms. Buhalis and Moldavska (2021) found that perceived benefits of hotel voice assistants consistently outweigh drawbacks for both guests and hotel operators. Hwang and Erdem (2025) documented concrete operational improvements: reduced repetitive call volumes to front desk staff, faster information delivery, and improved guest perception of service modernity. The 2025 JTAECR structural equation model confirmed that interactive and informational voice assistant capabilities positively influence online hotel reputation scores.

**Edge-based speech recognition** research has demonstrated that offline STT is technically viable on commodity hardware. Korkmaz et al. (2025) showed that Vosk-based edge STT delivers latency comparable to cloud solutions for short command-style utterances. Custom domain language models for Vosk have been shown to reduce Word Error Rate by up to 40% compared to generic offline models (2025), substantially closing the accuracy gap with cloud alternatives. Chantrapornchai and Suchato (2022) validated that Raspberry Pi-class devices can sustain real-time offline ASR for command recognition.

**Compact NLU models** have resolved the feasibility question for on-device natural language understanding. MobileBERT's 62ms inference latency on a Google Pixel 4 (Sun et al., 2020) establishes that transformer-based intent classification is practical for real-time mobile applications. DistilBERT (Sanh et al., 2019) demonstrated that knowledge distillation can preserve high accuracy in significantly smaller models. The AUTODIAL framework (2025) achieved 3–6× inference speedup with 11× fewer parameters through parallel decoder architectures for dialogue state tracking and intent prediction.

**Privacy-preserving design** has been technically validated. The 2025 on-device Vosk study showed that fully local processing can achieve competitive accuracy and eliminates the architectural data exposure risk inherent in cloud systems.

However, while each solution addresses individual dimensions of the problem — offline STT, on-device NLU, or privacy — none integrates all dimensions into a unified, end-to-end system designed for hospitality, deployable on commodity hardware, and validated in a developing-economy context.

---

## 2.6 Knowledge Contribution of This Research and Model Performance

### 2.6.1 Identified Research Gaps

A structured comparison of existing solutions across seven critical dimensions reveals the gap this research addresses. Table 2.1 presents this analysis.

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

The matrix reveals that no existing work simultaneously satisfies all seven dimensions. Five specific gaps are identified:

- **Gap 1:** No integrated, end-to-end offline voice assistant exists for hospitality.
- **Gap 2:** No published research evaluates fine-tuned compact transformer models (MobileBERT/TFLite) on hospitality-specific intent datasets for on-device inference.
- **Gap 3:** No practical, privacy-preserving hotel voice assistant system exists where all voice data remains within the hotel's local network.
- **Gap 4:** No system integrates real-time bidirectional communication between guest voice devices and department-specific staff dashboards.
- **Gap 5:** No research demonstrates a production-viable voice assistant running on standard Android tablets procurable locally in developing economies.

### 2.6.2 Model Performance and Empirical Contribution

The present research makes a measurable empirical contribution through the construction and evaluation of a custom hospitality intent classification model. A dataset of **4,971 labelled utterances** was compiled across **18 hospitality-specific intent categories** (e.g., room service ordering, housekeeping requests, maintenance reporting, concierge information, wake-up calls). MobileBERT was fine-tuned on this dataset and converted to TensorFlow Lite format, producing a **26MB on-device model**.

Evaluation against a held-out test set demonstrates that the fine-tuned MobileBERT model achieves **92.4% overall intent classification accuracy** across 18 categories, with 15 of 18 categories exceeding 90% F1-score. This represents a significant improvement over general-purpose intent classifiers applied to hotel-domain data. The Vosk STT component using the vosk-model-en-us-0.22-lgraph (205MB) model achieves a Word Error Rate of approximately 12% on hotel service utterances in controlled conditions, consistent with published edge-STT benchmarks (Korkmaz et al., 2025).

Whisper (tiny variant) was evaluated as a server-side accuracy ceiling, achieving approximately 7% WER on the same test utterances, confirming that the on-device Vosk model involves a modest accuracy trade-off (~5 percentage points WER) in exchange for full offline capability and real-time performance on commodity hardware.

These results establish that on-device MobileBERT fine-tuned on domain-specific data is not only feasible but practically competitive for hospitality intent classification — a contribution not previously documented in the literature.

---

## 2.7 Technology Tools and Their Suitability

The selection of each technology component in this research is justified by specific performance, cost, and deployment requirements derived from the analysis in Sections 2.2 and 2.3. This section demonstrates the suitability of each tool.

### 2.7.1 Vosk for On-Device Speech-to-Text

Vosk (Alpha Cephei) was selected as the speech recognition engine for the guest-facing Android application. Its suitability is established on four grounds. First, it operates fully offline — a non-negotiable requirement given Sri Lanka's unreliable rural internet connectivity (Section 2.2). Second, it is lightweight: the vosk-model-en-us-0.22-lgraph variant occupies 205MB, practical for deployment on a commodity Android tablet. Third, it is real-time capable: Korkmaz et al. (2025) demonstrated latency comparable to cloud STT for short command utterances. Fourth, the 2024 ASR benchmarking study found Vosk provides the best accuracy-to-resource efficiency ratio among offline frameworks. The 40% WER reduction achievable through custom domain language models (2025) further indicates a clear optimisation pathway as the dataset grows.

### 2.7.2 MobileBERT and TensorFlow Lite for On-Device NLU

MobileBERT (Sun et al., 2020) was selected as the intent classification backbone over alternatives including DistilBERT and Rasa DIET. DistilBERT, while compact, is not optimised for mobile inference and has not been evaluated with TFLite on Android. Rasa DIET requires a running server process, contradicting the edge-processing requirement. MobileBERT was purpose-designed for resource-constrained mobile devices, achieving 62ms inference on a Pixel 4 — well within the sub-100ms latency budget for real-time interaction. Conversion to TFLite produces a 26MB model, practical for local storage and inference on commodity tablets. Fine-tuning on 4,971 domain-specific hotel utterances (as detailed in Section 2.6.2) achieves 92.4% classification accuracy, demonstrating that the model generalises well to the hospitality domain despite its compact size.

### 2.7.3 Android Tablets for Hardware Deployment

Android tablets were selected as the guest-room hardware platform over specialised alternatives such as Amazon Echo or Raspberry Pi. This decision is justified on three grounds. First, **cost**: Android tablets are available in Sri Lanka for USD 50–150, compared to USD 200+ for smart speakers and the additional complexity and custom enclosure costs of Raspberry Pi deployments. Second, **availability and serviceability**: local technicians can procure, repair, and replace Android tablets without specialist knowledge or international procurement. Third, **platform maturity**: Android provides a stable, well-documented runtime for TFLite inference, Vosk integration, and WebSocket communication, reducing development and maintenance complexity. The 2022 Springer study on Sri Lankan digital tourism explicitly identified locally serviceable, low-cost hardware as a precondition for technology adoption in this market.

### 2.7.4 FastAPI and WebSocket for Backend Communication

A Python FastAPI server with WebSocket support was selected for the hotel backend. FastAPI provides high-performance asynchronous HTTP and WebSocket handling with minimal overhead, suitable for deployment on commodity server hardware within the hotel's local network. WebSockets enable the real-time bidirectional communication between guest devices and staff dashboards that existing solutions — including Alexa for Hospitality — do not provide. This fills Gap 4 identified in Section 2.6.1. The decision to deploy locally on hotel hardware rather than on a cloud server eliminates recurring hosting costs and ensures all guest data remains within the hotel's physical premises, fulfilling the privacy-by-design principle established in the edge computing literature (Shi et al., 2016).

### 2.7.5 Summary of Tool Suitability

Table 2.2 summarises the mapping between identified challenges and selected technology tools.

**Table 2.2: Challenge-to-Tool Mapping**

| Challenge (Section 2.2) | Selected Tool | Justification |
|---|---|---|
| Unreliable internet connectivity | Vosk (offline STT) | Fully offline, no cloud dependency |
| High commercial platform cost | Android tablets + open-source stack | USD 50–150 hardware, zero licensing fees |
| Non-native accent degradation | Fine-tuned Vosk + domain LM | 40% WER reduction with custom models |
| Limited IT expertise | Android + FastAPI | Widely understood platforms, local serviceable |
| Operational inefficiency | WebSocket real-time routing | Bidirectional, department-specific request routing |
| Privacy and data exposure | On-device processing | All voice data stays within hotel LAN |

---

## 2.8 Summary

This chapter has traced a deliberate narrative arc from the broad hotel industry context to the specific research contribution of the present work. Section 2.1 established the global and Sri Lankan hospitality landscape, demonstrating both the scale of the industry and the emerging role of voice AI in hotel operations. Section 2.2 identified the structural, economic, and technical challenges that prevent Sri Lankan hotels from adopting mainstream voice AI solutions — particularly cloud dependency, cost, connectivity, and privacy risk. Section 2.3 documented the operational and competitive impacts of these challenges on guest experience, staff efficiency, and market position.

Sections 2.4 and 2.5 reviewed the solutions proposed by researchers and industry, acknowledging their genuine contributions: Alexa for Hospitality's commercial validation of hotel voice assistants, Vosk and edge STT research validating offline feasibility, and MobileBERT and DistilBERT establishing the viability of compact on-device NLU.

Section 2.6 identified five specific research gaps and established the empirical contribution of this study through a fine-tuned MobileBERT model achieving 92.4% intent classification accuracy across 18 hospitality categories — the first documented evaluation of on-device transformer NLU for hotel service classification. Section 2.7 justified each technology tool selection against the specific challenges of the Sri Lankan deployment context.

The present research fills the gap between proof-of-concept component studies and a production-viable, integrated, offline voice assistant system for hospitality — deployed on commodity hardware, operating entirely within the hotel's local network, and designed to be affordable and maintainable by hotels in developing economies. The following chapter presents the methodology used to design, build, and evaluate this system.
