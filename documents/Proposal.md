# Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models

## Introduction

Sri Lanka's hospitality industry is a major part of the national economy, and hotels are always looking for ways to improve the guest experience while keeping costs down. The most common way guests request room services today, calling the front desk or filling in a paper form is slow and unreliable. Requests get delayed, staff miss things, and there is no easy way to track what was asked for or when.

Systems like Amazon Alexa for Hospitality and Google Nest Hub have tried to solve this with voice assistants. But these products depend on cloud servers, need a reliable internet connection, and send guest voice data to third-party servers. That raises privacy concerns. They also cost a lot to run hardware, licensing, and API fees add up quickly. For most hotels in Sri Lanka, that kind of investment simply is not practical.

This research builds a voice assistant designed specifically for hotel room service. It runs entirely on a standard commodity mobile device which provides speech-to-text (STT), intent classification, and text-to-speech (TTS) capabilities operating entirely on-device within a local network. No internet is needed. No guest data leaves the building. Guests speak their request, the device understands it, routes it to the right department, and staff can see and manage all requests on a web dashboard in real time.

## Problem Statement

Hotels in developing economies like Sri Lanka face a specific problem: the commercial voice assistant options are either too expensive or require cloud access that is not always reliable. Beyond cost, there is a real privacy issue guests are increasingly uncomfortable knowing their voice is being sent to a remote server.

Phone-based room service has its own problems too. There are language barriers, long wait times, and no way to systematically track whether requests are being handled. The whole process depends on staff availability at any given moment.

## Research Question

Can a low-cost, fully offline voice assistant system, built on a standard commodity mobile device, achieve sufficient technical accuracy and performance to be a viable alternative to traditional room service communication in Sri Lankan hotels?

The aim of this project is to design, build, and evaluate a system that runs inside a hotel's local network with no cloud dependency at all.

## Objectives

1. Design and build a fully offline, low-cost voice assistant prototype covering on-device speech recognition, intent classification, and the complete request lifecycle (submission, confirmation, cancellation, and rating) deployable on commodity mobile hardware without any cloud dependency.
2. Create a hotel-specific intent classification dataset covering the most common room service request categories and train an intent classification model that can handle the accuracy challenges introduced by on-device speech recognition.
3. Build a lightweight backend with automated department routing and a web-based staff dashboard, supporting real-time bidirectional communication between guests and hotel staff.
4. Evaluate the system's intent classification accuracy, end-to-end latency, and cost per request compared to cloud-based alternatives and demonstrate that a fully privacy-preserving, offline voice processing architecture is achievable on low-cost hardware.

## Literature Review

Voice assistants in hotels have been studied quite a bit. Buhalis and Moldavska (2022) found that while guests appreciate these systems, 67% feel uncomfortable with cloud-connected microphones in their rooms. Commercial deployments like Alexa for Hospitality need continuous internet access and ongoing API subscriptions which is a real barrier for smaller hotels in developing countries.

Offline Speech Recognition has come a long way. Vosk (Alpha Cephei, 2020) is a lightweight offline speech recognition library that works well in privacy-sensitive environments. Korkmaz et al. (2025) showed Vosk running on edge hardware with very low latency, making it a practical choice for real-time use. OpenAI's Whisper (Radford et al., 2023) is more accurate, but it needs far more computing power than a budget Android device can provide.

Small-Scale NLU Models make it possible to run natural language understanding on a phone or tablet. MobileBERT (Sun et al., 2020) compresses the original BERT model to 4.3× smaller and 5.5× faster while keeping 96% of its accuracy. It can run on Android via TFLite. DistilBERT (Sanh et al., 2019) is another option 40% smaller, but retaining 97% of BERT's performance. Bujel et al. (2021) showed that fine-tuning these models with roughly 1,000 examples per class is enough to get strong results on domain-specific tasks, which is useful here since a large labelled hospitality dataset does not already exist.

Edge Computing and Privacy are central to this project's design. Shi et al. (2016) established that processing data at the network edge rather than sending it to the cloud cuts both latency and privacy risk. Purdue University (2018) documented specific privacy vulnerabilities in hotel Alexa deployments. More recent work (2025) showed that a custom Vosk model can achieve a 40% Word Error Rate (WER) reduction over cloud STT while keeping all audio local.

Task-Oriented Dialogue research (Larson et al., 2022) shows there is a clear gap in hospitality-specific datasets and on-device NLU pipelines. Rasa's DIET architecture (Bunk et al., 2020) handles intent classification well, but it runs on a server not on a mobile device. No published system combines offline STT, on-device NLU, and a hotel management backend in one integrated product.

**Research Gap:** The individual components offline STT, on-device NLU, hotel management systems all exist, but nobody has combined them into a single, fully offline hospitality voice assistant. More specifically, training NLU models to handle speech recognition output is an established technique for cloud systems, but no published work applies this to offline, on-device deployments in a hospitality context. This research addresses that gap directly.

## Methodology

### 1. Literature Review

A thorough literature review will be conducted covering:

- Voice assistants in hospitality: commercial products and guest experience research
- Offline and edge-based speech recognition systems
- Small-scale NLU models for resource-constrained devices
- Privacy-preserving AI and edge computing architectures
- Task-oriented dialogue systems and intent classification
- Sri Lankan hospitality context and digital readiness

### 2. Dataset Preparation

- The first step is deciding what the system needs to understand what kinds of requests hotel guests typically make and categorize them, and which department each one belongs to.
- Once those categories are defined, a labelled dataset will be built with enough examples per category to train a reliable intent classifier. The examples will cover different ways of phrasing the same request formal, casual, short, indirect because guests do not all speak the same way. The class distribution will be kept balanced so no single intent dominates the training data.
- The dataset will also need to reflect what the model actually receives in practice not perfect, clean text, but the kind of slightly imperfect output that comes from a speech recognition engine. The data will be split into training and validation sets using a stratified approach, so every intent is represented fairly in both.

### 3. Model Training

- An offline speech recognition model will be selected based on suitability for low-resource commodity hardware. The intent classification model will be a compact, fine-tuned language model chosen for its ability to run on-device.
- The model will be trained on the prepared dataset and converted to a mobile device compatible format for on-device deployment. The NLU pipeline will combine rule-based keyword matching for common, high-confidence requests with the neural model for the wider range of phrasing giving better overall reliability than either approach alone.

### 4. System Development

The full system will be built across three components:

- **Android App (Guest Device):** On-device speech recognition and intent classification, covering the full request lifecycle submission, confirmation, cancellation, and rating with real-time status updates from the hotel server.
- **Backend Server:** Manages the request lifecycle, routes requests to the correct department automatically, and handles real-time bidirectional communication with both guest devices and staff.
- **Staff Dashboard:** A web-based interface showing department-specific request queues, allowing staff to update request status and send messages back to guests.

### 5. Evaluation

The system will be evaluated across three dimensions:

- **Intent classification accuracy:** measured on both clean text and speech-transcribed input, to quantify any accuracy gap introduced by the on-device speech recognition step
- **System performance:** end-to-end latency per pipeline stage and resource consumption on budget Android hardware
- **Cost:** cost-per-request estimate compared to cloud-based voice assistant alternatives (Alexa, Google, Dialogflow)

## High-Level System Conceptual View

The system operates across three distinct layers connected via a local Wi-Fi network within the hotel premises. No external internet connection is required for core operations. All AI processing occurs on the guest device. The server handles only coordination and persistence.

## Timeline

| Work Package | Planned Duration |
|---|---|
| WP1: Research and Planning | 5 weeks |
| WP2: Dataset Development and Model Training | 5 weeks |
| WP3: Android Application Development | 8 weeks |
| WP4: Backend and Dashboard Development | 5 weeks |
| WP5: Integration, Testing and Evaluation | 6 weeks |
| WP6: Report Writing and Submission | 13 weeks |
| **Total** | **25 weeks** |

## Expected Outcome

1. **A Working Prototype:** A complete, offline voice assistant running on a commodity Android device. The system handles the full request flow - voice input, intent classification, department routing, staff notification, and guest feedback - without any internet connection. It consists of an Android guest app, a backend server, and a web-based staff dashboard.
2. **A Hotel-Specific Intent Classification Dataset:** A labelled dataset of natural language utterances covering the most common hotel room service request categories, designed to reflect realistic speech conditions. To the best of the author's knowledge, this would be the first publicly documented intent dataset built specifically for hospitality room service.
3. **A Fine-Tuned On-Device Intent Classification Model:** A small-scale language model fine-tuned on the hotel intent dataset and optimized for on-device inference on commodity Android hardware. The model is expected to achieve greater than 90% intent classification accuracy under realistic speech recognition conditions.
4. **Performance Evaluation:**
   - Intent classification accuracy under both clean text and speech-transcribed input conditions
   - End-to-end latency measurements per pipeline stage
   - Cost-per-request comparison against cloud-based voice assistant alternatives
5. **A Privacy-Preserving Architecture:** All voice processing stays on the guest device no audio is ever transmitted outside the hotel network. This gives the system a structural privacy guarantee that goes beyond policy promises and makes it GDPR-compatible for hotels serving international guests.

## Conclusion

No existing system combines offline speech recognition, on-device intent classification, and a real-time hotel management backend into a single, affordable product. This project builds exactly that. By running all AI processing on the guest's device with no cloud dependency, the system avoids the recurring costs and privacy risks of commercial alternatives like Alexa for Hospitality.

The research will contribute a hotel-specific intent dataset, a fine-tuned on-device intent classification model, and a complete working prototype deployable on commodity Android hardware. It will also investigate whether training the model on realistic speech-transcribed examples alongside clean text can address the accuracy challenges that on-device speech recognition introduces an area that no existing published work has explored in this specific offline hospitality context.

This is a realistic, deployable solution for hotels in developing economies that want to modernize guest services without expensive infrastructure or cloud subscriptions.

## References

Alpha Cephei. (2020). Vosk Offline Speech Recognition API. https://alphacephei.com/vosk/

Buhalis, D. & Moldavska, I. (2021). In-room Voice-Based AI Digital Assistants Transforming On-Site Hotel Services and Guests' Experiences. In *Information and Communication Technologies in Tourism 2021*. Springer.

Buhalis, D. & Moldavska, I. (2022). Voice Assistants in Hospitality: Using Artificial Intelligence for Customer Service. *Journal of Hospitality and Tourism Technology*, 13(3), 386–403.

Bujel, K., Laffy, J. & Kochenderfer, M. (2021). Effectiveness of Pre-training for Few-shot Intent Classification. arXiv:2109.05782.

Bunk, T. et al. (2020). DIET: Dual Intent and Entity Transformer. Rasa Technologies.

Chantrapornchai, C. & Suchato, A. (2022). IoT Device Control with Offline Automatic Speech Recognition on Edge Device. IEEE Conference Publication.

Hwang, J. & Erdem, M. (2025). "U" in User Experience (UX) Stands for the Frontline Employee: A Case Study of Voice Assistant Technology Use in Hotels. *Journal of Hospitality & Tourism Cases*.

Korkmaz, A. et al. (2025). Real-Time Speech-to-Text on Edge: A Prototype System for Ultra-Low Latency Communication with AI-Powered NLP. *Information*, 16(8), 685.

Larson, S. et al. (2022). A Survey of Intent Classification and Slot-Filling Datasets for Task-Oriented Dialog. arXiv:2207.13211.

Purdue University. (2018). Amazon Alexa Devices in Hotels Raise Privacy Concerns. Purdue University Newsroom.

Radford, A. et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). ICML 2023.

Sanh, V. et al. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *NeurIPS 2019 Workshop*.

Shi, W. et al. (2016). Edge Computing: Vision and Challenges. *IEEE Internet of Things Journal*, 3(5), 637–646.

Sun, Z. et al. (2020). MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. ACL 2020.
