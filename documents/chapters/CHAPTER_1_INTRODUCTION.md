# CHAPTER 1: INTRODUCTION

## 1.1 Introduction

This chapter introduces the research context, motivation, and scope of the study. It begins with the background of voice assistant technology and its relevance to the hospitality industry, particularly in developing economies such as Sri Lanka. The chapter then identifies the core problem and research gaps, followed by the four research objectives that guide this study. Finally, it outlines the overall structure of the dissertation.

---

## 1.2 Background to the Project

The hospitality industry is a significant contributor to the Sri Lankan economy. Sri Lanka attracted over 2 million international tourists in 2024, with India accounting for approximately 20% of all arrivals (Sri Lanka Tourism Development Authority (SLTDA), 2024). Hotels are continuously looking for ways to improve guest experience while managing operational costs. Despite this, room service operations in most hotels still rely on traditional methods: telephone calls to the front desk and manual request handling. These approaches are slow, prone to miscommunication, and give hotel management no easy way to track what was requested, when it was fulfilled, or how long it took.

The emergence of voice-based virtual assistants — such as Amazon Alexa for Hospitality and Google Nest for Hotels — has introduced new possibilities for automating guest service operations. These systems allow guests to place requests, inquire about hotel amenities, and interact with services through natural voice commands, reducing friction and improving response times. However, these commercial solutions carry significant drawbacks for hotels in developing countries. They rely entirely on cloud infrastructure, require continuous internet connectivity, and send guest voice data to third-party servers — raising privacy concerns. Buhalis and Moldavska (2022) found that 67% of guests feel uncomfortable with cloud-connected microphones in their hotel rooms, and documented risks of third-party skills accessing guest audio without consent. Beyond privacy, the cost structure of cloud-based systems — proprietary hardware at USD 99–200 per device, combined with ongoing Application Programming Interface (API) and subscription fees — is simply not feasible for most small and mid-sized hotels in Sri Lanka.

Internet connectivity adds another layer of difficulty. While Sri Lanka's infrastructure has improved in urban areas, hotels outside Colombo regularly experience outages or poor connectivity (Wickramasinghe and Ratnayake, 2022). A system that stops working whenever the internet is unavailable is not suitable for this market. These limitations make existing commercial voice assistants impractical for the majority of Sri Lankan hotels, despite the clear operational need.

This project addresses that gap directly. It proposes and builds _Sera_ — a voice assistant designed specifically for hotel room service that runs entirely on a standard Android tablet without any internet dependency. All speech recognition and intent classification happen on the guest's device using two small, open-source AI models: Vosk for offline speech-to-text and MobileBERT for intent classification. No voice data leaves the room. Requests are routed automatically to the correct hotel department over the local network, and staff manage all incoming requests through a web-based dashboard in real time. The total hardware cost per room sits below USD 150 — within reach of small independent hotels.

The research question guiding this study is: _Can a low-cost, fully offline voice assistant prototype, built on a standard Android device, achieve sufficient technical accuracy and performance to be a viable alternative to traditional room service communication in Sri Lankan hotels?_

---

## 1.3 Project Objectives

The main objectives of this project are:

### 1.3.1 Design and Develop a Low-Cost, Offline Voice Assistant Prototype

Design and build a fully offline, end-to-end voice assistant prototype covering on-device speech recognition, intent classification, and the complete request lifecycle — including submission, confirmation, cancellation, and guest rating — deployable on commodity Android hardware in the USD 50–150 price range, without any cloud or internet dependency.

### 1.3.2 Build a Hospitality-Domain Dataset and Train a Noise-Aware NLU Model

Create a labelled intent classification dataset covering the most common room service request categories in Sri Lankan hotels, and train a MobileBERT-based intent classifier that can handle the accuracy challenges introduced by on-device speech recognition — specifically, the transcription noise produced by the Vosk Speech-to-Text (STT) engine under real deployment conditions.

### 1.3.3 Build a Lightweight Backend with Real-Time Staff Communication

Implement a lightweight FastAPI and SQLite backend with automated department routing and a browser-accessible staff dashboard, supporting real-time bidirectional communication between guest devices and hotel staff via WebSocket — with no cloud infrastructure required.

### 1.3.4 Evaluate System Accuracy, Latency, and Cost-Effectiveness

Evaluate the prototype's intent classification accuracy, speech recognition word error rate, end-to-end response latency, and per-room hardware cost against the requirements defined for a viable room service alternative — and demonstrate that a fully privacy-preserving, offline voice processing architecture is achievable on low-cost commodity hardware.

---

## 1.4 Overview of This Report

The remainder of this dissertation is structured as follows.

**Chapter 2 — Literature Review** surveys the academic and commercial landscape of voice assistants in hospitality, offline speech recognition systems, small-scale Natural Language Understanding (NLU) models for resource-constrained devices, and privacy-preserving edge AI architectures. It identifies the research gaps that this project addresses.

**Chapter 3 — Methodology** describes the overall research approach: a Design Science Research framework combined with iterative software development, and a controlled three-model NLU experiment to evaluate the accuracy impact of real on-device STT conditions.

**Chapter 4 — Requirements** covers requirements elicitation from both the academic literature and a hotel management and guest survey survey, and defines the functional and non-functional requirements used to evaluate the prototype throughout the project.

**Chapter 5 — Analysis** translates the requirements into a system architecture and technology choices, justifying the selection of Vosk, MobileBERT, FastAPI, and Android as the core technology stack.

**Chapter 6 — Design** details the system design: the three-tier architecture, data model, API contracts, Android UI flows, and the hybrid NLU pipeline combining rule-based keyword matching with MobileBERT neural inference.

**Chapter 7 — Implementation** documents how the system was built across four development iterations, covering the dataset construction pipeline, model training, Android application, and backend service.

**Chapter 8 — Testing** presents the evaluation results: the three-model NLU accuracy comparison, Vosk word error rate analysis, end-to-end latency measurements, and Non-Functional Requirement (NFR) compliance assessment.

**Chapter 9 — Project Management** reflects on the development process, timeline, risk management, and the iterative decisions made during implementation.

**Chapter 10 — Critical Appraisal** honestly assesses the limitations of the prototype, including the synthetic evaluation data, tokenizer mismatch on Android, and the constraints on generalisability.

**Chapter 11 — Conclusions** summarises the findings, revisits the research question, and outlines the most important directions for future work.

**Chapter 12 — Student Reflections** offers a personal account of the learning experience, challenges encountered, and insights gained throughout the project.

---

## 1.5 Summary

The Sri Lankan hospitality sector has a clear operational gap: commercial voice assistants are too expensive, too cloud-dependent, and too privacy-invasive for most hotels in the market. The four objectives in Section 1.3 address this gap through the design, development, and evaluation of a fully offline, low-cost voice assistant prototype. The research question asks whether such a prototype can achieve sufficient technical accuracy and performance to be a viable alternative to traditional room service communication — a question the remaining chapters build toward answering. Chapter 2 surveys the existing academic and commercial work and identifies the specific gaps this project addresses.
