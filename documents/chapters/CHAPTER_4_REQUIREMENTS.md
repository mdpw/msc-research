# CHAPTER 4: REQUIREMENTS

## 4.1 Introduction

This chapter documents how the system requirements were gathered and prioritised. Understanding what the system needs to do — and what constraints it must operate within — required input from two different perspectives: the hotel staff who would manage the system day-to-day, and the guests who would actually use it. Both perspectives informed the final requirements, and the differences between them highlighted some tensions that shaped the overall design.

---

## 4.2 Requirements Elicitation

### 4.2.1 Hotel Management Interviews

Semi-structured interviews were conducted with management and front-desk staff at [X] small to mid-sized hotels across the Southern and Western Provinces of Sri Lanka. These locations were chosen because they represent the target deployment context — hotels in tourist-facing areas that handle a consistent volume of guest requests but lack the IT infrastructure or budget of larger chain hotels.

The interviews were semi-structured rather than fully scripted. This allowed conversation to move naturally toward the concerns most relevant to each property, while still covering the same core topics across all sites. The interviews covered:

- How guest service requests are currently handled
- The most common types of requests and how often they occur
- Where the existing process breaks down
- How requests are coordinated between departments
- The technology and budget constraints the hotel operates under
- Whether staff would be willing to adopt a voice-based system in guest rooms

**Key findings:**

1. Every hotel interviewed relied on telephone calls to the front desk as the primary method for handling all guest requests. During peak hours — particularly evenings and early mornings — this created a visible bottleneck, with calls going unanswered or staff having to prioritise between multiple incoming requests.

2. Miscommunication was a recurring problem, especially with international guests. Staff reported frequent cases where guests' requests were misunderstood over the phone, resulting in the wrong items being delivered or requests being routed to the wrong department.

3. None of the hotels had a formal system for tracking request status or measuring how long requests took to fulfil. When guests followed up on a pending request, staff had to ask colleagues directly, which caused further delays.

4. Recurring subscription costs were consistently flagged as a dealbreaker. Several managers mentioned that they had looked at existing voice assistant products but were deterred by monthly fees or the requirement for a dedicated internet connection with guaranteed uptime.

5. Internet connectivity was unreliable at several properties, particularly those located outside Colombo. Some hotels experienced regular outages of several hours, making any cloud-dependent system an unacceptable operational risk.

### 4.2.2 Guest Survey

A survey was administered to [X] individuals who had prior experience staying in hotels as guests. Respondents were recruited from [context, e.g., university contacts, social networks] and all had stayed in at least one hotel within the past 12 months. The survey aimed to understand:

- Satisfaction with current methods for making room service requests
- Comfort level with using a voice assistant for hotel services
- Privacy concerns about voice-enabled devices in hotel rooms
- Which services they would most value through a voice interface
- Preferences for how the system should confirm a request before submitting it

**Key findings:**

1. [X]% of respondents reported experiencing delays or miscommunication when making service requests by telephone during a hotel stay.

2. [X]% expressed willingness to use a voice assistant for room service if they were assured that audio was not recorded or transmitted externally.

3. The most commonly requested services across the survey were housekeeping, towel and toiletry requests, food ordering, and wake-up calls — which directly informed the 18 intent categories defined for the system.

4. A clear majority preferred voice input with on-screen text confirmation before submission, rather than voice-only interaction. This preference for a review step before committing a request was a consistent theme, particularly around food orders where errors would be most disruptive.

5. Privacy was identified as the most significant concern. [X]% of respondents said they would be uncomfortable with a cloud-connected device listening in their hotel room. This finding strongly reinforced the decision to process all audio on-device.

---

## 4.3 Functional Requirements

The following functional requirements were derived from the combined findings of the management interviews and guest survey. Requirements that emerged from a single source are noted accordingly; those supported by both are considered higher priority.

**Table 4.1: Functional Requirements**

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-01 | The system shall accept guest service requests through voice input | Must | Guest survey, Management interviews |
| FR-02 | The system shall convert speech to text on the device without internet connectivity | Must | Management interviews (connectivity issues) |
| FR-03 | The system shall classify guest requests into one of 18 predefined hotel service intent categories | Must | Management interviews (routing needs) |
| FR-04 | The system shall display a text confirmation of the recognised request before submission | Must | Guest survey (preference for confirmation step) |
| FR-05 | The system shall provide voice feedback confirming that a request has been submitted | Must | Guest survey |
| FR-06 | The system shall deliver requests to the staff dashboard in real time over the hotel's local network | Must | Management interviews (response time concerns) |
| FR-07 | The system shall allow staff to update request status (pending, in progress, completed, cancelled) | Must | Management interviews (no existing tracking method) |
| FR-08 | The system shall route requests to the appropriate hotel department automatically based on intent | Must | Management interviews (coordination challenges) |
| FR-09 | The system shall enable bidirectional messaging between staff and the guest's room device | Should | Management interviews |
| FR-10 | The system shall allow guests to cancel a pending request by voice command | Should | Guest survey |
| FR-11 | The system shall allow guests to rate a completed service on a 1–5 scale | Should | Management interviews (no performance measurement) |
| FR-12 | The system shall allow staff to transfer a request to a different department via the dashboard | Should | Management interviews |
| FR-13 | The system shall maintain a record of all requests for operational review | Could | Management interviews |

---

## 4.4 Non-Functional Requirements

The non-functional requirements define the constraints under which the system must operate. Several of these were not explicitly stated by interviewees but were inferred from the practical realities they described — for example, the requirement for offline operation came directly from reports of unreliable internet connectivity, and the cost constraint was set based on budget ranges mentioned across multiple properties.

**Table 4.2: Non-Functional Requirements**

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR-01 | The system shall operate fully without internet connectivity | Full offline capability | Unreliable connectivity at target hotel locations |
| NFR-02 | All voice processing shall occur on-device; no audio or transcript shall be transmitted externally | Zero external voice data | Privacy concerns identified by both guests and management |
| NFR-03 | End-to-end response time from voice input to voice confirmation shall not exceed 5 seconds | < 5 seconds | Guest experience — comparable to a telephone call response |
| NFR-04 | The system shall run on commodity Android tablets costing under $150 USD | < $150 per room | Budget constraints of target small hotels |
| NFR-05 | The system shall achieve a minimum intent classification accuracy of 90% on real speech input | ≥ 90% | Reliability threshold for service routing |
| NFR-06 | The system shall support concurrent operation across multiple room devices on a single hotel server | Multi-room | Practical deployment on a single on-site machine |
| NFR-07 | The staff dashboard shall be accessible from any device with a web browser without installation | Browser-only access | Minimise additional hardware requirements for staff |
| NFR-08 | The system shall require no specialist IT expertise for initial deployment and daily operation | Zero IT dependency | Limited or absent IT staff at small Sri Lankan hotels |

---

## 4.5 Requirements Prioritisation

Requirements were prioritised using the MoSCoW method, with input from the hotel management interviews to determine what would be genuinely necessary for adoption versus what would be a nice addition.

**Table 4.3: MoSCoW Prioritisation**

| Priority | Requirements |
|----------|-------------|
| Must Have | FR-01, FR-02, FR-03, FR-04, FR-05, FR-06, FR-07, FR-08, NFR-01, NFR-02, NFR-04 |
| Should Have | FR-09, FR-10, FR-11, FR-12, NFR-03, NFR-05, NFR-06 |
| Could Have | FR-13, NFR-07, NFR-08 |
| Won't Have (this release) | Multilingual support, voice-based guest feedback collection, integration with existing hotel PMS software |

The "Won't Have" category reflects deliberate scope decisions rather than undesirable features. Multilingual support — particularly Sinhala and Tamil — was mentioned by several managers as highly valuable for local-language guests, but it falls outside the scope of this research prototype given the limitations of available offline STT models for those languages. These are discussed as future work in Chapter 11.

---

## 4.6 Summary

Requirements were gathered through semi-structured interviews with hotel management at [X] properties in Sri Lanka and a survey of [X] past hotel guests. This dual-perspective approach ensured the system design was grounded in both operational realities and actual guest preferences, rather than assumptions made from either side alone.

The findings converged on a clear picture: hotels need a system that reduces the telephone bottleneck, improves request tracking, and is affordable to deploy without ongoing subscription costs or IT support. Guests want a system that is fast, easy to use, and — most critically — does not transmit their voice data anywhere outside the room.

These findings directly shaped the system's core design constraints: fully offline operation, on-device audio processing, commodity Android hardware, and zero recurring software costs. The "Won't Have" items identified through prioritisation are revisited as future work in Chapter 11. The following chapter presents the analysis of these requirements and the evaluation of candidate technologies to fulfil them.
