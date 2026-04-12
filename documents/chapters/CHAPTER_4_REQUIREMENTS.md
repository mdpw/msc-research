# CHAPTER 4: REQUIREMENTS

## 4.1 Introduction

This chapter covers how the system requirements were gathered and prioritised. Since this is a research prototype without a hotel industry client, the requirements came from two places: published academic literature on hotel operations and voice assistant adoption (to understand what hotels actually need), and a small survey of people with recent hotel stay experience (to understand what guests want). Together, these requirements define what "viable alternative to traditional room service communication" means in the Sri Lankan hotel context — and they set the standard against which the prototype is later evaluated.

---

## 4.2 Requirements Elicitation

### 4.2.1 Hotel Operational Requirements — Literature Review

There is no hotel client for this project, so the operational requirements had to come from somewhere else. The approach used here — drawing requirements from published research when no direct client exists — is standard practice in Design Science Research (Hevner et al., 2004). The literature review in Chapter 2 covered several studies on voice assistant adoption and hotel operations (Buhalis and Moldavska, 2021, 2022; Hwang and Erdem, 2025; Yilmaz et al., 2025), and these were combined with Sri Lanka-specific sources (Wickramasinghe and Ratnayake, 2022; SLTDA, 2024) to work out what operational problems the system actually needs to solve.

**Key findings from the literature:**

1. **Telephone-based service creates bottlenecks.** All guest requests going through a single front desk line means that during busy periods, calls go unanswered, staff juggle multiple requests at once, and response times suffer (Buhalis and Moldavska, 2022). Hwang and Erdem (2025) confirm this from the staff side, showing that manual request handling is one of the main workload problems frontline hotel staff face.

2. **Miscommunication is a real problem, especially with international guests.** When requests are handled verbally over the phone, things get misunderstood — the wrong items get delivered, or the request goes to the wrong department. Buhalis and Moldavska (2021) point to this as one of the main reasons hotels have started looking at voice assistants: a structured, intent-based system removes the guesswork from freeform phone conversations. Yilmaz et al. (2025) also connect these kinds of service errors directly to negative online reviews.

3. **Small hotels have no way to track requests.** Most independent hotels have no system for logging what was requested, when it was fulfilled, or how long it took (Buhalis and Moldavska, 2022). When a guest asks about a pending request, staff have to physically check with colleagues. A simple digital log with real-time status updates would directly fix this.

4. **Subscription costs stop small hotels from adopting commercial solutions.** Commercial platforms need dedicated hardware per room, ongoing subscription fees, and reliable internet — none of which fits the budget of a small independent hotel (Buhalis and Moldavska, 2022). This is especially relevant in Sri Lanka, where the accommodation sector is mostly small, owner-operated properties (SLTDA, 2024).

5. **Internet connectivity is unreliable outside Colombo.** Wickramasinghe and Ratnayake (2022) document that digital infrastructure quality varies a lot across Sri Lanka, with hotels outside the capital regularly losing connectivity for hours at a time. A system that needs constant internet access is simply not practical for this market. This finding directly drove the offline-first requirement.

To supplement the literature, an informal consultation was conducted with a Head Chef with over 15 years of experience across multiple Sri Lankan hotels. He described the current process: guest requests arrive at the front desk by telephone, the front desk forwards them to the relevant department, and the department manager records them in a physical logbook. Items are only entered into a computer system at final billing. No automated routing or real-time tracking is involved at any stage. This directly confirms Findings 1, 3, and 4 above, and provides local practitioner evidence that the problem this system addresses is real and current.

### 4.2.2 Guest Survey

A short survey was given to 20 individuals with recent hotel stay experience, recruited through convenience sampling — family members, relatives, and colleagues known to the researcher. The sample covered a range of ages (under 20 to 60–70), since voice assistant comfort levels tend to vary across age groups. The full survey instrument and demographic summary are in Appendix C.

The survey covered:

- How respondents currently make room service requests and whether they had experienced problems
- How comfortable they would be using a voice assistant for hotel services
- Privacy concerns about voice-enabled devices in hotel rooms
- Which services they would use most through a voice interface
- How they would want the system to confirm a request before submitting it

**Key findings:**

1. **75% (15/20)** said they had experienced delays or miscommunication when making service requests by phone during a hotel stay. This directly backed up what the literature says about telephone-based service being a bottleneck.

2. **65% (13/20)** said they would use a voice assistant for room service if they knew the audio was not recorded or sent anywhere outside the room. Privacy assurance was the key factor — several respondents who initially said they would not use one changed their answer once the on-device processing was explained.

3. The most commonly requested services were towels and bedding (90%), housekeeping (85%), food and beverage orders (80%), and toiletries (65%) — which directly shaped the 18 intent categories built into the system.

4. **80% (16/20)** preferred voice input with an on-screen text confirmation before the request was submitted, rather than voice-only interaction. This was especially strong for food orders, where mistakes are most disruptive, and directly drove the design of the confirmation step in the guest application.

5. Privacy was a notable concern. **25% (5/20)** said they would be uncomfortable with a cloud-connected voice device in their hotel room. This reinforced the decision to keep all audio processing on the device and not transmit any voice data externally.

---

## 4.3 Functional Requirements

The requirements below came from combining the literature findings (Section 4.2.1) with the guest survey results (Section 4.2.2). Where a requirement came from only one source, that is noted. Requirements backed by both sources were treated as higher priority.

**Table 4.1: Functional Requirements**

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| FR-01 | The system shall accept guest service requests through voice input | Must | Guest survey; Buhalis and Moldavska (2022) |
| FR-02 | The system shall convert speech to text on the device without internet connectivity | Must | Wickramasinghe and Ratnayake (2022) — connectivity constraints |
| FR-03 | The system shall classify guest requests into one of 18 predefined hotel service intent categories | Must | Buhalis and Moldavska (2021, 2022) — routing needs |
| FR-04 | The system shall display a text confirmation of the recognised request before submission | Must | Guest survey — preference for confirmation step |
| FR-05 | The system shall provide voice feedback confirming that a request has been submitted | Must | Guest survey |
| FR-06 | The system shall deliver requests to the staff dashboard in real time over the hotel's local network | Must | Hwang and Erdem (2025) — response time concerns |
| FR-07 | The system shall allow staff to update request status (pending, in progress, completed, cancelled) | Must | Buhalis and Moldavska (2022) — no existing tracking method |
| FR-08 | The system shall route requests to the appropriate hotel department automatically based on intent | Must | Buhalis and Moldavska (2022) — coordination challenges |
| FR-09 | The system shall enable bidirectional messaging between staff and the guest's room device | Should | Hwang and Erdem (2025) |
| FR-10 | The system shall allow guests to cancel a pending request by voice command | Should | Guest survey |
| FR-11 | The system shall allow guests to rate a completed service on a 1–5 scale | Should | Yilmaz et al. (2025) — service quality measurement |
| FR-12 | The system shall allow staff to transfer a request to a different department via the dashboard | Should | Buhalis and Moldavska (2022) |
| FR-13 | The system shall maintain a record of all requests for operational review | Could | Buhalis and Moldavska (2022) |

The 18 intent categories in FR-03 were defined by looking at room service menus from Sri Lankan hotels, the use cases in Buhalis and Moldavska (2021, 2022), and the service categories supported by Alexa for Hospitality.

**Table 4.1a: Predefined Intent Categories (FR-03)**

| Intent Category | Description |
|----------------|-------------|
| towel_request | Requests for towels |
| room_cleaning | Housekeeping and cleaning requests |
| food_order | Food, beverage, and room service orders |
| toiletries_request | Bathroom amenity requests |
| pillow_request | Pillow and bedding requests |
| temperature_control | Heating and cooling requests |
| blanket_request | Blanket and comforter requests |
| maintenance | Technical and repair requests |
| laundry_service | Laundry and dry cleaning requests |
| concierge_general | General information and service queries |
| wake_up_call | Alarm and wake-up requests |
| concierge_taxi | Transportation and taxi requests |
| do_not_disturb | Privacy and do-not-disturb requests |
| lighting_control | Light adjustment requests |
| noise_complaint | Noise and disturbance complaints |
| emergency | Emergency and urgent assistance |
| checkout_billing | Checkout and billing inquiries |
| misc_request | Uncategorised or general requests |

---

## 4.4 Non-Functional Requirements

The non-functional requirements define the constraints the system has to work within. Each one follows directly from the findings in Section 4.2, with sources noted in the table.

**Table 4.2: Non-Functional Requirements**

| ID | Requirement | Target | Rationale |
|----|-------------|--------|-----------|
| NFR-01 | The system shall operate fully without internet connectivity | Full offline capability | Unreliable connectivity at target hotel locations |
| NFR-02 | All voice processing shall occur on-device; no audio or transcript shall be transmitted externally | Zero external voice data | Privacy concerns from guests and literature |
| NFR-03 | End-to-end response time from voice input to voice confirmation shall not exceed 5 seconds | < 5 seconds | Guest experience — comparable to a telephone response |
| NFR-04 | The system shall run on commodity Android tablets costing under $150 USD | < $150 per room | Budget constraints of target small hotels |
| NFR-05 | The system shall achieve a minimum intent classification accuracy of 90% on real speech input | ≥ 90% | Reliability threshold for service routing |
| NFR-06 | The system shall support concurrent operation across multiple room devices on a single hotel server | Multi-room | Practical deployment on a single on-site machine |
| NFR-07 | The staff dashboard shall be accessible from any device with a web browser without installation | Browser-only access | Minimise additional hardware requirements for staff |

---

## 4.5 Requirements Prioritisation

Once the requirements were defined, they were prioritised using the Must have, Should have, Could have, Won't have (MoSCoW) method. The split between Must Have and Should Have was based on what the literature points to as essential for adoption in small hotel settings, versus what would be useful but is not a dealbreaker (Buhalis and Moldavska, 2022; Hwang and Erdem, 2025).

**Table 4.3: MoSCoW Prioritisation**

| Priority | Requirements |
|----------|-------------|
| Must Have | FR-01, FR-02, FR-03, FR-04, FR-05, FR-06, FR-07, FR-08, NFR-01, NFR-02, NFR-04 |
| Should Have | FR-09, FR-10, FR-11, FR-12, NFR-03, NFR-05, NFR-06 |
| Could Have | FR-13, NFR-07 |
| Won't Have (this release) | Multilingual support, voice-based guest feedback collection, integration with existing hotel Property Management System (PMS) software |

The "Won't Have" items are deliberate scope decisions, not oversights. Multilingual support — particularly Sinhala and Tamil — would be highly valuable for Sri Lankan guests, but it is out of scope for this prototype given the current limitations of available offline STT models for those languages. These are discussed as future work in Chapter 11.

---

## 4.6 Summary

Requirements were gathered from three sources: published academic literature on hospitality operations, an informal consultation with an experienced Sri Lankan hotel professional, and a convenience survey of 20 people with recent hotel stay experience (Section 4.2). Using all three gave a clearer picture than any single source alone would have.

The picture that emerged is consistent: hotels need something that removes the telephone bottleneck, gives staff a way to track requests, and can be deployed without subscription fees or IT support. Guests want it to be fast, straightforward, and — above everything else — not sending their voice data anywhere outside their room.

These requirements directly shaped the design: offline operation, on-device audio processing, commodity Android hardware, and no subscription fees. A prototype that fails on any one of these would not be a credible alternative to traditional room service communication in a small Sri Lankan hotel. Multilingual support is picked up as future work in Chapter 11. The next chapter covers the technology choices and analysis that turned these requirements into a working design.
