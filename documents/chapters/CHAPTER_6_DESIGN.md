# CHAPTER 6: DESIGN

## 6.1 Introduction

This chapter presents the detailed system design of the low-cost offline voice assistant prototype. The technologies selected and justified in Chapter 5 are brought together here into a cohesive architecture. The chapter covers the three-tier system architecture, the voice processing pipeline, the database schema, the API and communication design, and the user interface design. Where the design includes novel or less obvious decisions, the reasoning behind them is explained. Diagrams are complemented by discussion that goes beyond restating what is visually evident.

---

## 6.2 System Architecture

The system is structured around one core principle: all computationally intensive AI tasks — speech recognition and intent classification — run on the guest device. The server handles only coordination, request persistence, and communication routing. This separation means the server needs no GPU and can run on any low-cost laptop or desktop already available in the hotel.

### 6.2.1 Three-Tier Architecture

**Figure 6.1: System Architecture Diagram**

*(See attached architecture diagram — Figure 6.1)*

```
+====================================================================+
|                      HOTEL LOCAL NETWORK                           |
|                                                                    |
|  TIER 1: GUEST DEVICE          TIER 2: HOTEL SERVER               |
|  (Android Tablet per room)     (Any low-cost PC/Laptop)           |
|                                                                    |
|  +------------------------+    +------------------------------+    |
|  | AudioRecorder          |    | FastAPI Application          |    |
|  | (16kHz PCM, VAD)       |    |                              |    |
|  |   |                    |    | +---------+  +------------+  |    |
|  |   v                    |    | | SQLite  |  | WebSocket  |  |    |
|  | VoskService            |    | | DB      |  | Conn Mgr   |  |    |
|  | (vosk-small-en-in-0.4) | HTTP| | 5 tables|  | /ws/guest/ |  |    |
|  |   |                    |---->| |         |  | /ws/dash   |  |    |
|  |   v                    |    | +---------+  +------------+  |    |
|  | NLUService (Hybrid)    |    | | Routing Engine             |  |    |
|  | +-Rule keywords (regex)|    | | DB-driven + keyword        |  |    |
|  | +-MobileBERT TFLite    |    | | fallback + Front Desk      |  |    |
|  |   (26MB, 18 intents)   |    | | default                    |  |    |
|  |   |                    |    | +----------------------------+  |    |
|  |   v                    |    +-----+------------------------+    |
|  | TextToSpeechService    |          |                             |
|  | (Android native TTS)   |          | WebSocket                  |
|  |   ^                    |          v                             |
|  | WebSocketService       |    TIER 3: STAFF DASHBOARD             |
|  | <------WebSocket--------|    (Any Web Browser)                  |
|  +------------------------+    +------------------------------+    |
|                                | HTML/CSS/JS Single-Page App  |    |
|                                | WebSocket client             |    |
|                                | Dept-filtered request queue  |    |
|                                | Real-time messaging UI       |    |
|                                +------------------------------+    |
+====================================================================+
       All communication over hotel Wi-Fi — no internet required
```

**Tier 1 — Guest Device (Android Tablet)**

Each guest room has one Android tablet running the application. The device handles the full AI pipeline independently: audio capture, Vosk speech recognition, hybrid NLU classification, and TTS confirmation. Communication with the server happens over HTTP for request submission and WebSocket for receiving real-time updates.

The key components on the device are:
- `AudioRecorder` — captures 16kHz, 16-bit PCM mono audio in 4,096-byte chunks, with voice activity detection based on RMS energy threshold (0.02)
- `VoskService` — on-device speech-to-text using `vosk-model-small-en-in-0.4` (~36MB)
- `NLUService` — hybrid two-tier classification pipeline (rule-based keywords → MobileBERT TFLite)
- `TextToSpeechService` — Android native TTS for voice confirmations and status announcements
- `ApiService` — HTTP REST client (OkHttp 4.11.0) for submitting requests and retrieving history
- `WebSocketService` — persistent WebSocket connection for real-time updates from the server

**Tier 2 — Hotel Server**

The server is a FastAPI application that can run on any PC or laptop connected to the hotel's local network. It exposes 17 endpoints (14 HTTP + 2 WebSocket + 1 dashboard serve), manages a SQLite database, handles department routing, and broadcasts events to connected clients via WebSocket.

The routing engine uses a hybrid approach: intent-based routing through the `intent_department_mapping` database table as the primary method, with a keyword-analysis fallback for cases where intent is missing or unrecognised, and Front Desk as the final default.

**Tier 3 — Staff Dashboard**

The staff dashboard is a web-based single-page application served directly by the FastAPI backend at `/dashboard`. It requires no installation — any staff member can open it in a browser on a desktop, tablet, or phone. It connects to the server's `/ws/dashboard` WebSocket endpoint to receive real-time updates and displays a department-filtered view of the request queue.

---

## 6.3 Voice Processing Pipeline Design

The voice processing pipeline covers the complete guest interaction from pressing the microphone button to the request appearing on the staff dashboard.

### 6.3.1 Pipeline Overview

**Figure 6.2: Voice Request Pipeline Sequence**

*(See attached sequence diagram — Figure 6.2)*

The pipeline has three major phases:

**Phase 1 — Audio Capture and Transcription**

When the guest presses the microphone button, the `AudioRecorder` begins capturing 16kHz PCM audio, feeding 4,096-byte chunks to the Vosk recogniser in real time. The VAD monitors the RMS energy of each chunk against a threshold of 0.02. Recording continues until either 1,500ms of continuous silence is detected (end of speech) or the maximum duration of 10,000ms is reached. The transcription produced by Vosk is then cleaned — filler words and greetings like "Hi Sera" are stripped, and the text is normalised to lowercase.

**Phase 2 — Intent Classification**

The cleaned transcription first passes through a cancel detection check using a regex pattern for "cancel order [number]". If matched, the request ID is extracted, a voice confirmation is played, and the cancel request is sent to the server. If not matched, the text enters the hybrid NLU pipeline.

**Phase 3 — Confirmation and Submission**

Once an intent is classified above the confidence threshold (0.60 for neural; 0.99 for keyword match), the system reads the recognised request back to the guest via TTS: *"You'd like [request description]. Shall I submit this?"* The guest's spoken yes/no response is captured. On confirmation, the request is sent to the server via HTTP POST. The server stores it, routes it to the appropriate department, and broadcasts a `new_request` WebSocket event to all connected staff dashboard instances.

### 6.3.2 Hybrid NLU Pipeline — Core Design Feature

The most distinctive feature of this system is the two-stage NLU pipeline. It was not part of the original design — it emerged during development when purely neural classification was observed producing lower-than-expected confidence on simple, unambiguous requests (e.g., "I need towels" classified as `pillow_request` with 0.72 confidence after minor Vosk transcription variations).

**Figure 6.3: Hybrid NLU Pipeline Flow**

```
Cleaned Transcription
        |
        v
+---------------------------+
| Cancel Pattern Check      |
| Regex: "cancel order \d+" |
+---------------------------+
        |
    Match? --Yes--> Extract ID → Voice Confirm → Cancel via HTTP → END
        |
       No
        |
        v
+---------------------------+        Match Found
| Tier 1: Keyword Matching  |------------------------> Intent + confidence=0.99
| Pre-compiled regex         |                              |
| 17 intent dictionaries    |                              v
| Multi-word contextual     |                    [Confirmation Step]
| phrases (not single words)|
+---------------------------+
        |
      No match
        |
        v
+---------------------------+
| Tier 2: MobileBERT TFLite |
| hotel_mobilebert_v2.tflite|
| Tokenise (max 32 tokens)  |
| Softmax over 18 classes   |
+---------------------------+
        |
        v
+---------------------------+
| Confidence >= 0.60?       |
+---------------------------+
        |
    No ---> TTS: "Sorry, could not understand your request." --> END
        |
      Yes
        |
        v
[Confirmation Step]
TTS: "You'd like [intent description]. Shall I submit this?"
        |
    No ---> TTS: "Request cancelled." --> END
        |
      Yes
        |
        v
HTTP POST to server → Store, route, broadcast → END
```

**Tier 1 — Rule-Based Keyword Matching**

The keyword dictionary covers 17 intent categories (all except `misc_request`). Each intent is mapped to a list of multi-word contextual phrases rather than single words, specifically to avoid false positives. For example, `food_order` uses phrases like "bottled water", "glass of water", and "room service" rather than matching the single word "water" — which could appear in unrelated contexts like "the water in the bathroom is leaking" (a `maintenance` request). The patterns are pre-compiled at application startup to eliminate compilation overhead during inference. A match returns the intent immediately with a fixed confidence score of 0.99, bypassing the neural model entirely.

**Tier 2 — MobileBERT Neural Inference**

When no keyword match is found, the text is tokenised using the BERT uncased vocabulary (30,522 tokens), padded or truncated to 32 tokens, and passed through the `hotel_mobilebert_v2.tflite` interpreter (26MB). The output logits are converted to class probabilities via softmax across all 18 intent categories. The highest-probability class is selected as the predicted intent.

The 0.60 confidence threshold was set empirically during development. Too high, and valid but unusual phrasings would be rejected unnecessarily. Too low, and misclassified requests would reach staff. The threshold was tuned to reject genuinely ambiguous input while accepting natural language variation.

---

## 6.4 Database Design

The database uses SQLite and contains five tables. The schema was designed to be both sufficient for the prototype and structurally compatible with a future migration to PostgreSQL without schema changes.

**Figure 6.4: Entity-Relationship Diagram**

*(See attached ER diagram — Figure 6.4)*

```
+----------------+       +---------------------+       +------------------+
|    rooms       |       | intent_dept_mapping |       |   departments    |
+----------------+       +---------------------+       +------------------+
| PK id          |       | PK id               |       | PK id            |
|    room_number |       |    intent (UNIQUE)  |       |    name (UNIQUE) |
|    floor       |       | FK dept_name        |------>|    description   |
|    room_type   |       +---------------------+       +------------------+
+-------+--------+                                              ^
        |                                                       |
        | room_number FK                                        |
        v                                                       |
+---------------------------+          +----------------------+ |
|         requests          |          |   staff_messages     | |
+---------------------------+          +----------------------+ |
| PK  id                    |1       * | PK  id               | |
| FK  room_number           |----------| FK  request_id       | |
|     request_text          |          |     message          | |
|     intent                |          |     staff_name       | |
| FK  department            |----------+     created_at       | |
|     status                |          +----------------------+ |
|     rating (nullable)     |                                   |
|     created_at            |-----------------------------------+
|     completed_at (null)   |
+---------------------------+
```

**Table descriptions:**

**`rooms`** — Pre-seeded with 15 rooms across three floors (101–105, 201–205, 301–305) covering Standard, Deluxe, and Suite types. The `room_number` field is the identifier used throughout the system — on request cards, WebSocket channel names, and the request history endpoint.

**`departments`** — Pre-seeded with five departments: Housekeeping, Room Service, Maintenance, Front Desk, and Concierge. Storing departments in a table rather than hardcoding them allows a hotel to rename or restructure departments without touching the application code.

**`intent_department_mapping`** — The routing table that maps each of the 18 intent categories to a department. This is the primary routing source consulted when a request is submitted. The design externalises routing logic from application code, meaning a hotel can redirect, say, `concierge_taxi` to Front Desk (if they have no dedicated concierge) by updating a single database row rather than modifying and redeploying the server.

**`requests`** — The core entity. Stores every service request from initial submission through completion. The `status` field tracks the request lifecycle (`pending` → `in_progress` → `completed` or `cancelled`). The `rating` field is nullable and only populated if the guest provides feedback after completion. The `completed_at` timestamp is recorded only when a request reaches `completed` status.

**`staff_messages`** — Stores all messages sent from staff to guest rooms, linked to the request they relate to. One request can have multiple messages.

---

## 6.5 API and Communication Design

### 6.5.1 REST API Endpoints

The backend exposes 15 HTTP endpoints following RESTful conventions, plus 2 WebSocket endpoints and 1 dashboard HTML serve — 17 endpoints in total.

**Table 6.1: HTTP API Endpoints**

| Method | Endpoint | Purpose | Called By |
|--------|----------|---------|-----------|
| GET | `/` | API status, intent count, department list | Monitoring |
| GET | `/api/departments` | List all department names | Dashboard |
| GET | `/api/departments/detail` | Departments with descriptions | Dashboard |
| GET | `/api/rooms` | List all registered room numbers | Dashboard / Admin |
| GET | `/api/intent-mapping` | Intent-to-department mapping | Debug / Admin |
| GET | `/api/intent-mapping/detail` | Full intent mapping with descriptions | Debug / Admin |
| GET | `/api/requests` | All requests in database | Dashboard |
| GET | `/api/request-history` | Request history for a specific room (`?room_number=`) | Guest device |
| GET | `/dashboard` | Serve the staff dashboard HTML | Browser |
| POST | `/api/submit-request` | Submit a new service request | Guest device |
| POST | `/api/update-status` | Update a request's status | Dashboard |
| POST | `/api/update-department` | Transfer request to a different department | Dashboard |
| POST | `/api/cancel-request` | Cancel a pending or in-progress request | Guest device |
| POST | `/api/send-message` | Send a staff message to a guest room | Dashboard |
| POST | `/api/rate-request` | Submit a guest rating (1–5) for a completed request | Guest device |

All endpoints use JSON for request and response bodies. Pydantic models validate incoming data automatically and return clear error messages for malformed requests.

### 6.5.2 WebSocket Communication Design

Two WebSocket endpoints handle real-time bidirectional communication between the server and clients. The guest and dashboard channels are deliberately kept separate — each client receives only the events relevant to it.

**Guest channel: `/ws/guest/{room_number}`**

Each guest device maintains a persistent WebSocket connection identified by its room number. The server pushes three event types:

| Event Type | Fields | Trigger |
|------------|--------|---------|
| `status_update` | `request_id`, `status`, `message` | Staff updates request status |
| `department_update` | `request_id`, `message` | Request transferred to different department |
| `staff_message` | `request_id`, `message`, `staff_name` | Staff sends a message to the room |

On the Android side, all incoming WebSocket events are read aloud via TTS so guests receive updates without needing to look at the screen.

**Dashboard channel: `/ws/dashboard`**

The staff dashboard connects to a single shared WebSocket endpoint. All connected dashboard instances receive the same event stream — when one staff member updates a request, all other open dashboards update in real time.

| Event Type | Fields | Trigger |
|------------|--------|---------|
| `initial` | `requests[]` (full list) | On first connection — gives the dashboard its full current state |
| `new_request` | `id`, `room_number`, `request_text`, `intent`, `department`, `status`, `created_at`, `rating` | Guest submits a new request |
| `status_update` | `request_id`, `status` | Request status changes |
| `department_update` | `request_id`, `department` | Request transferred |
| `staff_message` | `request_id`, `message`, `staff_name` | Message added |
| `rating_update` | `request_id`, `rating` | Guest submits a rating |

The `initial` event is sent immediately when a dashboard client connects, giving it the full current state of all requests without requiring a separate HTTP call.

**Figure 6.5: WebSocket Communication Topology**

```
+================================================================+
|                   HOTEL LOCAL WI-FI NETWORK                   |
|                                                                |
|  Room 101 Device  <---WebSocket--->  /ws/guest/101            |
|  Room 102 Device  <---WebSocket--->  /ws/guest/102  +-------+ |
|  Room 103 Device  <---WebSocket--->  /ws/guest/103  |FastAPI| |
|                                                     | Conn  | |
|  Housekeeping Dashboard <--WebSocket--> /ws/dash    | Mgr   | |
|  Front Desk Dashboard   <--WebSocket--> /ws/dash    +-------+ |
|                                                                |
|  Guest devices receive room-specific events only.             |
|  Dashboard receives all events for all rooms.                 |
+================================================================+
```

### 6.5.3 Request Lifecycle State Machine

A request transitions through a defined set of states from submission to completion. Invalid state transitions are prevented in the application logic — for example, a `completed` request cannot be moved back to `in_progress`.

**Figure 6.6: Request State Diagram**

```
                    Guest submits request
                            |
                            v
                        [pending]
                       /         \
          Guest cancels           Staff marks in progress
               |                          |
               v                          v
          [cancelled]              [in_progress]
                                  /            \
                       Guest cancels        Staff marks complete
                             |                     |
                             v                     v
                        [cancelled]          [completed]
                                                   |
                                          Optional: Guest rates
                                          service (1–5 stars)
```

At each state transition, a WebSocket event is broadcast — to the relevant guest device channel and to all connected dashboard instances simultaneously.

---

## 6.6 Department Routing Design

When a request is submitted via `POST /api/submit-request`, the server determines which department should handle it. The routing engine uses a three-level hierarchy:

1. **Intent-based routing (primary)** — The submitted request includes an `intent` field (classified on-device). The server looks this up in the `intent_department_mapping` table and routes accordingly.

2. **Keyword fallback (secondary)** — If the intent is missing or not found in the mapping table, the server analyses the `request_text` using hardcoded keyword lists for each department.

3. **Default routing** — If neither method produces a match, the request is sent to Front Desk.

**Table 6.2: Intent-to-Department Routing**

| Department | Intents | Colour |
|------------|---------|--------|
| Housekeeping | room_cleaning, towel_request, toiletries_request, blanket_request, pillow_request, laundry_service, do_not_disturb | #10b981 (green) |
| Room Service | food_order | #f59e0b (amber) |
| Maintenance | maintenance, temperature_control, lighting_control | #ef4444 (red) |
| Front Desk | wake_up_call, checkout_billing, noise_complaint, emergency, misc_request | #3b82f6 (blue) |
| Concierge | concierge_general, concierge_taxi | #8b5cf6 (purple) |

Staff can manually transfer a request to a different department via the dashboard dropdown. When this happens, the request is removed from the originating department's queue and appears immediately in the receiving department's queue via WebSocket.

---

## 6.7 User Interface Design

Both interfaces were designed around the specific usage context — not generic HCI guidelines. The guest app needs to work for someone in a dark hotel room, possibly from bed, with no prior training. The staff dashboard needs to handle a busy shift with multiple departments viewing the same system simultaneously.

### 6.7.1 Guest Application Interface

The app is built in Jetpack Compose with Material Design 3. The single-screen layout is divided into three functional sections.

**Figure 6.7: Guest Application Interface**

*(See attached annotated screenshot — Figure 6.7)*

**Section 1 — Room Information Bar (top)**
Displays the room number, Wi-Fi connection status, and the configured server network profile. This gives both guests and maintenance staff immediate context without navigating menus.

**Section 2 — Voice Interaction Area (centre, prominent)**
The centrepiece of the app is a large circular microphone button. During recording, the button pulses with an animation and a 20-bar audio level visualiser shows the captured signal. Below the button, a status line ("Tap to speak" → "Listening..." → "Processing...") gives real-time feedback. The last transcription, recognised intent, confidence score, and department colour indicator are shown after each interaction.

**Section 3 — Request History (bottom, scrollable)**
A scrollable list of the room's requests, sorted by status priority: in progress at the top, then pending, completed, and cancelled at the bottom. Each request card shows the request ID, status badge (colour-coded by state), request text (truncated to two lines), department colour, and a relative timestamp ("2 mins ago"). Pending and in-progress requests have a cancel button; completed requests have a 5-star rating selector.

**Table 6.3: Guest UI Design Decisions**

| Decision | Rationale |
|----------|-----------|
| Large central microphone button | Hotel rooms are often dimly lit. A large, prominent tap target reduces errors, especially for guests using the device from bed or at a distance. |
| Voice confirmation before submission | Guest survey findings (Chapter 4) showed a clear preference for a review step before committing a request, especially for food orders where errors are most disruptive. |
| Status-sorted request list | Guests care most about active requests. Showing in-progress items first reduces the need to scroll through completed history. |
| Department colour coding | Consistent colours (green for Housekeeping, amber for Room Service, red for Maintenance, blue for Front Desk, purple for Concierge) allow instant visual recognition without reading labels. |
| Relative timestamps ("2 mins ago") | More natural than absolute timestamps for guests checking whether a request has been acknowledged. |
| No keyboard input anywhere | The entire guest interaction is voice-driven. Removing keyboard input eliminates a barrier for guests unfamiliar with on-screen typing. |

### 6.7.2 Staff Dashboard Interface

The dashboard is a web-based single-page application served at `/dashboard`. It works on any modern browser — a desktop at the front desk, a tablet used by housekeeping, or a phone carried by maintenance.

**Figure 6.8: Staff Dashboard Interface**

*(See attached annotated screenshot — Figure 6.8)*

**Login Screen**
Staff select their department from a dropdown, enter their name, and click login. This minimal login approach means there is no account management or password reset complexity — appropriate for a prototype where the focus is on service routing, not authentication.

**Dashboard Layout**
The main view has a header showing the department name and staff name, a statistics bar with four count cards (Pending, In Progress, Completed, Cancelled), and the request queue filtered to show only the logged-in department's requests.

Each request card displays: room number (prominently highlighted), request text, intent label, status badge, timestamp, action buttons (In Progress, Complete), a department transfer dropdown, a message input field with send button, message history, and the guest rating if one has been submitted.

A connection status indicator in the bottom-right corner shows a green "Connected" badge when the WebSocket is active and a red "Disconnected" badge if the connection is lost. This is particularly important in hotel environments where Wi-Fi reliability may vary.

**Table 6.4: Staff Dashboard Design Decisions**

| Decision | Rationale |
|----------|-----------|
| Department-filtered view | Staff only see requests relevant to their role. During a busy shift, seeing all departments' requests would create unnecessary noise and slow response times. |
| Web-based, no installation | Hotels do not need to install software on staff devices. Any browser works, including on personal phones, satisfying NFR-07. |
| Real-time updates without page refresh | WebSocket ensures new requests appear instantly with a notification sound. Staff do not need to manually refresh, reducing the chance of missed requests. |
| Inline messaging | Staff can communicate with guests directly from the request card without switching to a separate interface, keeping all context in one place. |
| Room number prominently displayed | Room number is the primary identifier staff use to locate and serve a guest. It is visually emphasised on every card. |
| Connection status indicator | Hotel Wi-Fi can be intermittent. A visible status indicator allows staff to immediately identify when they are not receiving live updates. |

---

## 6.8 Summary

This chapter has presented the system design across five dimensions. The three-tier architecture separates on-device AI processing (guest tablet) from server-side coordination (FastAPI + SQLite) and staff-facing management (web dashboard), enabling deployment on low-cost hardware without GPU requirements.

The most technically significant design feature is the hybrid two-tier NLU pipeline, which combines pre-compiled rule-based keyword matching for common, unambiguous requests with MobileBERT neural inference for linguistically complex or indirect phrasings. This was not an upfront design choice — it emerged from practical observation during iterative development.

The database schema stores routing logic externally in the `intent_department_mapping` table, which allows hotels to customise department structures without modifying source code. The five-table design is straightforward to migrate to PostgreSQL for larger deployments.

The user interface designs for both the guest app and staff dashboard are driven by the specific usage context of hotel rooms and service workflows rather than generic principles. The following chapter describes the implementation of this design in detail.
