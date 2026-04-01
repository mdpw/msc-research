# CHAPTER 6: DESIGN

## 6.1 Introduction

This chapter covers the detailed system design of the low-cost offline voice assistant prototype. The technologies chosen in Chapter 5 are put together here into a working architecture. It walks through the three-tier system structure, the voice processing pipeline, the database schema, the API design, and both user interfaces. Every design choice was made to satisfy the four viability constraints — offline operation, on-device privacy, acceptable latency, and low hardware cost — and where a choice is not immediately obvious, the reasoning is explained alongside the technical detail.

---

## 6.2 System Architecture

The whole system is built around one core idea: all the heavy AI work — speech recognition and intent classification — happens on the guest's tablet. The server only handles coordination, storing requests, and routing them to the right department. This means the server needs no GPU, and can run on any basic laptop or desktop the hotel already owns.

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

Each guest room has one Android tablet running the app. The tablet handles the entire AI pipeline on its own — audio capture, speech recognition via Vosk, NLU classification, and TTS confirmation. It communicates with the server over HTTP to submit requests and via WebSocket to receive real-time status updates.

The key components on the device are:
- `AudioRecorder` — captures 16kHz, 16-bit PCM mono audio in 4,096-byte chunks, with voice activity detection based on Root Mean Square (RMS) energy threshold (0.02)
- `VoskService` — on-device speech-to-text using `vosk-model-small-en-in-0.4` (~36MB)
- `NLUService` — hybrid two-tier classification pipeline (rule-based keywords → MobileBERT TFLite)
- `TextToSpeechService` — Android native TTS for voice confirmations and status announcements
- `ApiService` — HTTP Representational State Transfer (REST) client (OkHttp 4.11.0) for submitting requests and retrieving history
- `WebSocketService` — persistent WebSocket connection for real-time updates from the server

**Tier 2 — Hotel Server**

The server is a FastAPI application that runs on any PC or laptop on the hotel's local network. It exposes 17 endpoints (14 HTTP + 2 WebSocket + 1 dashboard serve), manages a SQLite database, routes requests to departments, and pushes events to connected clients via WebSocket.

Routing uses a three-level hierarchy: first, intent-based lookup in the `intent_department_mapping` table; then a keyword-analysis fallback if the intent is missing or unrecognised; and finally Front Desk as the default if neither method finds a match.

**Tier 3 — Staff Dashboard**

The staff dashboard is a web-based single-page app served by the FastAPI backend at `/dashboard`. No installation is needed — any staff member can open it in a browser on a desktop, tablet, or phone. It connects to the `/ws/dashboard` WebSocket endpoint and shows each department's own filtered view of the live request queue.

---

## 6.3 Voice Processing Pipeline Design

The voice processing pipeline covers the full guest interaction — from the moment the microphone button is pressed to the request appearing on the staff dashboard.

### 6.3.1 Pipeline Overview

**Figure 6.2: Voice Request Pipeline Sequence**

*(See attached sequence diagram — Figure 6.2)*

The pipeline has three major phases:

**Phase 1 — Audio Capture and Transcription**

When the guest presses the microphone button, `AudioRecorder` starts capturing 16kHz Pulse Code Modulation (PCM) audio in 4,096-byte chunks and feeds them to Vosk in real time. A voice activity detector (VAD) monitors the RMS energy of each chunk against a threshold of 0.02. Recording stops when 1,500ms of silence is detected or the 10-second maximum is reached. The Vosk transcription is then cleaned — filler words and greetings like "Hi Sera" are removed, and the text is lowercased.

**Phase 2 — Intent Classification**

The cleaned text first goes through a cancel detection check using a regex pattern for "cancel order [number]". If it matches, the request ID is extracted, a voice confirmation plays, and the cancellation is sent to the server. If not, the text moves into the hybrid NLU pipeline.

**Phase 3 — Confirmation and Submission**

Once an intent is classified above the confidence threshold (0.60 for the neural model; 0.99 for keyword matches), the app reads the request back to the guest via TTS: *"You'd like [request description]. Shall I submit this?"* The guest's spoken yes or no is captured. On confirmation, an HTTP POST is sent to the server, which stores the request, routes it to the right department, and broadcasts a `new_request` WebSocket event to all connected staff dashboards.

### 6.3.2 Hybrid NLU Pipeline — Core Design Feature

A key design feature for ensuring NLU reliability under real pipeline conditions is the two-stage hybrid NLU pipeline. It was not planned from the start — it came out of a practical problem noticed during development: purely neural classification was giving unexpectedly low confidence on simple, clear requests. For example, "I need towels" was being classified as `pillow_request` with only 0.72 confidence after small Vosk transcription variations.

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

The keyword dictionary covers 17 intent categories (everything except `misc_request`). Each intent is mapped to multi-word contextual phrases rather than single words, which avoids false positives. For example, `food_order` uses phrases like "bottled water", "glass of water", and "room service" — not just "water" on its own, which could appear in a completely different context like "the water in the bathroom is leaking" (a `maintenance` request). The patterns are pre-compiled at app startup to avoid compilation overhead at inference time. A keyword match returns the intent immediately with a fixed confidence of 0.99, skipping the neural model entirely.

**Tier 2 — MobileBERT Neural Inference**

When there is no keyword match, the text is tokenised using the BERT uncased vocabulary (30,522 tokens), padded or truncated to 32 tokens, and passed through the `hotel_mobilebert_v2.tflite` interpreter (26MB). The output logits are converted to probabilities via softmax across all 18 intent classes, and the highest-probability class is taken as the predicted intent.

The 0.60 confidence threshold was determined through testing during development. Setting it too high meant valid but unusually phrased requests would get rejected; too low, and misclassified requests would end up with staff. The chosen value filters out genuinely ambiguous input while still accepting natural speech variation.

---

## 6.4 Database Design

The database uses SQLite and has five tables. The schema is deliberately simple, but structured so that migrating to PostgreSQL for a larger deployment would be straightforward — no schema changes required.

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

**`rooms`** — Pre-seeded with 15 rooms across three floors (101–105, 201–205, 301–305) with Standard, Deluxe, and Suite types. The `room_number` field is used as the identifier everywhere — on request cards, WebSocket channel names, and the history endpoint.

**`departments`** — Pre-seeded with five departments: Housekeeping, Room Service, Maintenance, Front Desk, and Concierge. Keeping departments in a table rather than hardcoded in the code means a hotel can rename or restructure without touching any application logic.

**`intent_department_mapping`** — Maps each of the 18 intent categories to a department. This is the first place the server looks when routing a submitted request. Externalising routing into a database table means a hotel can, for example, redirect `concierge_taxi` to Front Desk (if they have no dedicated concierge) by changing one row, not redeploying the server.

**`requests`** — The central table. Every service request is stored here from submission to completion. The `status` field tracks the lifecycle (`pending` → `in_progress` → `completed` or `cancelled`). The `rating` field is nullable and only filled in if the guest rates the service. The `completed_at` timestamp is only recorded when status reaches `completed`.

**`staff_messages`** — Stores messages sent from staff to guest rooms, linked to the relevant request. A single request can have multiple messages.

---

## 6.5 API and Communication Design

### 6.5.1 REST API Endpoints

The backend exposes 15 HTTP endpoints following RESTful conventions, plus 2 WebSocket endpoints and 1 endpoint to serve the dashboard HTML — 17 in total.

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

All endpoints use JSON for both requests and responses. Pydantic models handle input validation automatically and return clear error messages when data is malformed.

### 6.5.2 WebSocket Communication Design

Two WebSocket endpoints handle real-time communication between the server and clients. The guest and dashboard channels are kept separate by design — each client only gets the events relevant to it.

**Guest channel: `/ws/guest/{room_number}`**

Each guest device keeps a persistent WebSocket connection identified by its room number. The server pushes three types of events:

| Event Type | Fields | Trigger |
|------------|--------|---------|
| `status_update` | `request_id`, `status`, `message` | Staff updates request status |
| `department_update` | `request_id`, `message` | Request transferred to different department |
| `staff_message` | `request_id`, `message`, `staff_name` | Staff sends a message to the room |

On the Android app, all incoming WebSocket events are read aloud via TTS, so guests are notified of updates without having to look at the screen.

**Dashboard channel: `/ws/dashboard`**

The staff dashboard connects to a single shared WebSocket endpoint. Every connected dashboard instance gets the same event stream — when one staff member updates a request, all other open dashboards reflect it instantly.

| Event Type | Fields | Trigger |
|------------|--------|---------|
| `initial` | `requests[]` (full list) | On first connection — gives the dashboard its full current state |
| `new_request` | `id`, `room_number`, `request_text`, `intent`, `department`, `status`, `created_at`, `rating` | Guest submits a new request |
| `status_update` | `request_id`, `status` | Request status changes |
| `department_update` | `request_id`, `department` | Request transferred |
| `staff_message` | `request_id`, `message`, `staff_name` | Message added |
| `rating_update` | `request_id`, `rating` | Guest submits a rating |

The `initial` event is sent as soon as a dashboard client connects, giving it the full current state of all requests without needing a separate HTTP call.

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

A request moves through a defined set of states from submission to completion. Invalid transitions are blocked in the application logic — for example, a `completed` request cannot be moved back to `in_progress`.

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

Every state transition triggers a WebSocket broadcast — to the relevant guest device channel and to all connected dashboard instances at the same time.

---

## 6.6 Department Routing Design

When a request arrives via `POST /api/submit-request`, the server decides which department should handle it using a three-level hierarchy:

1. **Intent-based routing (primary)** — The request includes an `intent` field classified on-device. The server looks this up in the `intent_department_mapping` table and routes accordingly.

2. **Keyword fallback (secondary)** — If the intent is missing or not found in the table, the server analyses the `request_text` using its own keyword lists per department (separate from the on-device NLU).

3. **Default routing** — If neither method finds a match, the request goes to Front Desk.

**Table 6.2: Intent-to-Department Routing**

| Department | Intents | Colour |
|------------|---------|--------|
| Housekeeping | room_cleaning, towel_request, toiletries_request, blanket_request, pillow_request, laundry_service, do_not_disturb | #10b981 (green) |
| Room Service | food_order | #f59e0b (amber) |
| Maintenance | maintenance, temperature_control, lighting_control | #ef4444 (red) |
| Front Desk | wake_up_call, checkout_billing, noise_complaint, emergency, misc_request | #3b82f6 (blue) |
| Concierge | concierge_general, concierge_taxi | #8b5cf6 (purple) |

Staff can also manually transfer a request to a different department using a dropdown on the dashboard. The request disappears from the original department's queue and appears instantly in the new one via WebSocket.

---

## 6.7 User Interface Design

Both interfaces were designed around the specific context they will be used in, not generic HCI principles. The guest app needs to work for someone in a dimly lit hotel room, possibly from bed, with no training whatsoever. The staff dashboard needs to hold up during a busy shift with multiple departments using it at the same time.

### 6.7.1 Guest Application Interface

The app is built with Jetpack Compose and Material Design 3. The entire interface is a single screen divided into three functional sections.

**Figure 6.7: Guest Application Interface**

*(See attached annotated screenshot — Figure 6.7)*

**Section 1 — Room Information Bar (top)**
Shows the room number, Wi-Fi connection status, and the configured server network profile. Both guests and maintenance staff can see the key context at a glance without navigating any menus.

**Section 2 — Voice Interaction Area (centre, prominent)**
The centrepiece is a large circular microphone button. While recording, it pulses with an animation and a 20-bar audio visualiser shows the captured signal level. A status line below it ("Tap to speak" → "Listening..." → "Processing...") gives the guest continuous feedback. After each interaction, the transcription, recognised intent, confidence score, and department colour are displayed.

**Section 3 — Request History (bottom, scrollable)**
A scrollable list of the room's requests, sorted by priority: in-progress at the top, then pending, completed, and cancelled at the bottom. Each card shows the request ID, a colour-coded status badge, the request text (truncated to two lines), department colour, and a relative timestamp ("2 mins ago"). Pending and in-progress requests have a cancel button; completed ones have a 5-star rating selector.

**Table 6.3: Guest UI Design Decisions**

| Decision | Rationale |
|----------|-----------|
| Large central microphone button | Hotel rooms are often dimly lit. A large, prominent tap target reduces errors, especially for guests using the device from bed or at a distance. |
| Voice confirmation before submission | Guest survey findings (Chapter 4) showed a clear preference for a review step before committing a request, especially for food orders where errors are most disruptive. |
| Status-sorted request list | Guests care most about active requests. Showing in-progress items first reduces the need to scroll through completed history. |
| Department colour coding | Consistent colours (green for Housekeeping, amber for Room Service, red for Maintenance, blue for Front Desk, purple for Concierge) allow instant visual recognition without reading labels. |
| Relative timestamps ("2 mins ago") | More natural than absolute timestamps for guests checking whether a request has been acknowledged. |
| No keyboard input anywhere | The entire guest interaction is voice-driven. Removing the keyboard lowers the barrier for guests who are not comfortable with on-screen typing. |

### 6.7.2 Staff Dashboard Interface

The dashboard is a web-based single-page app served at `/dashboard`. It works on any modern browser — a desktop at the front desk, a tablet used by housekeeping, or a phone a maintenance engineer carries around.

**Figure 6.8: Staff Dashboard Interface**

*(See attached annotated screenshot — Figure 6.8)*

**Login Screen**
Staff pick their department from a dropdown, enter their name, and click login. This minimal approach keeps things simple — there is no account management or password reset to worry about, which is reasonable for a prototype focused on service routing rather than authentication.

**Dashboard Layout**
The main view shows a header with the department and staff name, a statistics bar with four count cards (Pending, In Progress, Completed, Cancelled), and the request queue filtered to the logged-in department's requests only.

Each request card shows: room number (prominently highlighted), request text, intent label, status badge, timestamp, action buttons (In Progress, Complete), a department transfer dropdown, a message input with a send button, message history, and the guest rating if one has been submitted.

A connection status indicator in the bottom-right corner shows a green "Connected" badge when the WebSocket is live and a red "Disconnected" badge if it drops. This is especially useful in hotel settings where Wi-Fi can be unreliable.

**Table 6.4: Staff Dashboard Design Decisions**

| Decision | Rationale |
|----------|-----------|
| Department-filtered view | Staff only see requests relevant to their role. During a busy shift, seeing all departments' requests would create unnecessary noise and slow response times. |
| Web-based, no installation | Hotels do not need to install software on staff devices. Any browser works, including on personal phones, satisfying NFR-07. |
| Real-time updates without page refresh | WebSocket ensures new requests appear instantly with a notification sound. Staff do not need to manually refresh, reducing the chance of missed requests. |
| Inline messaging | Staff can communicate with guests directly from the request card without switching to a separate interface, keeping all context in one place. |
| Room number prominently displayed | Room number is the primary identifier staff use to locate and serve a guest. It is visually emphasised on every card. |
| Connection status indicator | Hotel Wi-Fi can be intermittent. A visible status indicator lets staff know straight away if they have stopped receiving live updates. |

---

## 6.8 Summary

This chapter has walked through the system design from architecture to interface. The three-tier structure separates on-device AI processing (guest tablet) from server-side coordination (FastAPI + SQLite) and the staff-facing web dashboard, which means the system can be deployed on low-cost hardware with no GPU.

The most technically interesting part is the hybrid NLU pipeline. It was not in the original plan — it came out of a practical problem found during development, where purely neural classification was giving low confidence on simple requests. Combining pre-compiled keyword matching for clear-cut intents with MobileBERT for more complex ones turned out to be a better solution than either approach alone.

The database stores routing logic in the `intent_department_mapping` table rather than in code, which means hotels can reconfigure department assignments without touching any source files. The five-table schema is also straightforward to migrate to PostgreSQL if needed.

Both interfaces were shaped by real-world usage scenarios — hotel rooms and busy service workflows — rather than general design guidelines. The next chapter covers the actual implementation of this design.
