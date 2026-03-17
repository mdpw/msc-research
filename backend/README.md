# Backend — Hotel Voice Assistant API

FastAPI backend for the MSc research project: *Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models.*

## Overview

The backend runs on a hotel LAN server (laptop/Raspberry Pi). Android devices on the same WiFi network send voice-recognised intent requests via WebSocket. The backend routes requests to the correct department and serves a real-time staff dashboard.

```
Android App  →  WebSocket  →  FastAPI Backend  →  SQLite DB
                                    ↓
                            Staff Dashboard (browser)
```

## Structure

```
backend/
├── app/
│   ├── main.py        # FastAPI app, REST endpoints, WebSocket handlers
│   ├── models.py      # Pydantic request/response models
│   ├── database.py    # SQLite setup and queries
│   └── _init_.py
├── dashboard.html     # Staff dashboard UI (served at /dashboard)
├── test_api.py        # API endpoint tests
└── requirements.txt
```

## Setup

```bash
python -m venv hotel-backend-env
hotel-backend-env\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API root: `http://<server-ip>:8000`
- Staff dashboard: `http://<server-ip>:8000/dashboard`
- API docs: `http://<server-ip>:8000/docs`

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/api/requests` | Submit a guest request |
| GET | `/api/requests` | Get all requests |
| PUT | `/api/requests/{id}/status` | Update request status |
| WS | `/ws/dashboard` | Real-time dashboard updates |
| WS | `/ws/guest/{room}` | Per-room guest connection |

## Android Connection

Update `ServerConfig.kt` in the Android app with this server's local IP address before building.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.104.1 | Web framework |
| uvicorn | 0.24.0 | ASGI server |
| websockets | 12.0 | WebSocket support |
| pydantic | 2.5.0 | Data validation |
