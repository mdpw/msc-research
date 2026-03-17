# Android App — Offline Hotel Voice Assistant

Android client for the MSc research project: *Low-Cost Offline Voice Assistant for Hospitality Services in Sri Lanka Using Small-Scale Neural Models.*

## Overview

The app runs entirely on-device with no internet required for core functionality. A guest taps the mic button, speaks a hotel request, and the app processes it locally using Vosk (STT) and MobileBERT (NLU), then sends a structured request to the hotel backend over the local WiFi network.

```
Mic button → AudioRecorder → VoskService (STT) → NLUService (NLU) → WebSocketService → Backend
                                                        ↓
                                               TextToSpeechService (response)
```

## Architecture

| File | Responsibility |
|------|---------------|
| `MainActivity.kt` | UI, mic button, orchestrates the pipeline |
| `AudioRecorder.kt` | Captures PCM audio from microphone |
| `VoskService.kt` | Offline STT using Vosk (`vosk-model-small-en-in-0.4`) |
| `NLUService.kt` | Intent classification using MobileBERT TFLite (18 intents) |
| `TextToSpeechService.kt` | Android TTS for spoken responses |
| `WebSocketService.kt` | Sends requests to backend over local WiFi |
| `ApiService.kt` | REST API calls to backend |
| `NetworkUtils.kt` | WiFi/network detection helpers |
| `ServerConfig.kt` | Backend host/port configuration |

## Models (in `assets/models/`)

| Path | Description | Size |
|------|-------------|------|
| `vosk-model-small-en-in-0.4/` | Vosk STT — Indian English, offline | ~50 MB |
| `nlu/hotel_mobilebert.tflite` | MobileBERT intent classifier (Model C, noise-aware) | 25.1 MB |
| `nlu/label_map.json` | 18-class intent label map | — |
| `nlu/vocab.json` | Tokenizer vocabulary | — |

## NLU — 18 Intents

`food_order`, `room_cleaning`, `towel_request`, `toiletries_request`, `maintenance`,
`concierge_taxi`, `wake_up_call`, `checkout_billing`, `pillow_request`, `blanket_request`,
`laundry_service`, `noise_complaint`, `concierge_general`, `do_not_disturb`, `emergency`,
`lighting_control`, `temperature_control`, `misc_request`

## Requirements

- Android Studio (Hedgehog or later)
- Android SDK 34, min SDK 26 (Android 8.0)
- Physical device or emulator with microphone support

## Build & Run

1. Open the `android/` folder in Android Studio
2. Wait for Gradle sync to complete
3. Connect a device (USB debugging on) or start an emulator
4. Click **Run 'app'** (▶)

## Backend Connection

The app connects to the Python backend over local WiFi. Update the host in `ServerConfig.kt` to match your backend machine's IP address before building.

## Key Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `com.alphacephei:vosk-android` | 0.3.32 | Offline STT |
| `org.tensorflow:tensorflow-lite` | 2.17.0 | MobileBERT inference |
| `com.squareup.okhttp3:okhttp` | 4.11.0 | WebSocket to backend |
| Kotlin Coroutines | 1.7.3 | Async audio/network ops |
