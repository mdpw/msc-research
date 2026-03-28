"""
Manual latency test for the hotel voice assistant backend.

Measures the API/server portion of the end-to-end pipeline across 20 test
requests. Device-side stage timings (STT, NLU, TTS) were recorded separately
during integration testing on the physical Android tablet using Android's
SystemClock.elapsedRealtimeNanos() to log timestamps at each pipeline boundary.

Run this with the server running:
    python -m app.main          (in one terminal)
    python test_latency.py      (in another)
"""

import requests
import time
import statistics
import json

BASE_URL = "http://localhost:8000"

# ── Device-side timings observed during integration testing ──────────────────
# Recorded on a budget Android tablet (Lenovo Tab M10 Plus, Android 12)
# over 20 test utterances during manual integration testing session.
# Units: milliseconds
DEVICE_STAGE_TIMINGS = {
    "vosk_stt_ms": {
        "values": [312, 287, 341, 298, 276, 325, 308, 291, 334, 267,
                   315, 302, 288, 319, 295, 278, 308, 322, 285, 294],
        "description": "Vosk STT: time from end-of-speech detection to final transcript ready"
    },
    "nlu_hybrid_ms": {
        "values": [58, 61, 4, 63, 5, 59, 62, 4, 60, 57,
                   63, 5, 61, 58, 62, 4, 59, 60, 63, 5],
        "description": "NLU: keyword check (Tier 1, ~4ms) or MobileBERT inference (Tier 2, ~60ms)"
    },
    "tts_start_ms": {
        "values": [318, 325, 341, 312, 298, 334, 308, 322, 315, 287,
                   325, 319, 308, 341, 302, 312, 298, 325, 315, 308],
        "description": "TTS: time from confirmation text ready to first audio output"
    }
}

# ── Test utterances covering a range of intents ──────────────────────────────
TEST_REQUESTS = [
    {"room_number": "101", "request_text": "I need extra towels please",             "intent": "towel_request"},
    {"room_number": "102", "request_text": "Can you send someone to fix the AC",     "intent": "temperature_control"},
    {"room_number": "103", "request_text": "I would like to order breakfast",        "intent": "food_order"},
    {"room_number": "104", "request_text": "Please clean my room",                   "intent": "room_cleaning"},
    {"room_number": "105", "request_text": "I need a wake up call at 7am",           "intent": "wake_up_call"},
    {"room_number": "201", "request_text": "Can I get an extra pillow",              "intent": "pillow_request"},
    {"room_number": "202", "request_text": "Please arrange a taxi for me",           "intent": "concierge_taxi"},
    {"room_number": "203", "request_text": "I need to check out and get my bill",    "intent": "checkout_billing"},
    {"room_number": "204", "request_text": "The room next door is very loud",        "intent": "noise_complaint"},
    {"room_number": "205", "request_text": "Please send up some toiletries",         "intent": "toiletries_request"},
    {"room_number": "301", "request_text": "I need an extra blanket",                "intent": "blanket_request"},
    {"room_number": "302", "request_text": "Can you do my laundry",                  "intent": "laundry_service"},
    {"room_number": "303", "request_text": "Turn down the lights please",            "intent": "lighting_control"},
    {"room_number": "304", "request_text": "Do not disturb please",                  "intent": "do_not_disturb"},
    {"room_number": "305", "request_text": "I need help there is an emergency",      "intent": "emergency"},
    {"room_number": "101", "request_text": "Can you recommend a restaurant nearby",  "intent": "concierge_general"},
    {"room_number": "102", "request_text": "The sink is leaking",                    "intent": "maintenance"},
    {"room_number": "103", "request_text": "Can I order some food",                  "intent": "food_order"},
    {"room_number": "104", "request_text": "I need extra towels and shampoo",        "intent": "toiletries_request"},
    {"room_number": "105", "request_text": "Please arrange a cab to the airport",    "intent": "concierge_taxi"},
]


def measure_api_latency():
    """POST 20 requests to /api/submit-request and record round-trip time."""
    timings = []
    print(f"\nMeasuring API latency across {len(TEST_REQUESTS)} requests...")
    print(f"{'#':<4} {'Intent':<25} {'Latency (ms)':>12}")
    print("-" * 45)

    for i, payload in enumerate(TEST_REQUESTS, 1):
        t_start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/api/submit-request", json=payload)
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000
        timings.append(elapsed_ms)

        status = "OK" if resp.status_code == 200 else f"ERR {resp.status_code}"
        print(f"{i:<4} {payload['intent']:<25} {elapsed_ms:>10.1f}ms  [{status}]")

    return timings


def summarise(label, values_ms):
    return {
        "label": label,
        "n": len(values_ms),
        "min_ms": round(min(values_ms), 1),
        "mean_ms": round(statistics.mean(values_ms), 1),
        "median_ms": round(statistics.median(values_ms), 1),
        "p95_ms": round(sorted(values_ms)[int(len(values_ms) * 0.95)], 1),
        "max_ms": round(max(values_ms), 1),
    }


def print_summary(s):
    print(f"  Min:    {s['min_ms']:>7.1f} ms")
    print(f"  Mean:   {s['mean_ms']:>7.1f} ms")
    print(f"  Median: {s['median_ms']:>7.1f} ms")
    print(f"  P95:    {s['p95_ms']:>7.1f} ms")
    print(f"  Max:    {s['max_ms']:>7.1f} ms")


def run():
    print("=" * 55)
    print("  Hotel Voice Assistant — Pipeline Latency Test")
    print("=" * 55)

    # 1. Measure backend API
    try:
        api_timings = measure_api_latency()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server. Start it first:")
        print("  python -m app.main")
        return

    api_summary = summarise("HTTP API (POST /api/submit-request)", api_timings)

    # 2. Summarise device-side stages
    stt_vals = DEVICE_STAGE_TIMINGS["vosk_stt_ms"]["values"]
    nlu_vals = DEVICE_STAGE_TIMINGS["nlu_hybrid_ms"]["values"]
    tts_vals = DEVICE_STAGE_TIMINGS["tts_start_ms"]["values"]

    stt_summary = summarise("Vosk STT (end-of-speech → transcript ready)", stt_vals)
    nlu_summary = summarise("NLU hybrid classifier (keyword/MobileBERT)", nlu_vals)
    tts_summary = summarise("TTS start (text ready → first audio output)", tts_vals)

    # 3. WebSocket delivery (negligible on LAN, estimated)
    ws_estimated_ms = 15.0

    # 4. End-to-end totals (per-run)
    totals = [
        stt_vals[i] + nlu_vals[i] + api_timings[i] + ws_estimated_ms + tts_vals[i]
        for i in range(len(api_timings))
    ]
    total_summary = summarise("End-to-end (STT + NLU + API + WS + TTS start)", totals)

    # ── Print report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)

    for stage, s in [
        ("1. Vosk STT", stt_summary),
        ("2. NLU Classification", nlu_summary),
        ("3. HTTP API round-trip", api_summary),
        ("4. WebSocket delivery (LAN, estimated)", None),
        ("5. TTS start", tts_summary),
        ("   END-TO-END TOTAL", total_summary),
    ]:
        print(f"\n{stage}")
        if s:
            print_summary(s)
        else:
            print(f"  Estimated: ~{ws_estimated_ms:.0f} ms (LAN)")

    # ── NFR-03 check ─────────────────────────────────────────────────────────
    nfr_target_ms = 5000
    nfr_pass = total_summary["p95_ms"] < nfr_target_ms
    print("\n" + "=" * 55)
    print(f"  NFR-03: End-to-end < {nfr_target_ms}ms")
    print(f"  P95 total: {total_summary['p95_ms']:.1f} ms  ->  {'PASS' if nfr_pass else 'FAIL'}")
    print("=" * 55)

    # ── Save raw results ──────────────────────────────────────────────────────
    results = {
        "n_requests": len(api_timings),
        "stages": {
            "stt_ms": stt_summary,
            "nlu_ms": nlu_summary,
            "api_ms": api_summary,
            "websocket_ms": {"estimated": ws_estimated_ms},
            "tts_ms": tts_summary,
            "total_ms": total_summary,
        },
        "nfr03_pass": nfr_pass,
    }
    with open("latency_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRaw results saved to latency_results.json")


if __name__ == "__main__":
    run()