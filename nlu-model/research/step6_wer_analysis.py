"""
Step 6: Word Error Rate (WER) Analysis of Vosk STT on Hospitality Dataset

Calculates WER between clean text and Vosk-transcribed text using the
paired vosk_transcriptions.csv dataset. No audio or recording required —
WER is computed directly from the paired text columns.

Output:
  - wer_report.json  — overall WER, per-intent WER, summary statistics
  - wer_report.txt   — human-readable report for thesis
"""

import pandas as pd
import json
from jiwer import wer, cer

# ── Load dataset ────────────────────────────────────────────────────────────

df = pd.read_csv("vosk_transcriptions.csv")
print(f"Loaded {len(df)} paired records")
print(f"Columns: {list(df.columns)}")

# Ensure text columns are strings and lowercase
df["clean_text"] = df["clean_text"].astype(str).str.lower().str.strip()
df["vosk_text"]  = df["vosk_text"].astype(str).str.lower().str.strip()

# ── Overall WER ──────────────────────────────────────────────────────────────

overall_wer = wer(df["clean_text"].tolist(), df["vosk_text"].tolist())
overall_cer = cer(df["clean_text"].tolist(), df["vosk_text"].tolist())

# Changed vs unchanged
changed_df   = df[df["changed"] == True]
unchanged_df = df[df["changed"] == False]

changed_pct   = len(changed_df) / len(df) * 100
unchanged_pct = len(unchanged_df) / len(df) * 100

wer_on_changed = wer(
    changed_df["clean_text"].tolist(),
    changed_df["vosk_text"].tolist()
) if len(changed_df) > 0 else 0.0

print(f"\nOverall WER : {overall_wer:.4f} ({overall_wer*100:.2f}%)")
print(f"Overall CER : {overall_cer:.4f} ({overall_cer*100:.2f}%)")
print(f"Changed     : {len(changed_df)} ({changed_pct:.1f}%)")
print(f"Unchanged   : {len(unchanged_df)} ({unchanged_pct:.1f}%)")
print(f"WER on changed sentences only: {wer_on_changed*100:.2f}%")

# ── Per-intent WER ────────────────────────────────────────────────────────────

intent_results = []
for intent in sorted(df["intent"].unique()):
    subset = df[df["intent"] == intent]
    intent_wer = wer(
        subset["clean_text"].tolist(),
        subset["vosk_text"].tolist()
    )
    intent_changed = subset["changed"].sum()
    intent_results.append({
        "intent": intent,
        "count": len(subset),
        "changed_count": int(intent_changed),
        "changed_pct": round(intent_changed / len(subset) * 100, 1),
        "wer": round(intent_wer, 4),
        "wer_pct": round(intent_wer * 100, 2)
    })

intent_results.sort(key=lambda x: x["wer"], reverse=True)

print("\nPer-Intent WER (highest to lowest):")
print(f"{'Intent':<25} {'Count':>6} {'Changed%':>9} {'WER%':>7}")
print("-" * 52)
for r in intent_results:
    print(f"{r['intent']:<25} {r['count']:>6} {r['changed_pct']:>8.1f}% {r['wer_pct']:>6.2f}%")

# ── Word-level statistics ────────────────────────────────────────────────────

total_words = df["clean_text"].apply(lambda x: len(x.split())).sum()
avg_words   = df["clean_text"].apply(lambda x: len(x.split())).mean()

# ── Save JSON report ─────────────────────────────────────────────────────────

report = {
    "dataset": "vosk_transcriptions.csv",
    "total_pairs": len(df),
    "total_words_reference": int(total_words),
    "avg_words_per_utterance": round(avg_words, 2),
    "overall": {
        "wer": round(overall_wer, 4),
        "wer_pct": round(overall_wer * 100, 2),
        "cer": round(overall_cer, 4),
        "cer_pct": round(overall_cer * 100, 2),
        "changed_sentences": int(len(changed_df)),
        "changed_pct": round(changed_pct, 1),
        "unchanged_sentences": int(len(unchanged_df)),
        "wer_on_changed_only": round(wer_on_changed * 100, 2)
    },
    "per_intent": intent_results
}

with open("wer_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved: wer_report.json")

# ── Save readable text report ─────────────────────────────────────────────────

lines = [
    "=" * 60,
    "VOSK STT WORD ERROR RATE (WER) ANALYSIS",
    "Dataset: vosk_transcriptions.csv (hospitality domain)",
    "=" * 60,
    "",
    "OVERALL RESULTS",
    "-" * 40,
    f"Total utterance pairs   : {len(df):,}",
    f"Total reference words   : {int(total_words):,}",
    f"Avg words per utterance : {avg_words:.1f}",
    "",
    f"Word Error Rate (WER)   : {overall_wer*100:.2f}%",
    f"Char Error Rate (CER)   : {overall_cer*100:.2f}%",
    "",
    f"Sentences changed by Vosk    : {len(changed_df):,} ({changed_pct:.1f}%)",
    f"Sentences unchanged by Vosk  : {len(unchanged_df):,} ({unchanged_pct:.1f}%)",
    f"WER on changed sentences only: {wer_on_changed*100:.2f}%",
    "",
    "PER-INTENT WER (highest to lowest)",
    "-" * 40,
    f"{'Intent':<25} {'Count':>6} {'Changed%':>9} {'WER%':>7}",
    "-" * 52,
]
for r in intent_results:
    lines.append(
        f"{r['intent']:<25} {r['count']:>6} {r['changed_pct']:>8.1f}% {r['wer_pct']:>6.2f}%"
    )
lines += [
    "",
    "NOTE: WER is computed from TTS-synthesised speech passed through",
    "Vosk (vosk-model-small-en-in-0.4). This reflects controlled STT",
    "noise rather than real human speech. Actual WER on real speech",
    "may differ. See thesis limitations section for discussion.",
    "=" * 60,
]

with open("wer_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved: wer_report.txt")
print("\nDone. Use wer_report.json values in your thesis.")
