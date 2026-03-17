# Research Pipeline: STT Error-Aware MobileBERT Training

## What This Does

Investigates the accuracy gap when MobileBERT receives Vosk-transcribed text
vs clean text, and shows that noise-aware training closes this gap.

## Run Order

```
step1_expand_dataset.py        → expand to ~10k clean records
step2_generate_vosk_noise.py   → generate Vosk-transcribed noisy versions
step3_train_three_models.py    → train Model A, B, C
step4_evaluate.py              → measure the gap and improvement
step5_convert_best_model.py    → deploy best model to Android
```

## Setup

```bash
pip install anthropic gtts vosk transformers torch tensorflow tf2onnx
pip install scikit-learn pandas matplotlib seaborn
```

Download Vosk model:
- Go to https://alphacephei.com/vosk/models
- Download `vosk-model-en-us-0.22` (most accurate, matches Android model)
- Extract into this folder as `vosk-model/`

Set Claude API key (for step 1):
```bash
set ANTHROPIC_API_KEY=your-key-here
```

## Expected Output (Thesis Table)

| Model              | Test Input  | Accuracy | F1 Macro |
|--------------------|-------------|----------|----------|
| A (clean-trained)  | Clean text  | ~92%     | ~0.92    |  ← baseline
| A (clean-trained)  | Vosk output | ~78-82%  | ~0.79    |  ← THE GAP
| B (vosk-trained)   | Vosk output | ~84-86%  | ~0.85    |
| C (mixed-trained)  | Vosk output | ~88-91%  | ~0.89    |  ← your fix

The gap between row 1 and row 2 is your research finding.
The improvement in row 4 is your contribution.

## Timeline (from March 15)

| Week        | Task                          |
|-------------|-------------------------------|
| Mar 15-22   | Step 1 + Step 2               |
| Mar 23-29   | Step 3 + Step 4               |
| Mar 30-Apr 5 | Step 5 + write results chapter |
| Apr 6-12    | Update literature review       |
| Apr 13-20   | Buffer / submission prep       |
