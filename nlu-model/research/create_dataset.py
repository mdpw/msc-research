"""
Dataset Creation Script
=======================
Generates a clean hotel intent classification dataset matching the exact
output format of vosk-model-small-en-in-0.4 (Indian English).

Dataset specification:
  - 18 hotel service intent categories
  - 560 utterances per intent
  - Total: 10,080 records
  - Format: lowercase, no punctuation, no digits, natural spoken English
  - Matches Vosk STT output: contractions without apostrophes (cant, dont, im),
    compound words split as Vosk hears them (house keeping, check out, a c, t v)

Output:
  new_hotel_dataset.csv

Usage:
  $env:ANTHROPIC_API_KEY = "your-key-here"
  python create_dataset.py

Cost: approximately $2-3 USD using Claude Haiku
"""

import os
import time
import pandas as pd
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_PER_INTENT = 560                       # utterances per intent
BATCH_SIZE        = 60                        # utterances requested per API call
MAX_ATTEMPTS      = 10                        # max API retries per intent
OUTPUT_FILE       = 'new_hotel_dataset.csv'
CHECKPOINT_FILE   = 'new_hotel_dataset_checkpoint.csv'
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# ── Intent definitions ────────────────────────────────────────────────────────
# 18 hotel service intents with descriptions and seed examples
# Seeds are written in Vosk output format (no punctuation, no apostrophes)
INTENTS = {
    "blanket_request": {
        "desc": "Guest requesting extra blankets or duvets because they are cold",
        "seeds": [
            "can i get an extra blanket",
            "the room is cold send a duvet please",
            "i need another blanket for tonight",
            "could you bring a warm blanket to my room",
            "we need two more blankets please",
        ]
    },
    "checkout_billing": {
        "desc": "Guest asking about check out time their bill or requesting late check out",
        "seeds": [
            "what time is check out",
            "can i have a copy of my bill",
            "i would like a late check out please",
            "can i extend my stay by one more night",
            "what are the charges on my account",
        ]
    },
    "concierge_general": {
        "desc": "Guest asking for recommendations bookings or general concierge help",
        "seeds": [
            "what restaurants are nearby",
            "can you recommend a good place to eat",
            "what time does the swimming pool open",
            "can you book a table for two tonight",
            "is there a pharmacy close to the hotel",
        ]
    },
    "concierge_taxi": {
        "desc": "Guest requesting a taxi cab or transport arrangement",
        "seeds": [
            "i need a taxi to the airport",
            "can you arrange a cab for eight in the morning",
            "please book a car for me",
            "i need transport to the city center",
            "can you call a taxi right now",
        ]
    },
    "do_not_disturb": {
        "desc": "Guest asking not to be disturbed or to skip house keeping",
        "seeds": [
            "please do not disturb",
            "skip house keeping today",
            "i dont want anyone to come to my room",
            "please leave me alone for a few hours",
            "no service needed today thank you",
        ]
    },
    "emergency": {
        "desc": "Guest reporting a safety issue locked out or needing urgent help",
        "seeds": [
            "i am locked out of my room",
            "there is smoke coming from the corridor",
            "i need urgent help right away",
            "someone is trying to enter my room",
            "i feel very unsafe please send security",
        ]
    },
    "food_order": {
        "desc": "Guest ordering food or drinks to be delivered to their room",
        "seeds": [
            "i would like to order room service",
            "can i get a cheese sandwich sent to my room",
            "please bring me a coffee and some toast",
            "i would like to see the room service menu",
            "can you send up two bottles of water",
        ]
    },
    "laundry_service": {
        "desc": "Guest requesting laundry collection washing or ironing service",
        "seeds": [
            "i need my clothes washed please",
            "can you collect my laundry",
            "i need this shirt ironed for tomorrow morning",
            "when will my laundry be returned",
            "please send someone to pick up my clothes",
        ]
    },
    "lighting_control": {
        "desc": "Guest asking to adjust the room lighting brightness or switch lights on or off",
        "seeds": [
            "can you dim the lights please",
            "turn off the bedroom light",
            "make the room a bit brighter",
            "switch on the reading light",
            "the lights are too bright can you lower them",
        ]
    },
    "maintenance": {
        "desc": "Guest reporting something broken or not working in the room",
        "seeds": [
            "the a c is not working",
            "the tap in the bathroom is leaking",
            "my t v has no picture",
            "the toilet is not flushing properly",
            "the hair dryer in the bathroom is broken",
        ]
    },
    "misc_request": {
        "desc": "General guest requests such as hangers adapters umbrellas or stationery",
        "seeds": [
            "can i get some extra hangers please",
            "i need a power adapter for my laptop",
            "could you send up an umbrella",
            "i need a pen and some paper",
            "can you bring me a bottle opener",
        ]
    },
    "noise_complaint": {
        "desc": "Guest complaining about noise from neighbours or the corridor",
        "seeds": [
            "the people next door are very loud",
            "there is a lot of noise coming from the room above",
            "can you please ask the guests nearby to be quiet",
            "the noise from the corridor is keeping me awake",
            "someone is playing loud music near my room",
        ]
    },
    "pillow_request": {
        "desc": "Guest requesting extra pillows or a different type of pillow",
        "seeds": [
            "can i get an extra pillow please",
            "i need a softer pillow",
            "send two more pillows to my room",
            "could you replace the pillow with a firmer one",
            "i would like one more pillow for the bed",
        ]
    },
    "room_cleaning": {
        "desc": "Guest requesting room cleaning house keeping or bathroom cleaning",
        "seeds": [
            "can someone come and clean my room",
            "the bathroom needs cleaning please",
            "please send house keeping to my room",
            "i would like my room serviced today",
            "the floor needs mopping can you send someone",
        ]
    },
    "temperature_control": {
        "desc": "Guest asking to adjust air conditioning heating or room temperature",
        "seeds": [
            "the room is too cold can you turn up the heating",
            "please set the a c to a warmer temperature",
            "can you make the room a little cooler",
            "the a c is too strong please reduce it",
            "i would like the temperature adjusted in my room",
        ]
    },
    "toiletries_request": {
        "desc": "Guest requesting toiletries such as shampoo soap toothpaste or conditioner",
        "seeds": [
            "can you send more shampoo please",
            "we need some soap in the bathroom",
            "could you bring toothpaste to my room",
            "i need conditioner and body wash",
            "the toiletries in the bathroom have run out",
        ]
    },
    "towel_request": {
        "desc": "Guest requesting fresh or extra towels for bathroom or pool",
        "seeds": [
            "can i get some fresh towels please",
            "send extra bath towels to my room",
            "we need more towels in the bathroom",
            "could you replace the towels they are wet",
            "i need two more hand towels please",
        ]
    },
    "wake_up_call": {
        "desc": "Guest requesting a wake up call or alarm call for a specific time",
        "seeds": [
            "please wake me up at six in the morning",
            "i need a wake up call at seven thirty",
            "can you set an alarm call for five am",
            "wake me up tomorrow at half past six please",
            "i would like a morning call at eight o clock",
        ]
    },
}

# ── Text format rules injected into every API prompt ─────────────────────────
# Ensures all generated text matches vosk-model-small-en-in-0.4 output format
VOSK_FORMAT_RULES = """
CRITICAL - Write every sentence EXACTLY as vosk-model-small-en-in-0.4 would output it:

1. All lowercase - no capital letters
2. No punctuation - no commas, full stops, apostrophes, question marks, exclamation marks
3. Contractions without apostrophes: cant, dont, im, its, wont, wouldnt, youre, theres, ive
4. Numbers as spoken words: six thirty, five am, half past seven, twenty two degrees
5. Hotel compound words split as Vosk transcribes them:
   AC -> a c | TV -> t v | WiFi -> wi fi | housekeeping -> house keeping
   checkout -> check out | minibar -> mini bar | hairdryer -> hair dryer
6. Mix of casual short forms (need towels, ac broken) and polite longer forms
7. Between 3 and 15 words per sentence
8. Every sentence must be unique"""


# ── Cleaning and validation ───────────────────────────────────────────────────
STRIP_CHARS = str.maketrans('', '', '.,!?;:\'"()-/\\@#$%^&*+=[]{}|<>0123456789')

def clean(text: str) -> str:
    return ' '.join(text.strip().lower().translate(STRIP_CHARS).split())

def is_valid(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 3 or len(words) > 20:
        return False
    if any(c in '.,!?;:\'"()0123456789' for c in text):
        return False
    if not any(len(w) > 2 for w in words):
        return False
    return True


# ── API call ──────────────────────────────────────────────────────────────────
def generate_batch(intent: str, desc: str, seeds: list,
                   count: int, used: set) -> list:
    prompt = f"""You are generating speech recognition output training data for a hotel voice assistant.

Intent: {intent}
Description: {desc}

Example sentences (already in correct format):
{chr(10).join(f'- {s}' for s in seeds)}

Generate {count} NEW sentences a hotel guest might say for this intent.

{VOSK_FORMAT_RULES}

Return ONLY the sentences, one per line, no numbers no bullets no labels."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    results = []
    for line in response.content[0].text.strip().split('\n'):
        c = clean(line.lstrip('•-*0123456789. '))
        if is_valid(c) and c not in used:
            results.append(c)
            used.add(c)
    return results


# ── Main generation ───────────────────────────────────────────────────────────
all_rows   = []
used_texts = set()

# Resume from checkpoint if script was interrupted
if os.path.exists(CHECKPOINT_FILE):
    ckpt         = pd.read_csv(CHECKPOINT_FILE)
    all_rows     = ckpt.to_dict('records')
    used_texts   = set(ckpt['text'].tolist())
    done_intents = set(ckpt['intent'].unique())
    done_counts  = ckpt['intent'].value_counts()
    # Include intents that still need top-up
    partial = {i for i in done_intents if done_counts.get(i, 0) < TARGET_PER_INTENT}
    done_intents -= partial
    print(f"Resuming from checkpoint: {len(all_rows)} records")
else:
    done_intents = set()

print(f"\nTarget  : {TARGET_PER_INTENT} utterances x {len(INTENTS)} intents = "
      f"{TARGET_PER_INTENT * len(INTENTS):,} total")
print(f"Model   : vosk-model-small-en-in-0.4 output format")
print("=" * 55)

for intent, config in INTENTS.items():
    current = sum(1 for r in all_rows if r['intent'] == intent)

    if intent in done_intents and current >= TARGET_PER_INTENT:
        print(f"  DONE   {intent:<25} ({current} records)")
        continue

    needed = TARGET_PER_INTENT - current
    print(f"\n  {intent}  (need {needed} more)")

    generated = []
    attempts  = 0

    while len(generated) < needed and attempts < MAX_ATTEMPTS:
        ask_for = min(BATCH_SIZE, needed - len(generated) + 10)
        try:
            batch = generate_batch(
                intent, config['desc'], config['seeds'], ask_for, used_texts
            )
            generated.extend(batch)
            print(f"    batch {attempts+1}: +{len(batch)} -> {len(generated)}/{needed}")
        except Exception as e:
            print(f"    error: {e} - retrying in 5s")
            time.sleep(5)
        attempts += 1
        time.sleep(0.3)

    for text in generated[:needed]:
        all_rows.append({'text': text, 'intent': intent})

    print(f"    saved {min(len(generated), needed)} records for {intent}")
    pd.DataFrame(all_rows).to_csv(CHECKPOINT_FILE, index=False)

# ── Final output ──────────────────────────────────────────────────────────────
output_df = pd.DataFrame(all_rows).drop_duplicates(subset='text')
output_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*55}")
print(f"Dataset saved to {OUTPUT_FILE}")
print(f"Total records : {len(output_df)}")
print(f"Total intents : {output_df['intent'].nunique()}")
print()
all_good = True
for intent, count in sorted(output_df['intent'].value_counts().items()):
    status = "OK" if count >= TARGET_PER_INTENT else "SHORT"
    if status == "SHORT":
        all_good = False
    print(f"  {status:<6} {intent:<25} {count}")
print(f"\nAll intents at {TARGET_PER_INTENT}+: {all_good}")
