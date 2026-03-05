"""
Dataset Generator for Hotel Intent Classification
Uses Claude API to generate diverse, natural paraphrases for each intent.

Usage:
    pip install anthropic
    set ANTHROPIC_API_KEY=your-key-here
    python generate_dataset.py
"""

import anthropic
import csv
import json
import time
import random

# ── 18 Intents with seed examples ──
# Write 30-50 unique seed sentences per intent
# These should be genuinely different phrasings, not typo variants

INTENT_SEEDS = {
    "room_cleaning": [
        "can someone come clean the room",
        "the room hasn't been cleaned today",
        "my bathroom needs cleaning",
        "could you send housekeeping",
        "the room is quite dirty",
        "could you send the maid please",
        "I'd like the room tidied up",
        "there are stains on the carpet",
        "the floor needs mopping",
        "beds haven't been made",
        "please vacuum the room",
        "can I get a deep clean",
        "the trash hasn't been emptied",
        "room needs to be swept",
        "bathroom is not cleaned properly",
        "can i get my bathroom cleaned",
        "when does housekeeping come by",
        "the dustbin is overflowing",
        "please make the bed",
        "clean the washroom please",
    ],
    "towel_request": [
        "can I get some towels",
        "we need more towels please",
        "send extra bath towels",
        "I need a fresh towel",
        "the towels are dirty can I have new ones",
        "bring face towels to my room",
        "we ran out of towels",
        "can I have a hand towel",
        "need a bath mat",
        "please replace the towels",
        "no clean towels in the bathroom",
        "extra towels for two guests",
        "can you bring washcloths",
        "I need dry towels",
        "the pool towels are wet",
    ],
    "toiletries_request": [
        "I need toiletries",
        "can I get some soap",
        "we ran out of shampoo",
        "need toothpaste and toothbrush",
        "bring bathroom supplies",
        "I need a razor",
        "can I have some conditioner",
        "there's no body wash",
        "send some dental kit",
        "tissues are finished",
        "need toilet paper",
        "can I get a comb",
        "shaving kit please",
        "bathroom amenities needed",
        "lotion and soap please",
    ],
    "blanket_request": [
        "can I get an extra blanket",
        "I need more blankets it's cold",
        "send a duvet to my room",
        "bring a comforter please",
        "extra bedding please",
        "I need a thicker blanket",
        "the blanket is too thin",
        "can I have a quilt",
        "another blanket for the bed",
        "it's chilly I need a blanket",
        "need extra linen",
        "could you bring warm covers",
        "I'd like an additional blanket",
        "more bed covers please",
        "send up a fleece blanket",
    ],
    "pillow_request": [
        "can I get an extra pillow",
        "I need more pillows",
        "bring a firm pillow please",
        "the pillow is too flat",
        "I'd like a softer pillow",
        "send two more pillows",
        "can I have a feather pillow",
        "need a hypoallergenic pillow",
        "the pillows are uncomfortable",
        "extra pillow for my room",
        "replace the pillow please",
        "I need a body pillow",
        "can you bring pillowcases too",
        "more pillows for the bed",
        "a bolster pillow please",
    ],
    "food_order": [
        "I'd like to order room service",
        "can I see the menu",
        "I want to order breakfast",
        "bring me a sandwich please",
        "I need a bottle of water",
        "can I order coffee to my room",
        "dinner in my room please",
        "I'd like some fruit",
        "order a burger and fries",
        "hot chocolate please",
        "can I get a snack",
        "I want tea with milk",
        "send up the lunch menu",
        "I'd like to order pasta",
        "a glass of juice please",
        "wine and cheese platter",
        "midnight snack please",
        "ice cream for the kids",
        "continental breakfast",
        "can I order a pizza",
    ],
    "maintenance": [
        "the air conditioning is broken",
        "my shower isn't working",
        "the toilet is clogged",
        "there's a leak in the bathroom",
        "the TV remote doesn't work",
        "light bulb needs replacing",
        "the door lock is jammed",
        "faucet is dripping",
        "electrical outlet not working",
        "the window won't close",
        "drain is blocked",
        "the sink is leaking",
        "heater isn't working",
        "the safe won't open",
        "there's no hot water",
        "the fan is making noise",
        "bathroom light flickering",
        "the mini fridge isn't cooling",
        "key card not working",
        "curtain rod is broken",
    ],
    "temperature_control": [
        "the room is too cold",
        "can you turn up the heating",
        "the AC is too strong",
        "make the room warmer please",
        "lower the temperature",
        "the thermostat isn't working",
        "room temperature is uncomfortable",
        "turn down the air conditioning",
        "it's freezing in here",
        "I need the room cooler",
        "adjust the temperature please",
        "the heating is too high",
        "can you set it to 22 degrees",
        "the room feels stuffy",
        "AC is blowing cold air",
    ],
    "lighting_control": [
        "turn off the lights",
        "the room lights won't turn on",
        "dim the lights please",
        "can you adjust the lighting",
        "the bedside lamp doesn't work",
        "turn on the bathroom light",
        "the lights are too bright",
        "I can't find the light switch",
        "need brighter lights for reading",
        "the hallway light is out",
        "switch off all lights",
        "nightlight not working",
        "can I get a desk lamp",
        "the light keeps flickering",
        "turn on the main light",
    ],
    "wake_up_call": [
        "I need a wake up call at 6 AM",
        "set an alarm for 7 in the morning",
        "can you wake me at 5:30",
        "morning call at eight please",
        "I have an early flight wake me at 4",
        "please call my room at 6:30 AM",
        "set a wake up call for tomorrow",
        "I need to be up by 7 AM",
        "wake me before sunrise",
        "alarm at half past five",
        "reminder call at 6 AM",
        "I need a morning wake up",
        "can you arrange a wake up call",
        "please wake me at quarter to 7",
        "early morning call needed",
    ],
    "checkout_billing": [
        "I want to check out",
        "what time is checkout",
        "can I see my bill",
        "I'd like to settle my account",
        "late checkout please",
        "can I extend checkout time",
        "prepare my invoice",
        "I need a receipt",
        "what are the charges on my bill",
        "express checkout please",
        "I'm checking out now",
        "can I pay by card",
        "itemize the bill please",
        "remove the minibar charges",
        "I want to dispute a charge",
    ],
    "noise_complaint": [
        "the room next door is very noisy",
        "can you tell them to keep it down",
        "there's loud music from upstairs",
        "people are partying in the hallway",
        "I can't sleep because of the noise",
        "the construction noise is terrible",
        "someone is being very loud",
        "the neighbors are arguing loudly",
        "noise from the street is awful",
        "it's too noisy to sleep",
        "can something be done about the noise",
        "there's banging on the walls",
        "the room above is stomping",
        "loud TV from next door",
        "need a quieter room",
    ],
    "emergency": [
        "I need help immediately",
        "call an ambulance",
        "there's a fire",
        "I feel very sick",
        "someone is hurt",
        "I need a doctor right away",
        "medical emergency",
        "I can't breathe properly",
        "there's been an accident",
        "I need security now",
        "I feel unsafe",
        "chest pain help me",
        "I lost my room key and can't get in",
        "locked out of my room",
        "I smell gas in the room",
    ],
    "concierge_general": [
        "what are some good restaurants nearby",
        "can you recommend local attractions",
        "where is the nearest pharmacy",
        "I need directions to the airport",
        "what's there to do around here",
        "book a table at a restaurant",
        "are there any tours available",
        "where can I go shopping",
        "recommend a good spa",
        "what time does the pool close",
        "is there a gym in the hotel",
        "where is the nearest ATM",
        "can you help with sightseeing plans",
        "what events are happening today",
        "how far is the beach from here",
    ],
    "concierge_taxi": [
        "I need a taxi",
        "call a cab for me",
        "book a car to the airport",
        "I need transportation",
        "arrange a pickup please",
        "can you get me an Uber",
        "I need a ride to the city center",
        "taxi to the train station",
        "book a shuttle to the airport",
        "car service for tomorrow morning",
        "I need a driver",
        "arrange airport transfer",
        "how much is a taxi to downtown",
        "book a cab for two people",
        "I need transport to the mall",
    ],
    "laundry_service": [
        "I need my clothes washed",
        "is there laundry service",
        "can you iron my shirt",
        "dry cleaning for my suit",
        "when will my laundry be ready",
        "I need clothes pressed",
        "wash and fold service",
        "stain on my jacket can you clean it",
        "send someone to pick up laundry",
        "I need my dress dry cleaned",
        "how much is the laundry service",
        "express laundry please",
        "can I get my pants hemmed",
        "I need ironing service",
        "where is the laundry bag",
    ],
    "do_not_disturb": [
        "don't disturb me please",
        "no housekeeping today",
        "skip the room cleaning",
        "I don't want to be disturbed",
        "put the do not disturb sign",
        "no maid service this morning",
        "leave my room as it is",
        "I want privacy",
        "don't send anyone to my room",
        "skip turndown service",
        "no cleaning needed today",
        "I'll be sleeping don't knock",
        "privacy for the whole day",
        "cancel housekeeping for today",
        "do not enter my room",
    ],
    "misc_request": [
        "I need an iron",
        "can I get extra hangers",
        "where is the ice machine",
        "send an umbrella to my room",
        "I need a phone charger",
        "can I have a bathrobe",
        "bring a hair dryer",
        "I need an adapter plug",
        "extra key card please",
        "send a sewing kit",
        "I need slippers",
        "where can I print documents",
        "can I borrow an iron board",
        "bring a corkscrew please",
        "I need stationery",
    ],
}

TARGET_PER_INTENT = 1100  # ~1100 x 18 = ~19,800 total

PROMPT_TEMPLATE = """You are generating training data for a hotel voice assistant intent classifier.

Intent: {intent}
Description: {description}

Here are example phrases for this intent:
{examples}

Generate {count} MORE unique, natural phrases that a hotel guest might say for this intent.

Rules:
- Each phrase should be a genuinely different way of expressing the same intent
- Use varied sentence structures (questions, commands, statements, polite requests)
- Include casual/informal phrasings ("hey can I get...", "yo I need...")
- Include polite/formal phrasings ("I would appreciate if...", "would it be possible to...")
- Include indirect phrasings ("the room smells bad" for cleaning, "I'm hungry" for food)
- Include context-rich phrasings ("we just checked in and the room isn't clean")
- Do NOT include typos, misspellings, or spacing errors
- Do NOT repeat the example phrases
- Keep each phrase under 15 words
- One phrase per line, no numbering, no quotes

Output ONLY the phrases, one per line."""

INTENT_DESCRIPTIONS = {
    "room_cleaning": "Guest wants their room or bathroom cleaned, tidied, swept, mopped, or beds made",
    "towel_request": "Guest needs towels (bath, hand, face, pool towels, washcloths, bath mats)",
    "toiletries_request": "Guest needs bathroom supplies (soap, shampoo, toothpaste, tissues, toilet paper, razor)",
    "blanket_request": "Guest needs blankets, duvets, comforters, quilts, or extra bedding",
    "pillow_request": "Guest needs pillows (extra, softer, firmer, hypoallergenic)",
    "food_order": "Guest wants to order food, drinks, room service, or see the menu",
    "maintenance": "Something is broken, leaking, clogged, or not working in the room",
    "temperature_control": "Guest wants to adjust room temperature (AC, heating, thermostat)",
    "lighting_control": "Guest wants to control lights (on, off, dim, brightness, lamp issues)",
    "wake_up_call": "Guest wants a morning wake-up call or alarm set",
    "checkout_billing": "Guest wants to check out, see their bill, get a receipt, or handle payment",
    "noise_complaint": "Guest is bothered by noise from neighbors, street, or construction",
    "emergency": "Medical emergency, fire, security issue, feeling unsafe, locked out",
    "concierge_general": "Guest wants recommendations, directions, bookings, or local information",
    "concierge_taxi": "Guest needs a taxi, cab, Uber, airport transfer, or transportation",
    "laundry_service": "Guest needs clothes washed, ironed, dry cleaned, or pressed",
    "do_not_disturb": "Guest doesn't want to be disturbed, skip housekeeping, wants privacy",
    "misc_request": "Guest needs miscellaneous items (hangers, iron, charger, umbrella, slippers)",
}


def generate_paraphrases(client, intent, seeds, count):
    """Generate paraphrases using Claude API"""
    examples_text = "\n".join(f"- {s}" for s in seeds)

    prompt = PROMPT_TEMPLATE.format(
        intent=intent,
        description=INTENT_DESCRIPTIONS[intent],
        examples=examples_text,
        count=count
    )

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    lines = message.content[0].text.strip().split("\n")
    # Clean up: remove numbering, bullets, quotes, empty lines
    phrases = []
    for line in lines:
        line = line.strip().lstrip("0123456789.-) ").strip('"').strip("'").strip()
        if line and len(line) > 3:
            phrases.append(line.lower())

    return phrases


def main():
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    all_data = []

    for intent, seeds in INTENT_SEEDS.items():
        print(f"\n{'='*60}")
        print(f"Generating for: {intent}")
        print(f"Seeds: {len(seeds)}")

        # Start with seed examples
        for seed in seeds:
            all_data.append((seed.lower().strip(), intent))

        # Generate remaining needed
        needed = TARGET_PER_INTENT - len(seeds)
        generated = []

        # Generate in batches of ~200
        batch_size = 200
        while len(generated) < needed:
            remaining = needed - len(generated)
            batch_count = min(batch_size, remaining)

            print(f"  Generating batch of {batch_count}... (total so far: {len(generated)})")

            try:
                # Mix seeds with some already-generated for diversity
                example_pool = seeds.copy()
                if generated:
                    example_pool += random.sample(generated, min(10, len(generated)))
                random.shuffle(example_pool)

                phrases = generate_paraphrases(client, intent, example_pool[:25], batch_count)

                # Deduplicate
                existing = set(p for p, _ in all_data) | set(generated)
                new_phrases = [p for p in phrases if p not in existing]
                generated.extend(new_phrases)

                print(f"  Got {len(new_phrases)} unique phrases")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(5)

        # Add generated phrases
        for phrase in generated[:needed]:
            all_data.append((phrase, intent))

        print(f"  Total for {intent}: {len(seeds) + len(generated[:needed])}")

    # Shuffle the dataset
    random.shuffle(all_data)

    # Write CSV
    output_file = "hotel_intents_20k.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intent"])
        for text, intent in all_data:
            writer.writerow([text, intent])

    print(f"\n{'='*60}")
    print(f"Dataset saved to {output_file}")
    print(f"Total examples: {len(all_data)}")

    # Print distribution
    from collections import Counter
    dist = Counter(intent for _, intent in all_data)
    print("\nDistribution:")
    for intent, count in sorted(dist.items()):
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
