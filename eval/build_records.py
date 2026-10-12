"""Build the 72 decision test records (records.jsonl) that the reference answers in reference/ belong to.

Each line is a Clef request plus "id", "group" and "expected": {question_id: option_id} where a
right answer is known (85 answers in all). Four groups:

- card: the two examples from Clef's model card (5 known answers);
- home: six spoken requests to a house assistant, each asked which kind of device, which room, and
  whether to confirm before acting (18 answers, 16 known);
- arc: the first 40 questions of the ARC-Challenge test split (allenai/ai2_arc, CC BY-SA 4.0), four
  options each;
- banking77: 24 messages from the BANKING77 test split (CC BY 4.0), chosen with random.seed(77),
  77 intents each.

The ARC and BANKING77 rows are downloaded from the Hugging Face datasets server, so their text is
not stored in this repository. The output is the file the reference answers were computed on; the
script checks that each record's id has a reference answer.

usage: build_records.py [OUT_FILE]     (default: records.jsonl next to this script)
"""

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split=test&offset={offset}&length={length}"


def rows(dataset: str, config: str, offset: int, length: int) -> dict:
    url = ROWS.format(dataset=dataset, config=config, offset=offset, length=length)
    error = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read())
        except Exception as e:  # noqa: BLE001 - retried, then reported
            error = e
            time.sleep(3 * (attempt + 1))
    raise SystemExit(f"could not download {url}: {error}")


records = []

records.append({"id": "card-invoice", "group": "card",
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {"type": "choice", "instructions": "What is the invoice status?",
                   "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}},
    "expected": {"status": "overdue", "large": "true"}})
records.append({"id": "card-support", "group": "card",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"}},
    "expected": {"department": "technical", "urgency": "2", "outage": "true"}})

home_questions = {
    "domain": {"type": "choice", "instructions": "Which kind of device does the request concern?",
               "criteria": {"light": "Lights and lamps", "climate": "Heating, cooling, thermostat, fans",
                            "media": "Music, speakers, TV", "cover": "Blinds, curtains, garage door",
                            "lock": "Door locks", "none": "Not a device request"}},
    "room": {"type": "choice", "instructions": "Which room is meant?",
             "criteria": {"bedroom": "The bedroom", "kitchen": "The kitchen", "living_room": "The living room",
                          "garage": "The garage", "whole_house": "Every room or no particular room"}},
    "confirm": {"type": "noul", "instructions": "Should the assistant ask for confirmation before acting, because the action affects safety or security?"},
}
home = [
    ("Turn off the lights in the bedroom please.", {"domain": "light", "room": "bedroom", "confirm": "false"}),
    ("It's freezing in the living room, bump the heat up a couple of degrees.", {"domain": "climate", "room": "living_room", "confirm": "false"}),
    ("Unlock the front door, I'm outside with groceries.", {"domain": "lock", "confirm": "true"}),
    ("Open the garage.", {"domain": "cover", "room": "garage", "confirm": "true"}),
    ("Play some jazz in the kitchen.", {"domain": "media", "room": "kitchen", "confirm": "false"}),
    ("What's the capital of Peru?", {"domain": "none", "confirm": "false"}),
]
for i, (text, expected) in enumerate(home):
    records.append({"id": f"home-{i}", "group": "home", "state": {"spoken_request": text},
                    "questions": home_questions, "expected": expected})

for r in rows("allenai/ai2_arc", "ARC-Challenge", 0, 40)["rows"]:
    row = r["row"]
    criteria = dict(zip(row["choices"]["label"], row["choices"]["text"]))
    records.append({"id": f"arc-{row['id']}", "group": "arc", "state": {"question": row["question"]},
        "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                 "criteria": criteria}},
        "expected": {"answer": row["answerKey"]}})

random.seed(77)
offsets = sorted(random.sample(range(3080), 24))  # the BANKING77 test split has 3,080 rows
labels = None
for offset in offsets:
    page = rows("legacy-datasets/banking77", "default", offset, 1)
    if labels is None:
        labels = next(f["type"]["names"] for f in page["features"] if f["name"] == "label")
    row = page["rows"][0]["row"]
    records.append({"id": f"banking77-{offset}", "group": "banking77", "state": {"customer_message": row["text"]},
        "questions": {"intent": {"type": "choice", "instructions": "Which intent does the customer message express?",
                                 "criteria": {name: name.replace("_", " ") for name in labels}}},
        "expected": {"intent": labels[row["label"]]}})

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "records.jsonl"
with open(out, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
missing = [rec["id"] for rec in records if not (HERE / "reference" / f"{rec['id']}.json").exists()]
print(len(records), "records:", {g: sum(1 for r in records if r["group"] == g) for g in ("card", "home", "arc", "banking77")},
      "->", out)
if missing:
    raise SystemExit(f"no reference answer for {missing}")
