"""Decision latency against prompt length, read in full and with the start of the prompt reused.

For each target prompt length it sends one 6-option choice question (the intent question of the
jev-laya benchmark's state-length sweep, github.com/harrymunro/jev-laya-benchmark, MIT) about a
support transcript of that many tokens, in three ways:

- cold:   the transcript starts with a line no earlier request had, so nothing can be reused;
- repeat: the identical request again, as the jev-laya benchmark sent each state 20 times;
- grown:  the same transcript with a few new lines appended at its end, the way a conversation or
          a game log grows, so its start can be reused and the new end has to be read.

Times are measured at the client, around the HTTP request. Each response's timings (seconds in
vLLM, in the scoring network, tokens reused) are recorded beside them. Results are printed as a
table of medians and written as JSON lines to --out.

usage: latency_sweep.py SERVICE_URL --release DIR [--sizes 250,500,...] [--reps 5] [--out FILE]
       The service must allow the largest size (service.py --max-length).
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "service"))
import clef_code  # noqa: E402

clef_code.use_release()
from joint_schema_model import encode_record  # noqa: E402

QUESTION = {
    "intent": {
        "type": "choice",
        "instructions": "What is the primary thing the customer wants in the transcript?",
        "criteria": {
            "refund": "money back, a charge reversed, or a duplicate/incorrect payment returned",
            "technical_help": "help fixing a bug, error, outage, or something in the product not working",
            "billing_question": "a question about an invoice, a charge, a plan price, or a payment method, without asking for money back",
            "information": "general information, how a feature works, availability, or how to do something",
            "cancellation": "wants to cancel, close the account, or downgrade the plan",
            "other": "a request that fits none of the other options, such as a compliment, feedback, or a partnership enquiry",
        },
    }
}
NAMES = ["Ana", "Ben", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "June"]
ITEMS = ["the March invoice", "the export button", "the mobile app", "order #A8841", "the team plan", "the API key",
         "the shared calendar", "the backup job", "the login page", "the annual subscription"]
EVENTS = ["asked about {item} and was told to wait for an email", "reported that {item} showed an error after the update",
          "said {item} had worked fine until yesterday", "pasted a screenshot of {item} with the settings open",
          "mentioned that a colleague had the same problem with {item}", "asked whether {item} is included in the price",
          "confirmed the account email and the last four digits of the card", "was put on hold while the agent checked {item}",
          "repeated that {item} was charged twice this month", "thanked the agent but said the issue with {item} remains"]


def transcript_lines(seed: int):
    rng = random.Random(seed)
    turn = 0
    while True:
        turn += 1
        event = rng.choice(EVENTS).format(item=rng.choice(ITEMS))
        yield f"[{turn:05d}] {rng.choice(NAMES)} {event}."


def prompt_tokens(tokenizer, state: str) -> int:
    return len(encode_record(tokenizer, {"model": "clef", "state": state, "questions": QUESTION}, max_length=1 << 40).input_ids)


def build_lines(tokenizer, target: int, seed: int) -> list[str]:
    """Transcript lines whose prompt (with a header line) is close to `target` tokens."""
    lines, source = [], transcript_lines(seed)
    overhead = prompt_tokens(tokenizer, "Support transcript, case 0000000000\n")
    per_line = None
    while True:
        count = prompt_tokens(tokenizer, "Support transcript, case 0000000000\n" + "\n".join(lines))
        if count >= target:
            return lines
        if per_line is None and lines:
            per_line = (count - overhead) / len(lines)
        need = target - count
        step = max(1, int(need / per_line * 0.9)) if per_line else 20
        lines.extend(next(source) for _ in range(step))


def post(url: str, state: str) -> tuple[float, dict]:
    body = json.dumps({"model": "clef", "state": state, "questions": QUESTION}).encode()
    request = urllib.request.Request(url + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r}")
    return time.perf_counter() - started, reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    parser.add_argument("--sizes", default="250,500,1000,2000,4000,8000,16000",
                        help="target prompt lengths in tokens, comma-separated")
    parser.add_argument("--reps", type=int, default=5, help="requests per size and mode (fewer above 16,000 tokens)")
    parser.add_argument("--grow-lines", type=int, default=4, help="lines appended for each grown request")
    parser.add_argument("--out", default="latency-sweep.jsonl")
    parser.add_argument("--label", default="")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="seconds to wait after each request, as a client acting on an answer would")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(clef_code.release_dir())
    nonce = random.Random().randrange(10**9, 10**10)
    post(args.url, f"Support transcript, case {nonce - 1}\nwarm-up")  # first request after a start is slower
    rows = []
    out = open(args.out, "a")
    print(f"{'size':>6} {'mode':<7} {'tokens':>7} {'reused':>7} {'client ms':>10} {'vLLM ms':>8} {'head ms':>8}", flush=True)
    for size in [int(s) for s in args.sizes.split(",")]:
        lines = build_lines(tokenizer, size, seed=size)
        extra = transcript_lines(seed=size + 1)
        reps = args.reps if size <= 16000 else max(2, args.reps // 2)
        for rep in range(reps):
            nonce += 1
            state = f"Support transcript, case {nonce}\n" + "\n".join(lines)
            for mode in ("cold", "repeat", "grown"):
                if mode == "grown":
                    state += "\n" + "\n".join(next(extra) for _ in range(args.grow_lines))
                seconds, reply = post(args.url, state)
                t = reply["timings"]
                row = {"label": args.label, "size": size, "mode": mode, "rep": rep,
                       "tokens": reply["usage"]["input_tokens"], "reused": t["reused_tokens"],
                       "client_ms": round(seconds * 1000, 1), "vllm_ms": round(t["backbone_seconds"] * 1000, 1),
                       "head_ms": round(t["head_seconds"] * 1000, 1), "read_twice": bool(t.get("read_twice")),
                       "answer": reply["answers"]["intent"]["choice"]}
                rows.append(row)
                time.sleep(args.pause)
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{size:>6} {mode:<7} {row['tokens']:>7} {row['reused']:>7} {row['client_ms']:>10.1f} "
                      f"{row['vllm_ms']:>8.1f} {row['head_ms']:>8.1f}" + ("  read twice" if row["read_twice"] else ""),
                      flush=True)

    print("\nmedians")
    print(f"{'size':>6} {'tokens':>7} | {'cold ms':>8} {'repeat ms':>9} {'reused':>7} {'grown ms':>9} {'reused':>7}")
    for size in sorted({r["size"] for r in rows}):
        def med(mode, key):
            values = [r[key] for r in rows if r["size"] == size and r["mode"] == mode]
            return statistics.median(values) if values else float("nan")
        print(f"{size:>6} {med('cold', 'tokens'):>7.0f} | {med('cold', 'client_ms'):>8.1f} {med('repeat', 'client_ms'):>9.1f} "
              f"{med('repeat', 'reused'):>7.0f} {med('grown', 'client_ms'):>9.1f} {med('grown', 'reused'):>7.0f}")


if __name__ == "__main__":
    main()
