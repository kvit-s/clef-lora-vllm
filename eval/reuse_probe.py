"""When does vLLM reuse the start of a decision prompt? A sequence of decisions, each checking one case.

vLLM reuses whole 1,552-token blocks of a prompt it has read before, less the last two that match,
but only while the linear-attention layers' saved state at the block where reuse would end is still
in its cache. This script sends decisions to a decision service (with reuse on) and prints, for each,
how many tokens vLLM reused against how many the block rule allows given every decision sent before:

  A  first decision of a game                                    (nothing to reuse)
  B  a variant whose rules differ from the middle on, right after A
  C  the next turns of the game (each extends the previous prompt)
  D  a long question (64 options) on the current state, then the next turns with the same question
  E  the next turns again after CHAT_PROMPTS long chat requests have gone through vLLM
  F  the variant from B again, at the end

With --threshold N it instead looks for how much chat traffic pushes a game out of vLLM's cache: for
k = 1..N it sends a decision, then k chat requests of --chat-tokens new tokens each, then the next
decision, and prints whether that decision still reused what the block rule allows.

The held-back count of two was measured on a server with a speculative draft model; without one,
try --held-back-blocks 1.

usage: reuse_probe.py SERVICE_URL VLLM_URL --chat-model NAME --release DIR [--corpus FILE]
                      [--corpus-tokens 12000] [--chat-prompts 6] [--chat-tokens 12000] [--block 1552]
                      [--held-back-blocks 2] [--threshold N]
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "service"))
import clef_code  # noqa: E402

clef_code.use_release()
import game_test  # noqa: E402
from joint_schema_model import encode_record  # noqa: E402

LONG_QUESTION = {
    "go_to": {
        "type": "choice",
        "instructions": "Which tile should the explorer walk to next?",
        "criteria": {f"tile_{x}_{y}": f"Walk to tile ({x},{y}) by the shortest safe path, then stop and look around"
                     for x in range(8) for y in range(8)},
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("service_url")
    parser.add_argument("vllm_url")
    parser.add_argument("--chat-model", required=True, help="the name vLLM serves the base model under")
    parser.add_argument("--corpus", help="a long text file for the fixed part of the state; made up if omitted")
    parser.add_argument("--corpus-tokens", type=int, default=12000)
    parser.add_argument("--chat-prompts", type=int, default=6)
    parser.add_argument("--chat-tokens", type=int, default=12000)
    parser.add_argument("--block", type=int, default=1552)
    parser.add_argument("--held-back-blocks", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(clef_code.release_dir())
    text_ids = tokenizer(game_test.corpus_text(args.corpus), add_special_tokens=False)["input_ids"]
    rules = "GAME RULES AND NOTES\n\n" + tokenizer.decode(text_ids[:args.corpus_tokens]) + "\n\nGAME LOG\n"
    half = args.corpus_tokens // 2
    variant = ("GAME RULES AND NOTES\n\n" + tokenizer.decode(text_ids[:half])
               + "\n\nVARIANT: from here on the rules are replaced.\n" + tokenizer.decode(text_ids[-half:]) + "\n\nGAME LOG\n")
    log = game_test.game_log(60)
    sent: list[list[int]] = []

    def expected_reuse(ids):
        best = 0
        for earlier in sent:
            common = next((i for i, (a, b) in enumerate(zip(ids, earlier)) if a != b), min(len(ids), len(earlier)))
            best = max(best, common)
        blocks = min(best // args.block, (len(ids) - 1) // args.block) - args.held_back_blocks
        return best, max(0, blocks) * args.block

    def decide(label, state, questions=game_test.QUESTIONS):
        payload = {"model": "clef", "state": state, "questions": questions}
        ids = list(encode_record(tokenizer, payload, max_length=1 << 40).input_ids)
        common, expected = expected_reuse(ids)
        started = time.time()
        status, reply = game_test.post(args.service_url, payload)
        seconds = time.time() - started
        sent.append(ids)
        if status != 200:
            print(f"{label:<44} HTTP {status}: {reply}")
            return
        reused = reply["timings"]["reused_tokens"]
        mark = "" if reused == expected else "   <-- differs"
        print(f"{label:<44} {len(ids):>7} {common:>7} {expected:>8} {reused:>7} {seconds:>6.2f}{mark}", flush=True)

    chat_count = [0]

    def chat_traffic(count=None):
        total = 0
        for _ in range(args.chat_prompts if count is None else count):
            index = chat_count[0] = chat_count[0] + 1
            start = (index * 7919) % max(1, len(text_ids) - args.chat_tokens)
            prompt = f"Request {index} at {time.time()}: summarise the following notes.\n\n" + tokenizer.decode(text_ids[start:start + args.chat_tokens])
            body = {"model": args.chat_model, "max_tokens": 64, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": prompt}]}
            request = urllib.request.Request(args.vllm_url + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                total += json.loads(response.read())["usage"]["prompt_tokens"]
        return total

    def state(turns, base=rules):
        return base + "\n".join(log[:turns]) + "\n"

    print(f"{'case':<44} {'tokens':>7} {'common':>7} {'expected':>8} {'reused':>7} {'secs':>6}")
    if args.threshold:
        turns = 5
        decide(f"decision, {turns} turns", state(turns))
        for count in range(1, args.threshold + 1):
            turns += 1
            decide(f"decision, {turns} turns", state(turns))
            tokens = chat_traffic(count)
            print(f"   ({count} chat requests, {tokens} prompt tokens)")
            turns += 1
            decide(f"decision after them, {turns} turns", state(turns))
        return
    decide("A  first decision, 5 turns", state(5))
    decide("B  variant rules, differing from the middle", state(5, variant))
    for turns in (10, 15, 20):
        decide(f"C  next turns, {turns} turns", state(turns))
    decide("D  long question, 20 turns", state(20), LONG_QUESTION)
    decide("D  long question, 25 turns", state(25), LONG_QUESTION)
    decide("D  long question, 30 turns", state(30), LONG_QUESTION)
    decide("D  short question again, 35 turns", state(35))
    tokens = chat_traffic()
    print(f"   ({args.chat_prompts} chat requests, {tokens} prompt tokens in all)")
    decide("E  next turns after the chats, 40 turns", state(40))
    decide("E  next turns, 45 turns", state(45))
    decide("F  variant rules again", state(5, variant))


if __name__ == "__main__":
    main()
