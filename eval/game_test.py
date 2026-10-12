"""Decisions about a game whose state grows at its end: reuse against reading in full.

The state is one string: a long fixed text standing in for a game's rules (the start of --corpus,
cut to --corpus-tokens tokens), then a game log that gains --turns-per-decision made-up turns before
each decision. Every decision asks which step to take next (a choice among six steps) and whether
energy is low (yes or no). Each request goes to two decision services:

- REUSE_URL, a service keeping the starts of prompts, which lets vLLM serve the unchanged start from
  its prefix cache and fills those positions in from what it kept;
- FULL_URL, a service started with --reuse-memory-mb 0, which reads every prompt in full.

For each decision it prints how many tokens were reused, how many vLLM is expected to reuse, the time
each service took, and the largest difference between their probabilities. vLLM is expected to reuse
the whole --block-token blocks that match the previous prompt, less --held-back-blocks: the real
server holds back two (one because the speculative draft model needs the last matching block computed
again, one where the linear-attention layers' saved state ends); the stand-in for vLLM holds back none.
Exits non-zero if a request fails, the reuse is not as expected, or the probabilities differ by more
than --tolerance. On the real server two full reads of the same prompt already differ by up to about
0.1, so the tolerance there has to be above that; --repeat-full reads every prompt in full a second
time and prints that difference beside the one between reuse and the first full read.

Without --corpus, the fixed text is made up: numbered rules built from a small vocabulary.

usage: game_test.py REUSE_URL FULL_URL --release DIR [--corpus FILE] [--corpus-tokens 12000] [--decisions 8]
                    [--turns-per-decision 5] [--block 1552] [--held-back-blocks 0] [--tolerance 1e-4]
                    [--repeat-full] [--check-refusal]
"""

import argparse
import json
import random
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
import reuse  # noqa: E402

QUESTIONS = {
    "next_step": {
        "type": "choice",
        "instructions": "Which step should the explorer take next?",
        "criteria": {
            "north": "Move one tile north", "south": "Move one tile south", "east": "Move one tile east",
            "west": "Move one tile west", "rest": "Stay on this tile and recover energy",
            "dig": "Dig for treasure on the current tile",
        },
    },
    "low_energy": {"type": "noul", "instructions": "Is the explorer's energy dangerously low?"},
}
TERRAIN = ["forest", "river bank", "open field", "rocky hill", "swamp", "ruined wall", "sand dune"]
MOVES = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}


def game_log(turns: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    x, y, energy, lines = 0, 0, 60, []
    for turn in range(1, turns + 1):
        move = rng.choice(list(MOVES) + ["rest", "dig"])
        if move in MOVES:
            dx, dy = MOVES[move]
            x, y, energy = x + dx, y + dy, max(0, energy - rng.randint(1, 4))
            lines.append(f"Turn {turn}: the explorer moved {move} to ({x},{y}), a {rng.choice(TERRAIN)}. "
                         f"Energy is now {energy}.")
        elif move == "rest":
            energy = min(60, energy + 5)
            lines.append(f"Turn {turn}: the explorer rested at ({x},{y}). Energy is now {energy}.")
        else:
            found = rng.random() < 0.2
            energy = max(0, energy - 3)
            lines.append(f"Turn {turn}: the explorer dug at ({x},{y}) and found "
                         f"{'a silver coin' if found else 'nothing'}. Energy is now {energy}.")
    return lines


def made_up_rules(count: int = 2000, seed: int = 11) -> str:
    """A long fixed text standing in for a game's rules, when no --corpus is given."""
    rng = random.Random(seed)
    things = ["the explorer", "a guard", "the river", "each tile", "a lantern", "the map", "a coin", "the gate"]
    verbs = ["may cross", "must avoid", "can carry", "reveals", "blocks", "doubles", "halves", "resets"]
    ends = ["after sunset", "on even turns", "when energy is below ten", "unless it is raining",
            "while holding a lantern", "near the ruined wall", "once per game", "if the gate is open"]
    return "\n".join(f"Rule {i}: {rng.choice(things)} {rng.choice(verbs)} {rng.choice(things)} {rng.choice(ends)}."
                     for i in range(1, count + 1))


def corpus_text(path: str | None) -> str:
    return Path(path).read_text() if path else made_up_rules()


def post(url, payload):
    request = urllib.request.Request(url + "/v1/systemone", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def probabilities(answers):
    out = {}
    for question_id, answer in answers.items():
        if answer["type"] == "noul":
            out[question_id] = {"true": answer["noul"], "false": 1 - answer["noul"]}
        else:
            out[question_id] = answer["probabilities"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reuse_url")
    parser.add_argument("full_url")
    parser.add_argument("--corpus", help="a long text file for the fixed part of the state; made up if omitted")
    parser.add_argument("--corpus-tokens", type=int, default=12000)
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument("--turns-per-decision", type=int, default=5)
    parser.add_argument("--block", type=int, default=1552)
    parser.add_argument("--held-back-blocks", type=int, default=0)
    parser.add_argument("--match-unit", type=int, default=0,
                        help="vLLM's --prefix-match-unit, for the expected reuse (0: the block size)")
    parser.add_argument("--prime-margin", type=int, default=0,
                        help="the reuse service's --prime-margin, for the expected reuse (0: no priming)")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--check-refusal", action="store_true", help="also check that an over-long state is refused")
    parser.add_argument("--repeat-full", action="store_true", help="read every prompt in full twice, to show the run-to-run difference")
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(clef_code.release_dir())
    corpus_ids = tokenizer(corpus_text(args.corpus), add_special_tokens=False)["input_ids"][:args.corpus_tokens]
    rules = "GAME RULES AND NOTES\n\n" + tokenizer.decode(corpus_ids) + "\n\nGAME LOG\n"
    log = game_log(args.decisions * args.turns_per_decision)

    failures, worst, worst_repeat, previous_ids, disagreements = 0, 0.0, 0.0, None, 0
    unit = args.match_unit or args.block
    history: list[tuple[list[int], set[int]]] = []  # prompts sent unsalted, and the points each left

    def points(length: int) -> set[int]:
        """Where a prompt of `length` tokens leaves points to resume from: every block boundary and its
        last unit boundary (with the plugin's CLEF_FULL_CACHE_HITS=1)."""
        found = set(range(args.block, length + 1, args.block))
        if length // unit * unit:
            found.add(length // unit * unit)
        return found

    def expected_reuse(ids: list[int], needed: int) -> int:
        limit, best = min(len(ids) - 1, needed), 0
        for earlier, left in history:
            common = next((i for i, (a, b) in enumerate(zip(ids, earlier)) if a != b), min(len(ids), len(earlier)))
            best = max([best] + [b for b in left if b <= min(common, limit)])
        return best
    print(f"{'decision':>8} {'tokens':>7} {'reused':>7} {'expected':>8} {'reuse s':>8} {'full s':>7} {'max diff':>9} "
          f"{'full again':>10}  answer")
    for decision in range(1, args.decisions + 1):
        state = rules + "\n".join(log[:decision * args.turns_per_decision]) + "\n"
        payload = {"model": "clef", "state": state, "questions": QUESTIONS}
        encoded = encode_record(tokenizer, payload, max_length=1 << 40)
        ids = list(encoded.input_ids)
        needed = reuse.first_needed_position(encoded)
        allowed = 0
        if args.held_back_blocks:
            if previous_ids is not None:
                common = next((i for i, (a, b) in enumerate(zip(ids, previous_ids)) if a != b),
                              min(len(ids), len(previous_ids)))
                allowed = max(0, min(common // args.block, (len(ids) - 1) // args.block) - args.held_back_blocks) * args.block
        else:
            allowed = expected_reuse(ids, needed)
        previous_ids = ids
        started = time.perf_counter()
        status_r, reply_r = post(args.reuse_url, payload)
        reuse_seconds = time.perf_counter() - started
        started = time.perf_counter()
        status_f, reply_f = post(args.full_url, payload)
        full_seconds = time.perf_counter() - started
        if status_r != 200 or status_f != 200:
            print(f"decision {decision}: HTTP {status_r} / {status_f}: {reply_r} {reply_f}")
            failures += 1
            continue
        reused = reply_r["timings"]["reused_tokens"]
        history.append((ids, points(len(ids))))
        if args.prime_margin:
            prime_end = (needed - args.prime_margin) // unit * unit
            if prime_end > reused:
                history.append((ids[:prime_end], points(prime_end)))
        pr, pf = probabilities(reply_r["answers"]), probabilities(reply_f["answers"])
        diff = max(abs(pr[q][o] - pf[q][o]) for q in pr for o in pr[q])
        worst = max(worst, diff)
        again = ""
        if args.repeat_full:
            pa = probabilities(post(args.full_url, payload)[1]["answers"])
            repeat = max(abs(pa[q][o] - pf[q][o]) for q in pa for o in pa[q])
            worst_repeat = max(worst_repeat, repeat)
            again = f"{repeat:.2e}"
        step = reply_r["answers"]["next_step"]["choice"]
        note = " (read twice)" if reply_r["timings"].get("read_twice") else ""
        if step != reply_f["answers"]["next_step"]["choice"]:
            disagreements += 1
            note += f" (full read chose {reply_f['answers']['next_step']['choice']})"
        print(f"{decision:>8} {len(ids):>7} {reused:>7} {allowed:>8} {reuse_seconds:>8.3f} {full_seconds:>7.3f} "
              f"{diff:>9.2e} {again:>10}  {step} (low energy {reply_r['answers']['low_energy']['noul']:.2f}){note}")
        if decision > 1 and reused != allowed:
            failures += 1
        if diff > args.tolerance:
            failures += 1
    print(f"largest probability difference between reuse and reading in full: {worst:.2e}"
          + (f"; between two full reads: {worst_repeat:.2e}" if args.repeat_full else ""))
    print(f"decisions where reuse and the full read chose a different step: {disagreements}")

    if args.check_refusal:
        payload = {"model": "clef", "state": rules * (17000 // max(1, len(corpus_ids)) + 2), "questions": QUESTIONS}
        status, reply = post(args.reuse_url, payload)
        print(f"over-long state: HTTP {status} {reply.get('error', '')}")
        failures += status != 400
    if failures:
        print(f"{failures} failure(s)")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
