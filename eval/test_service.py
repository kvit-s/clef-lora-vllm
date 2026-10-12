"""Check the decision service against the CPU reference, over HTTP.

Build the records first (build_records.py), and start the service against vLLM with the Clef
adapter, or on hidden states saved by reference.py (no GPU needed):
    python service/service.py --release DIR --reference-dir HIDDEN_DIR --listen 127.0.0.1:8092
Then:
    python eval/test_service.py http://127.0.0.1:8092 [--records eval/records.jsonl]
                                [--reference eval/reference] [--groups card,home]

For every record with a reference result, the service's probabilities are compared with the ones
Cloudflare's code produced from Clef's bf16 weights on the CPU (with the scoring network in fp32),
and the answers are compared with the expected ones. Also checks that malformed requests are refused with 400.
Prints a summary and exits non-zero if any request failed.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def post(url, payload):
    request = urllib.request.Request(url + "/v1/systemone", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def answer_probabilities(question, answer):
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--records", default=str(Path(__file__).resolve().parent / "records.jsonl"))
    parser.add_argument("--reference", default=str(Path(__file__).resolve().parent / "reference"))
    parser.add_argument("--groups", default="")
    parser.add_argument("--column", default="probabilities_fp32")
    args = parser.parse_args()
    groups = set(filter(None, args.groups.split(",")))
    failures, worst, latencies = 0, 0.0, []
    stats = {}
    for line in open(args.records):
        record = json.loads(line)
        if groups and record["group"] not in groups:
            continue
        ref_path = Path(args.reference) / f"{record['id']}.json"
        if not ref_path.exists():
            continue
        reference = json.loads(ref_path.read_text())
        payload = {"model": "clef", "state": record["state"], "questions": record["questions"]}
        started = time.perf_counter()
        status, body = post(args.url, payload)
        latencies.append(time.perf_counter() - started)
        if status != 200:
            print(f"{record['id']}: HTTP {status} {body}")
            failures += 1
            continue
        group = stats.setdefault(record["group"], {"right": 0, "known": 0, "agree": 0, "questions": 0})
        for q in reference["questions"]:
            ref_probs = dict(zip(q["options"], q[args.column]))
            got = answer_probabilities(record["questions"][q["id"]], body["answers"][q["id"]])
            # the service rounds to 4 decimals, as Cloudflare's systemone_answer does
            diff = max(abs(got[o] - ref_probs[o]) for o in ref_probs)
            worst = max(worst, diff)
            ref_best = max(ref_probs, key=ref_probs.get)
            got_best = max(got, key=got.get)
            group["questions"] += 1
            group["agree"] += ref_best == got_best
            expected = record.get("expected", {}).get(q["id"])
            if expected is not None:
                group["known"] += 1
                group["right"] += got_best == expected
    for name, g in stats.items():
        print(f"{name}: {g['agree']}/{g['questions']} answers agree with the reference; "
              f"{g['right']}/{g['known']} match the expected answer")
    print(f"largest probability difference from the reference: {worst:.4f}")
    if latencies:
        latencies.sort()
        print(f"latency: median {latencies[len(latencies) // 2]:.3f}s, max {latencies[-1]:.3f}s over {len(latencies)} requests")

    bad = [
        ({"model": "clef", "state": "x", "questions": {"q": {"type": "maybe"}}}, "unknown question type"),
        ({"model": "clef", "questions": {"q": {"type": "noul"}}}, "missing state"),
        ({"model": "clef", "state": "x", "questions": {}}, "no questions"),
        ({"model": "clef", "state": "x", "images": ["data"], "questions": {"q": {"type": "noul"}}}, "pictures"),
        ({"model": "clef", "state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}}, "empty choice"),
    ]
    for payload, what in bad:
        status, body = post(args.url, payload)
        ok = status == 400
        failures += not ok
        print(f"refuses {what}: {'yes' if ok else f'NO (HTTP {status} {body})'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
