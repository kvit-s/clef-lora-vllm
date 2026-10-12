"""Chat speed on the running vLLM server: single-stream decode and reading a long prompt.

Run it against the production configuration and against the Clef test configuration, back to back,
to see what enabling the adapter costs chat requests that do not use it.

- decode: a fixed coding prompt, temperature 0, 512 tokens, streamed; tokens per second counted
  from the first token to the last. Three runs, median reported.
- prompt reading: a 10,000-token prompt with a random first line (so the prefix cache cannot help)
  and max_tokens 1; prompt tokens divided by the time to the answer. Three runs, median reported.

usage: speed_probe.py URL MODEL [--label NAME]
"""

import argparse
import json
import random
import statistics
import time
import urllib.request

CODE_PROMPT = ("Write a Python module implementing an LRU cache with a configurable capacity, "
               "time-based expiry per entry, and thread safety. Include docstrings and a short "
               "example of use at the end.")


def stream(url, payload):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started, first, tokens = time.time(), None, 0
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            if first is None and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    first = time.time()
    return started, first, time.time(), tokens


def decode_speed(url, model):
    payload = {"model": model, "messages": [{"role": "user", "content": CODE_PROMPT}], "max_tokens": 512,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
    _, first, end, tokens = stream(url, payload)
    return (tokens - 1) / (end - first)


def prompt_speed(url, model, length=10000):
    words = " ".join(random.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma"]) for _ in range(length))
    content = f"run {random.random()}\n{words}\nHow many times does the word alpha appear? Answer with a number."
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
               "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read())
    return body["usage"]["prompt_tokens"] / (time.time() - started), body["usage"]["prompt_tokens"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("model")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    decode_speed(args.url, args.model)  # warm-up
    decode = [decode_speed(args.url, args.model) for _ in range(3)]
    prompt = [prompt_speed(args.url, args.model) for _ in range(3)]
    print(json.dumps({
        "label": args.label, "model": args.model,
        "decode_tokens_per_second": round(statistics.median(decode), 1),
        "decode_runs": [round(d, 1) for d in decode],
        "prompt_tokens_per_second": round(statistics.median(p for p, _ in prompt)),
        "prompt_tokens": prompt[0][1],
    }))


if __name__ == "__main__":
    main()
