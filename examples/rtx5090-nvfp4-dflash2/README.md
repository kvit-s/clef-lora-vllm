# Worked example: one RTX 5090 serving chat and Clef decisions

This is the setup the repository was built and measured on: a Windows desktop with an RTX 5090
(32 GB), the language model running in Docker inside WSL2, serving Qwen3.8-27B to coding agents
and other chat clients. Clef decisions were added to the same server, without a second copy of
the model.

## The server before the adapter

- **Model:** `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`, a 4-bit NVFP4 checkpoint of about
  17 GiB; besides the weight scales it ships input scales for quantizing activations.
- **Server:** the DFlash2 vLLM image for RTX 50-series cards
  (`ghcr.io/seanyourhighness/vllm-sm12x-nvfp4-dflash2`, vLLM 0.27.2.dev0), with speculative decoding
  by a 1.9-billion-parameter draft model that proposes seven tokens ahead.
- **Cache:** 4-bit (NVFP4) keys and values, 6.5 GiB reserved, 264,176 token slots: one request at
  the full 262,144-token context, or up to eight shorter conversations.
- **Speed:** 243 tokens/s single-stream decode on a coding prompt at temperature 0; 10,072
  tokens/s reading a 10,000-token prompt.
- **Memory:** about 200 MiB of the card free. When the Windows desktop needs more, the display
  driver moves part of the model's memory to system RAM and decode falls to 20–60 tokens/s, so
  anything added has to come out of an existing reservation.

## What the adapter changed

`compose.clef.yaml` in this directory is an overlay for the DFlash2 project's own compose file.
The adapter's buffers (1.4 GB plus vLLM's bookkeeping) came out of the key-value reservation,
which went from 6.5 to 5 GiB: vLLM fits 194,648 token slots in it, and the longest request is
194,000 tokens. Measured on 2026-10-07:

| | Without the adapter | With the adapter |
|---|---|---|
| card memory in use (`nvidia-smi`) | 30,936–31,303 MiB | 31,293 MiB |
| card memory Windows holds in system RAM | 333–570 MB | 2,826 MB |
| chat decode, one conversation | 243 tokens/s | 206 tokens/s |
| prompt reading, 10,000 tokens | 10,072 tokens/s | 9,600 tokens/s |
| longest request | 262,144 tokens | 194,000 tokens |

Decode is 15% slower with the adapter loaded, whether or not decisions are being made. PCIe
traffic during a chat with the adapter loaded was 30–46 MB/s, against about 8,000 MB/s when the
card was paging to system RAM, so the loss comes from vLLM's LoRA support itself rather than from
memory moved off the card.

## Decisions

The 72 test records (`eval/`), with the full-precision CPU run of Clef as the reference:

| | Known answers right | Answers agreeing with the reference |
|---|---|---|
| Clef at bf16 on the CPU (the reference) | 83 of 85 | — |
| this setup | 84 of 85 | 84 of 87 |

The largest difference in any option's probability was 0.53. Two later runs scored 83 and 85 of
85; the answers that moved between runs were close calls such as one whose probability sits at
about 0.5. The last layer's output, compared with the reference over the first 16 ARC records
(`compare-hidden.json`), had a relative error of 0.38–0.45 and a mean cosine similarity per
position of 0.906–0.932 with the adapter, against 1.10–1.19 and 0.34–0.42 without it; the
remaining distance comes from the 4-bit checkpoint and cache rather than from the adapter.

| Prompt length | Requests | vLLM | Scoring network | Total (median) |
|---|---|---|---|---|
| under 300 tokens (median 217) | 41 | 154 ms | 40 ms | 197 ms |
| 300–700 tokens (median 445) | 7 | 167 ms | 53 ms | 219 ms |
| BANKING77, median 1,836 tokens | 24 | 203 ms | 128 ms | 336 ms |

Over all 72 the median was 212 ms. A one-token completion without the adapter took 123 ms for
217 tokens on the same server, so most of a short decision's time is vLLM's fixed cost per
request; the adapter and the capture add about 30 ms. Ten decisions sent while a 600-token chat
streamed took a median of 267 ms.

## Latency by prompt length, and against Jev

Measured on 2026-10-11 with `eval/latency_sweep.py`: one 6-option choice question about a
support transcript of growing length, sent from the same machine. For this run the service
allowed prompts of up to 65,536 tokens (`--max-length 65536`) and kept 4 GB of prompt starts.
Each cell is the median time at the client over five requests (two at 32,000 and 64,000 tokens):

| Prompt tokens | Read in full | The identical request again | Transcript grown by four lines | Tokens reused |
|---:|---:|---:|---:|---:|
| 310 | 177 ms | 156 ms | 173 ms | 0 |
| 739 | 208 ms | 209 ms | 204 ms | 0 |
| 1,015 | 219 ms | 208 ms | 223 ms | 0 |
| 2,022 | 351 ms | 359 ms | 455 ms | 0 |
| 4,002 | 744 ms | 716 ms | 731 ms | 0 |
| 8,019 | 1,432 ms | 677 ms | 686 ms | 4,656 |
| 16,002 | 3,036 ms | 892 ms | 953 ms | 12,416 |
| 32,013 | 7,036 ms | 1,438 ms | 1,466 ms | 27,936 |
| 64,015 | 16,588 ms | 2,171 ms | 2,142 ms | 60,528 |

The table was measured with the first version of the decision service's store of reused rows.
The current version keeps each prompt start's rows in one buffer instead of many small blocks
(see docs/how-it-works.md). Run back to back on the same server on 2026-10-11, the old version
before and after the new one, so that drift of the machine shows up as a difference between the
two old runs (medians of three requests, two at 32,000 and 64,000 tokens; `store-comparison.jsonl`):

| Prompt | Request | Old store (two runs) | Current store |
|---:|---|---:|---:|
| 16,002 | identical again | 989 / 977 ms | 797 ms |
| 16,089 | grown by four lines | 1,032 / 1,009 ms | 908 ms |
| 32,013 | identical again | 1,584 / 1,426 ms | 1,180 ms |
| 32,101 | grown by four lines | 1,576 / 1,506 ms | 1,209 ms |
| 64,015 | identical again | 2,532 / 2,112 ms | 1,467 ms |
| 64,102 | grown by four lines | 2,301 / 2,042 ms | 1,456 ms |
| 64,015 | read in full | 17,992 / 17,388 ms | 17,166 ms |

With `CLEF_FULL_CACHE_HITS=1` added to vLLM's environment in `compose.clef.yaml` (the plugin then
skips vLLM's held-back blocks for decisions and saves the state at a decision prompt's last block
boundary; docs/how-it-works.md), the same sweep gave (`sweep-full-cache-hits.jsonl`):

| Prompt | Grown by four lines | Identical again | Reused |
|---:|---:|---:|---:|
| 4,089 | 230 ms | 223 ms | 3,104 |
| 8,106 | 199 ms | 195 ms | 7,760 |
| 16,089 | 246 ms | 217 ms | 15,520 |
| 32,101 | 395 ms | 387 ms | 31,040 |
| 64,102 | 478 ms | 413 ms | 63,632 |

With the scoring network on the GPU as well (`CLEF_GPU_HEAD=1`, `CLEF_RELEASE` mounted,
`CLEF_GPU_STORE_MB=400`, the key-value reservation cut from 5 to 4 GiB and the longest request to
147,000 tokens; the service with `--gpu-head`), on 2026-10-12 (`sweep-gpu-scoring.jsonl`):

| Prompt | Read in full | Grown by four lines | Identical again | The network's share |
|---:|---:|---:|---:|---:|
| 1,015 | 169–176 ms | 165–183 ms | 164–166 ms | 25–26 ms |
| 8,019 | 1,036 ms | 175–178 ms | 179–182 ms | 24–47 ms |
| 16,002 | 2,083–2,150 ms | 181–193 ms | 188–196 ms | 24–63 ms |
| 32,013 | 4,824–4,828 ms | 310–311 ms | 288–295 ms | 34–108 ms |

Nothing is reused below about 4,700 tokens, because vLLM reuses whole 1,552-token blocks and
holds back the last two. Read in full, a 64,000-token decision spent 12.8 s in vLLM (about 5,000
tokens/s) and 3.7 s in the scoring network on the CPU. At 2,000 tokens the grown request was
about 100 ms slower than reading in full in all five tries, all of it in vLLM; it reuses nothing,
and the cause has not been investigated. The same sizes sent through the house's front door on
another machine took 1–41 ms longer at the median. The raw measurements are in `latency-sweep.jsonl`.

Prompts of 32,000 and 64,000 tokens ran without error, but Clef's accuracy beyond 16,384 tokens,
the default limit of Cloudflare's released code, has not been tested. Cloudflare describes Clef as
having a 64k context window, and its hosted Clef accepts 65,536 tokens.

**Published figures for Jev.** TypeSafe publishes no latency figures for Jev 1.13 and does not
document prompt caching either way. Its limits are 64k tokens per request and 32k for the state
plus the longest question. Two independent measurements exist:

- The [jev-laya benchmark](https://github.com/harrymunro/jev-laya-benchmark) (2026-09-21) timed
  the same kind of request, a 6-option choice about a support transcript, from a Mac whose round
  trip to the API was about 50 ms. It sent each state 20 times after three warm-up requests and
  reports the median.
- Cloudflare's [decision model leaderboard](https://clef-evals.workers-ai-mle.workers.dev), from
  the Decision Index's latency-v1 sample of 750 rows sent one at a time, gives Jev a median of
  524 ms and a p95 of 536 ms over HTTPS from Cloudflare's lab, and Clef, as reported by Cloudflare
  itself, 209 ms and 239 ms.

Side by side with the first, at the nearest prompt lengths (token counts come from different
tokenizers, and Jev's include its own question template):

| Prompt tokens, Jev / here | Jev p50, including about 50 ms of network | Here, read in full | Here, with the start reused |
|---|---:|---:|---:|
| 533 / 310 | 138 ms | 177 ms | too short to reuse |
| 670 / 739 | 188 ms | 208 ms | too short to reuse |
| 1,258 / 1,015 | 148 ms | 219 ms | too short to reuse |
| 2,046 / 2,022 | 165 ms | 351 ms | too short to reuse |
| 3,633 / 4,002 | 156 ms | 744 ms | too short to reuse |
| 6,823 / 8,019 | 183 ms | 1,432 ms | 677 ms |

Up to about 1,000 tokens this setup, on the same machine as its client, answers within 20–70 ms
of what the hosted API took for a client 50 ms away from it. Jev's time then stays nearly flat
while this setup's grows by about 200 ms per 1,000 tokens read, so at 7,000–8,000 tokens Jev was
about eight times faster than reading in full here, and four times faster than reading with the
start reused. Because the benchmark repeated identical requests and TypeSafe does not say
whether it caches, it is not known whether Jev's long-state figures include cached reads. Jev's
serving hardware is not published; this setup is one RTX 5090 shared with chat, with the scoring
network on the CPU, which accounts for about a quarter of the time on long prompts read in full.

## Reuse of a prompt's start

A test game with a 14,000-token fixed text and a log that gained five turns before each of 12
decisions (`eval/game_test.py`):

| | Prompt | Reused | Total |
|---|---|---|---|
| first decision | 14,434 tokens | 0 | 3.0 s |
| decisions 2–12 with reuse | 14,555–15,841 tokens | 10,864 | 0.95–1.25 s |
| the same prompts read in full | | 0 | 2.66–3.01 s |

vLLM held back the last two matching 1,552-token blocks of every cache hit. Reading each prompt
in full twice gave probabilities up to 0.141 apart, and reuse against the first full read up to
0.119, so reuse stays within vLLM's own run-to-run variation. A game's start survived between
48,000 and 60,000 tokens of other traffic in vLLM's cache (`eval/reuse_probe.py --threshold`).
