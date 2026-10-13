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

## What the overlay changes

`compose.clef.yaml` in this directory is an overlay for the DFlash2 project's own compose file;
its comments say how to build the image and start it. It also needs
`MAX_NUM_BATCHED_TOKENS=2048` in the project's `.env`: with the project's 4,096-token steps, two
long prompts read at once leave vLLM holding 600–880 MB more memory than before, which the full
card does not have, and Windows then moves part of vLLM's memory into system RAM until vLLM
restarts.

Decisions need two things on the card: the adapter's buffers, 1.4 GB in bf16 plus vLLM's
bookkeeping, and the scoring network with its working memory, about 490 MiB. Both come out of the
key-value reservation, which is 4.75 GiB instead of 6.5. vLLM fits 182,279 token slots in it, so the
longest request is 181,000 tokens instead of 262,144, and six short requests fit at once instead of
eight. Measured on 2026-10-12:

| | Without the adapter | With the adapter and the scoring network |
|---|---|---|
| card memory in use (`nvidia-smi`) | 30,936–31,303 MiB | 31,690 MiB |
| card memory Windows holds in system RAM | 333–570 MB | 2,959 MB |
| PCIe traffic to the card during a chat | | median 41 MB/s, at most 121 MB/s |

The extra memory Windows holds is not read back during chat: PCIe traffic stays at the level of
the idle desktop, against about 8,000 MB/s when the card pages.

Whether the adapter slows chat could not be measured, because chat speed on this server varies
between vLLM starts by more than any difference it might cause. The draft model's acceptance rate
on a fixed test prompt is the same on every run within one start and differs from one start to
the next, and decode speed follows it:

| Draft tokens accepted | Decode, tokens/s, with the adapter |
|---|---|
| 51% | 214–236 |
| 54–57% | 205–235 |
| 62–67% | 253–264 |

One start without the adapter decoded the same prompt at 243 tokens/s, inside that range.

Several chats at once, with the DFlash2 project's concurrency benchmark (five essay prompts of 512
tokens at temperature 0, on which about 30% of draft tokens are accepted):

| Streams | Total, tokens/s | Before the adapter |
|---:|---:|---:|
| 1 | 141–149 | 125 |
| 2 | 277–298 | 244 |
| 4 | 507–567 | 456–470 |
| 8 requested, 6 running | 568–581 | 801, all eight running |

## Decisions

The 72 test records (`eval/`), with Clef in bf16 on the CPU as the reference, gave 85 of the 85
known answers right, against 83 for the reference, and 85 of 87 answers agreed with the reference.
Answers close to an even split can flip between runs; other runs scored 83 and 84. The last
layer's output, compared with the reference over the first 16 ARC records (`compare-hidden.json`),
had a relative error of 0.38–0.45 and a mean cosine similarity per position of 0.906–0.932 with the
adapter, against 1.10–1.19 and 0.34–0.42 without it; the remaining distance comes from the 4-bit
checkpoint and cache rather than from the adapter.

Over the 72 records the median decision took 165 ms and the longest 421 ms. A one-token
completion without the adapter takes 123 ms for a 217-token prompt on the same server, so most of
a short decision's time is vLLM's fixed cost per request.

## Latency by prompt length

One 6-option choice question about a support transcript of growing length, sent from the same
machine with `eval/latency_sweep.py` and the service at `--max-length 32768`:

| Prompt tokens | Read in full | The identical request again | Transcript grown by four lines | Tokens reused |
|---:|---:|---:|---:|---:|
| 310 | 137 ms | 141 ms | 147 ms | 0 |
| 739 | 146 ms | 140 ms | 143 ms | 0 |
| 1,015 | 151 ms | 141 ms | 147 ms | 0 |
| 2,022 | 330 ms | 144 ms | 148 ms | 1,552 |
| 4,002 | 475 ms | 142 ms | 145 ms | 3,104 |
| 8,019 | 1,032 ms | 152 ms | 157 ms | 7,760 |
| 16,002 | 2,054 ms | 179 ms | 180 ms | 15,520 |
| 32,013 | 4,926 ms | 308 ms | 311 ms | 31,040 |

Clef's accuracy beyond 16,384 tokens, the default limit of Cloudflare's released code, has not
been tested.

## Reuse on a game

A test game with a 14,000-token fixed text and a log that gained five turns before each of 12
decisions, each asking which of six steps to take and whether energy is low
(`eval/game_test.py`):

| | Prompt | Reused | Total | Of which the scoring network |
|---|---|---|---|---|
| first decision, nothing to reuse | 14,434 tokens | 0 | 2.06 s | 97 ms |
| decisions 2–12 | 14,555–15,841 tokens | 13,968 | 0.17–0.45 s | 27–48 ms |

The time grows within the game until the state passes the next 1,552-token block boundary. Sent N
at a time, with the same state and different questions in each round:

| At once | New state, nothing cached | Five new turns | Same state, new questions |
|---:|---:|---:|---:|
| 1 | 2.53 s | 0.21 s | 0.23 s |
| 2 | 2.04 s | 0.34 s | 0.27 s |
| 4 | 2.10 s | 0.55 s | 0.55 s |
| 8 | 2.31–2.52 s | 0.54–1.04 s | 0.52–1.03 s |

When nothing is cached yet, only the first request reads the start; the others wait in vLLM's queue
and then reuse it, so eight decisions take about as long as one full read. A game's start survived
48,000 tokens of other traffic in vLLM's cache between two decisions and was gone after 60,000
(`eval/reuse_probe.py --threshold`, measured with a 5 GiB cache of 194,648 token slots).
