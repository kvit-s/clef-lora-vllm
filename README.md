# Clef as a LoRA adapter on a vLLM server

[Clef](https://huggingface.co/Cloudflare/clef) is Cloudflare's decision model: given a situation
and a set of typed questions, it returns a probability for every allowed answer instead of
writing text. It is built from Qwen3.8-27B, and its weights differ from the base model's only in
decoder layers 40–63, by a change of rank 256. This repository turns that change back into a
1.4 GB LoRA adapter and adds what a vLLM server needs to answer Clef's requests with it, so that a
server already running Qwen3.8-27B for chat can also answer decisions at `POST /v1/systemone`
(the request format of Clef's release and of TypeSafe's Jev), without a second copy of the model
in graphics memory.

It has four parts:

- **the adapter**, published on Hugging Face as `<hf-user>/clef-lora-r256`, and rebuilt by
  `extract/extract_adapter.py` from the two public models;
- **a vLLM plugin** (`vllm-plugin/`) that writes the last layer's output for decision requests to
  shared memory, because vLLM's API returns only tokens;
- **a decision service** (`service/`) that tokenizes requests with Cloudflare's own code and answers
  in the same format as Clef's release and TypeSafe's Jev. Clef's scoring network runs either in the
  service on the CPU, from hidden states the plugin writes, or inside vLLM on the GPU
  (`CLEF_GPU_HEAD=1` and `--gpu-head`), keeping prompt starts in NVFP4 in system RAM or graphics memory.
  It reuses the start of prompts that grow at the end, such as a game's fixed rules plus a growing
  log, which on the reference setup took a 15,000-token decision from 2.7–3.0 s to 0.95–1.25 s;
- **evaluation** (`eval/`): 72 test records with the answers of Clef at full precision, and
  scripts that compare a server's hidden states and answers with them.

How it works, in detail: [docs/how-it-works.md](docs/how-it-works.md).

## What has been tested

| What | Status |
|---|---|
| Adapter fidelity | Rebuilt weights differ from Clef's by 0.17% at most, the size of bf16 rounding; what the factorisation leaves out is 0.85–1.005 times the rounding noise already in Clef's weights |
| The whole path on a GPU | One setup: RTX 5090, a 4-bit NVFP4 checkpoint, a DFlash speculative draft model, vLLM 0.27.2.dev0 (a fork). 84 of 85 known answers right, against 83 for Clef at full precision; median 212 ms per decision ([details](examples/rtx5090-nvfp4-dflash2/README.md)) |
| Stock vLLM 0.31.0 | Checked by reading its source only: every attribute the plugin uses exists with the same meaning, and 0.31 uses the model runner the GPU tests ran on. Not yet run on a GPU |
| Other quantizations (bf16, FP8, AWQ, GPTQ) | Not tested |
| Everything that needs no GPU | `eval/run-cpu-tests.sh`, about a minute |

If you run it on another setup, please open an issue or pull request with what
`eval/compare_hidden.py` and `eval/test_service.py` print, and it will be added to
[Tested setups](#tested-setups).

## How a decision is served

```
client ── POST /v1/systemone ──► decision service (service/service.py, CPU)
                                   │ 1. tokenize with Cloudflare's encode_record and Clef's tokenizer
                                   │ 2. POST /v1/completions: model "clef", token ids, max_tokens 1,
                                   ▼    vllm_xargs {"clef_capture": KEY}
                                 vLLM serving Qwen3.8-27B, with the adapter and the plugin
                                   │ the adapter applies to this request only; chat is unaffected
                                   ▼ the plugin writes the prompt's hidden states
                                 /dev/shm/clef-hidden/KEY.bin
                                   │
                                 3. scoring network on the CPU → answer in the SystemOne format
```

The service and vLLM must share a filesystem for the capture files: the same machine, and for
vLLM in Docker, `--ipc=host` or a mounted directory.

## Requirements

- vLLM serving Qwen3.8-27B, in any precision vLLM can apply a LoRA adapter to, with about 1.4 GB
  of graphics memory to spare for the adapter's buffers.
- For the service: Python 3.10 or later and `requirements.txt` (CPU-only torch is enough), about
  3 GB of disk for the release files, and a few CPU cores.
- For rebuilding the adapter yourself: about 25 minutes and 37 GB of downloads, of which only a
  few tensors are on disk at once.

## Quick start

**1. Get the adapter.** Download it:

```bash
hf download <hf-user>/clef-lora-r256 --include "adapter_*" --local-dir clef-adapter
```

or rebuild it from Cloudflare's and Qwen's releases:

```bash
pip install -r requirements.txt
python extract/extract_adapter.py clef-adapter
```

**2. Start vLLM with the plugin and the adapter.** Install the plugin into vLLM's environment
(`pip install ./vllm-plugin`, or build an image with `vllm-plugin/Dockerfile`), then start vLLM
as you normally do, adding:

```bash
export CLEF_CAPTURE_DIR=/dev/shm/clef-hidden   # where the plugin writes
export CLEF_LORA_LAYERS=40-63                  # adapter buffers only for the layers Clef changed
vllm serve Qwen/Qwen3.8-27B ... \
    --enable-lora --max-lora-rank 256 --max-loras 1 --max-cpu-loras 1 \
    --lora-modules clef=/path/to/clef-adapter --enable-prefix-caching
```

`examples/vllm-serve.sh` is a complete command; `examples/rtx5090-nvfp4-dflash2/` is the setup
that was measured. vLLM's log should show `clef_capture: patched` and
`LoRA buffers limited to decoder layers 40-63`.

To run the scoring network on the GPU as well, also set `CLEF_GPU_HEAD=1` and
`CLEF_RELEASE=/path/to/clef-release` (the directory from step 3, readable inside vLLM's container) and,
optionally, `CLEF_GPU_STORE_MB` (default 512), `CLEF_STORE_DEVICE` (`cpu`, the default, or `cuda`) and
`CLEF_GPU_HEAD_DTYPE` (`bfloat16`, the default, or `float32`). Leave the scoring about 0.5 GB of
graphics memory with the defaults, about 1.6 GB with the store on the GPU (on a full card, by lowering
`--kv-cache-memory-bytes`). Start the service in step 3 with `--gpu-head`.

**3. Start the decision service.**

```bash
python service/fetch_release.py clef-release      # 2.8 GB: tokenizer, scoring network, code
python service/service.py --release clef-release --vllm http://127.0.0.1:8000 \
    --capture-dir /dev/shm/clef-hidden --listen 127.0.0.1:8091
```

The service loads Cloudflare's `joint_schema_model.py` from the release directory rather than
from this repository. `service/clef-decider.service` is a systemd unit for it.

**4. Ask a question.**

```bash
curl -s 127.0.0.1:8091/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "clef",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle the message?",
                   "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}}}'
```

Status codes: 200 answered; 400 malformed, contains pictures, or longer than 16,384 tokens; 502
vLLM did not answer or does not serve the adapter; 504 vLLM answered but no hidden states arrived
within 10 seconds (the plugin is not loaded, or the capture directory is not shared).

**5. Check your setup against Clef at full precision.**

```bash
python eval/build_records.py                       # the 72 test records, from public datasets
python eval/test_service.py http://127.0.0.1:8091  # answers against the reference answers
hf download <hf-user>/clef-lora-r256 --include "reference/hidden/*" --local-dir .
python eval/compare_hidden.py http://127.0.0.1:8000 /dev/shm/clef-hidden reference/hidden \
    --base qwen3.8-27b --release clef-release --limit 16
```

`test_service.py` prints, per group, how many answers agree with Clef at full precision and how
many match the known right answer; Clef itself gets 83 of the 85 known answers right.
`compare_hidden.py` compares the last layer's output with and without the adapter against the
reference; with the adapter it should be far closer. `eval/speed_probe.py` measures what loading
the adapter costs your chat traffic: run it once without the adapter and once with it.

## Reusing the start of long prompts

When consecutive decisions share a long start, vLLM skips that part of the prompt, and the service
fills in what the scoring network needs from rows it kept from the earlier prompt; the answer is
the same as reading in full, to 1.2 × 10⁻⁷ on the CPU. For this to happen, send the state as one
string with the unchanging text first and new events appended at the end, and keep anything
rewritten each turn at the end too. Do not send the state as a JSON object: Clef writes JSON with
sorted keys, which can move the growing part before the fixed part. Questions can change freely,
since they come after the state. Each response's `timings.reused_tokens` shows what was reused.
vLLM reuses only whole blocks (1,552 tokens for this model) and, with a speculative draft model,
holds back the last one or two, so the shared start has to exceed about 3,100–4,700 tokens before
anything is reused. `CLEF_FULL_CACHE_HITS=1` in vLLM's environment makes the plugin skip the
held-back blocks for decisions only; on the measured setup that took a grown 64k decision from
1.46 s to 0.48 s, a grown 16k one from 0.91 s to 0.25 s and a grown 8k one from 0.70 s to 0.20 s,
with no measurable effect on chat. Details:
[docs/how-it-works.md](docs/how-it-works.md#how-much-vllm-reuses).

## Tested setups

| GPU | Checkpoint and cache | vLLM | Known answers right (of 85) | Agree with reference (of 87) | Median decision | Chat decode with the adapter |
|---|---|---|---|---|---|---|
| RTX 5090, 32 GB | NVFP4 (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`), NVFP4 cache, DFlash draft model | 0.27.2.dev0 fork | 84 (later runs 83, 85) | 84 | 212 ms | 15% slower (243 → 206 tokens/s) |

With the scoring network on the GPU (bf16, prompt starts in system RAM) and `CLEF_FULL_CACHE_HITS=1`,
the same setup answered a decision whose prompt had grown in 146–187 ms up to 16,000 tokens and 293 ms
at 32,000, and one read in full in 1.04 s at 8,000 tokens and 4.8 s at 32,000, for about 0.5 GB of
graphics memory ([details](docs/how-it-works.md#the-scoring-network-on-the-gpu)).

Measured earlier on the same setup, with the network on the CPU, `eval/latency_sweep.py` timed one question about prompts of 310 to 64,015
tokens: 177 ms at 310 tokens, 1.4 s at 8,000 and about 17 s at 64,000 when read in full, and about
0.7 s, 0.8–0.9 s and 1.5 s at 8,000, 16,000 and 64,000 tokens when the start was reused; with
`CLEF_FULL_CACHE_HITS=1` (below), 0.20 s, 0.25 s and 0.48 s. The
[worked example](examples/rtx5090-nvfp4-dflash2/README.md#latency-by-prompt-length-and-against-jev)
compares these with the published figures for TypeSafe's Jev.
| Clef at bf16, CPU (the reference) | — | — | 83 | — | — | — |

## Limits

- **Text and JSON only.** Clef's code tokenizes its prompt in pieces, and the result differs from
  tokenizing the whole prompt at once, so pictures cannot be passed through vLLM's chat endpoint
  as Clef was trained. vLLM's token-in, token-out endpoint (`/inference/v1/generate`) accepts
  exact token ids together with picture data and is the route to investigate.
- **16,384 tokens per decision by default**, the default of Cloudflare's released code. Longer
  prompts are refused rather than shortened. `service.py --max-length 65536` allows up to 64k,
  the context Cloudflare states for Clef; prompts of 32,000 and 64,000 tokens ran on the measured
  setup, but Clef's accuracy beyond 16,384 tokens has not been tested here.
- **Chat pays while the adapter is loaded**, whether or not decisions are being made: 15% slower
  decode and a shorter longest request on the measured setup, from vLLM's LoRA support and from
  the memory the adapter takes.
- **One adapter** (`--max-loras 1`).
- **The plugin depends on vLLM internals.** It logs a warning on versions other than the tested
  ones. With vLLM's adaptive draft verification enabled (a "dspark" option, off by default), it
  refuses to capture rather than risk misplacing rows.

## Repository layout

| Path | What it is |
|---|---|
| `extract/extract_adapter.py` | rebuilds the adapter from `Cloudflare/clef` and `Qwen/Qwen3.8-27B` at pinned revisions |
| `extract/delta_rank.py` | prints the singular values of Clef's change to any tensor |
| `vllm-plugin/` | the `clef_capture` plugin (capture, scoring on the GPU, its NVFP4 store of prompt starts), its tests and a Dockerfile |
| `service/service.py` | the decision service |
| `service/reuse.py` | the scoring network over kept rows, and the store of prompt starts |
| `service/fetch_release.py` | downloads the release files the service needs |
| `service/clef_code.py` | loads Cloudflare's `joint_schema_model.py` from the release |
| `service/mock_vllm.py` | a stand-in for vLLM, for testing without a GPU |
| `eval/build_records.py`, `eval/reference/` | the 72 test records and Clef's answers to them at full precision |
| `eval/test_service.py`, `eval/compare_hidden.py` | compare a server's answers and hidden states with the reference |
| `eval/game_test.py`, `eval/reuse_probe.py` | reuse of prompt starts: correctness, and when vLLM reuses |
| `eval/speed_probe.py` | chat speed, to measure the adapter's cost |
| `eval/latency_sweep.py` | decision latency against prompt length, read in full and with the start reused |
| `eval/reference.py` | how the reference answers were produced (Clef in bf16 on the CPU) |
| `eval/run-cpu-tests.sh` | every test that needs no GPU |
| `examples/` | a stock vLLM command, and the measured RTX 5090 setup |

## Related projects

Other projects published ways to run Clef in the week after its release:
[trevest/Clef-LoRA-for-Qwen3.8-27B-GGUF](https://huggingface.co/trevest/Clef-LoRA-for-Qwen3.8-27B-GGUF)
and its MLX version share one copy of Qwen3.8-27B through an adapter in a patched llama.cpp and in
MLX; [kurcontko/clef-NVFP4](https://huggingface.co/kurcontko/clef-NVFP4) is a vLLM plugin that
serves Clef as a model of its own, and
[taylorsm9/clef-vllm-spark](https://github.com/taylorsm9/clef-vllm-spark) runs it with the bf16
weights on a DGX Spark. This repository was written separately and uses none of their code.

## License

Apache-2.0, as are Clef and Qwen3.8-27B. See [NOTICE](NOTICE) for what the adapter is derived from
and for the datasets the evaluation downloads.
