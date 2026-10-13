# Clef as a LoRA adapter on a vLLM server

[Clef](https://huggingface.co/Cloudflare/clef) is Cloudflare's decision model: you send it a
situation and a set of typed questions, and it returns a probability for every allowed answer
instead of writing text. It is Qwen3.8-27B with decoder layers 40–63 changed by a rank-256
fine-tune, plus a small scoring network.

This repository turns that change back into a 1.4 GB LoRA adapter and adds a vLLM plugin and a
small service, so that a vLLM server already running Qwen3.8-27B for chat also answers Clef's
requests at `POST /v1/systemone` (the request format of Clef's release and of TypeSafe's Jev),
from the same copy of the weights. Decisions whose prompt extends an earlier one, such as a
game's fixed rules followed by a growing log, reuse the earlier prompt's start: on an RTX 5090 a
16,000-token decision grown by four lines takes 180 ms instead of 2.1 s.

How it works, in detail: [docs/how-it-works.md](docs/how-it-works.md).

## How a decision is served

```
client ── POST /v1/systemone ──► decision service (service/service.py)
                                   │ tokenizes with Cloudflare's own code and Clef's tokenizer
                                   ▼ POST /v1/completions: model "clef", token ids, max_tokens 1
                                 vLLM serving Qwen3.8-27B, with the adapter and the plugin
                                   │ the adapter applies to this request only; chat is unaffected
                                   │ the plugin runs Clef's scoring network on the GPU
                                   ▼
                                 $CLEF_CAPTURE_DIR/KEY.json: one score per option
                                   │
                                 decision service → answer in the SystemOne format
```

The service and vLLM must share the capture directory: run them on the same machine, and for vLLM
in Docker, mount the directory into the container.

## Requirements

- vLLM serving Qwen3.8-27B, in any precision vLLM can apply a LoRA adapter to, with about 2 GB of
  graphics memory to spare: 1.4 GB for the adapter's buffers and about 0.5 GB for the scoring
  network. On a full card, lower `--kv-cache-memory-bytes` by that much.
- About 0.5 GB of system RAM for the starts of recent prompts.
- For the service: Python 3.10 or later and `requirements.txt` (CPU-only torch is enough).
- About 3 GB of disk for Clef's release files, which both vLLM and the service read.

## Quick start

**1. Get the adapter.** Download it:

```bash
hf download <hf-user>/clef-lora-r256 --include "adapter_*" --local-dir clef-adapter
```

or rebuild it from Cloudflare's and Qwen's releases (about 25 minutes and 37 GB of downloads, of
which only a few tensors are on disk at once):

```bash
pip install -r requirements.txt
python extract/extract_adapter.py clef-adapter
```

**2. Get Clef's release files** (2.8 GB: the tokenizer, the scoring network and Cloudflare's code):

```bash
python service/fetch_release.py clef-release
```

**3. Start vLLM with the plugin and the adapter.** Install the plugin into vLLM's environment
(`pip install ./vllm-plugin`, or build an image with `vllm-plugin/Dockerfile`), then start vLLM
as you normally do, adding:

```bash
export CLEF_CAPTURE_DIR=/dev/shm/clef-hidden    # where the plugin writes each decision's scores
export CLEF_LORA_LAYERS=40-63                   # adapter buffers only for the layers Clef changed
export CLEF_GPU_HEAD=1                          # run Clef's scoring network inside vLLM
export CLEF_RELEASE=/path/to/clef-release       # from step 2, readable inside vLLM's container
export CLEF_GPU_STORE_MB=400                    # system RAM for the starts of recent prompts
export CLEF_FULL_CACHE_HITS=1                   # with a speculative draft model, see below
vllm serve Qwen/Qwen3.8-27B ... \
    --enable-lora --max-lora-rank 256 --max-loras 1 --max-cpu-loras 1 \
    --lora-modules clef=/path/to/clef-adapter --enable-prefix-caching
```

vLLM's log should show `clef_capture: patched` and `LoRA buffers limited to decoder layers 40-63`.
`examples/vllm-serve.sh` is a complete command, and `examples/rtx5090-nvfp4-dflash2/` is the
Docker Compose overlay of the tested setup.

**4. Start the decision service.**

```bash
python service/service.py --release clef-release --vllm http://127.0.0.1:8000 \
    --capture-dir /dev/shm/clef-hidden --listen 127.0.0.1:8091 --gpu-head --max-length 32768
```

`--max-length` is the longest decision prompt accepted, in tokens; without it the limit is 16,384,
the default of Cloudflare's code. `service/clef-decider.service` is a systemd unit for the service.

**5. Ask a question.**

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

| Status | Meaning |
|---|---|
| 200 | answered |
| 400 | the request is malformed, contains pictures, or is longer than `--max-length` |
| 502 | vLLM did not answer, does not serve the adapter, or the scoring failed |
| 504 | vLLM answered, but no scores arrived within 10 seconds: the plugin is not loaded, or the capture directory is not shared |

**6. Check your setup against Clef at full precision.**

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
reference; with the adapter it should be far closer.

## Writing prompts that reuse their start

vLLM skips the part of a prompt it has already read, and the plugin fills in what the scoring
network needs for that part from what it kept of the earlier prompt; the answer is the same as
reading in full. For this to happen:

- Send the state as one string, with the unchanging text first and new events appended at the
  end. Keep anything rewritten each turn at the end too.
- Do not send the state as a JSON object: Clef writes JSON with sorted keys, which can move the
  growing part in front of the fixed part.
- Questions can change freely, since they come after the state.

vLLM reuses only whole blocks, 1,552 tokens for this model, so a decision reads up to 1,551
tokens of the older prompt again, plus whatever is new. With a speculative draft model, vLLM also
holds back the last one or two matching blocks of every cache hit; `CLEF_FULL_CACHE_HITS=1` turns
that off for decisions only, leaving chat as it was. Each response's `timings.reused_tokens`
shows how many tokens were reused.

## Tested setup

One setup so far: an RTX 5090 (32 GB) running `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
(4-bit) with a 4-bit key-value cache and a DFlash speculative draft model, in the DFlash2 image
(a fork of vLLM 0.27.2.dev0), with the settings in step 3.

- **Answers:** on the 72 test records, 83–85 of the 85 known answers right across runs, against 83
  for Clef at full precision; answers close to an even split can flip between runs.
- **Time per decision:**

  | Prompt tokens | Read in full | Grown by four lines |
  |---:|---:|---:|
  | 310 | 137 ms | 147 ms |
  | 4,002 | 475 ms | 145 ms |
  | 16,002 | 2,054 ms | 180 ms |
  | 32,013 | 4,926 ms | 311 ms |

- **Cost to chat:** the adapter and the scoring network came out of the key-value cache (6.5 →
  4.75 GiB), so the longest chat request is 181,000 tokens instead of 262,144, and six requests
  run at once instead of eight. Whether the adapter slows decoding could not be separated from
  the variation between vLLM starts (205–264 tokens/s for one conversation).

Details: [examples/rtx5090-nvfp4-dflash2/README.md](examples/rtx5090-nvfp4-dflash2/README.md).

Stock vLLM 0.31.0 has been checked only by reading its source: every attribute the plugin uses
exists with the same meaning. Other GPUs and quantizations (bf16, FP8, AWQ, GPTQ) have not been
tested. If you run it on another setup, please open an issue or pull request with what
`eval/compare_hidden.py` and `eval/test_service.py` print.

## Limits

- **Text and JSON only.** Clef's code tokenizes its prompt in pieces, and the result differs from
  tokenizing the whole prompt at once, so pictures cannot be passed through vLLM's chat endpoint
  as Clef was trained.
- **Prompt length.** 32,768 tokens on the tested setup; Clef's accuracy beyond 16,384 tokens has
  not been tested.
- **Chat pays while the adapter is loaded**, whether or not decisions are being made: the memory
  the adapter and the scoring network take comes out of chat's key-value cache.
- **One adapter** (`--max-loras 1`).
- **The plugin depends on vLLM internals.** It logs a warning on versions other than the tested
  ones.

## Repository layout

| Path | What it is |
|---|---|
| `extract/extract_adapter.py` | rebuilds the adapter from `Cloudflare/clef` and `Qwen/Qwen3.8-27B` at pinned revisions |
| `vllm-plugin/` | the `clef_capture` plugin (scoring on the GPU and the store of prompt starts), its tests and a Dockerfile |
| `service/` | the decision service, `fetch_release.py` for Clef's release files, and a systemd unit |
| `eval/` | the 72 test records with Clef's answers at full precision, and scripts that compare a server with them |
| `examples/` | a stock vLLM command, and the tested RTX 5090 setup |

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
