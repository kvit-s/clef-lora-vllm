# How Clef runs as an adapter on a vLLM server

This document explains the mechanism behind this repository: why Cloudflare's Clef can be served
as a LoRA adapter on a vLLM server that already runs Qwen3.8-27B, how a decision request travels
through vLLM and the decision service, and how the service reuses the start of long prompts. The
README covers installation and the measured results.

## What Clef does

[Clef](https://huggingface.co/Cloudflare/clef) (Apache-2.0, released 2026-09-30) answers typed
questions about a situation with a probability for every allowed answer, instead of writing
text. A request describes the situation in `state`, any string or JSON value, and asks questions
of three types: `noul` (the probability that a statement is true), `choice` (one of several named
options) and `score` (a point on an ordered list of descriptions).

```json
{"model": "clef",
 "state": "Our checkout started returning errors and orders are blocked.",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team should handle the message?",
                  "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
   "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
   "outage": {"type": "noul", "instructions": "Is a service down?"}}}
```

The format is TypeSafe's "System One" API, which TypeSafe's hosted model Jev answers at
`POST /v1/systemone`; Clef's release implements the same format. Internally, Clef lays out
the state, every question and every option as one prompt of at most 16,384 tokens, runs the model
over it once, and takes the last layer's output at every prompt position (its hidden states,
5,120 numbers per token). A scoring network of its own (`joint_head.safetensors`, 256 MB: two
routing layers and four decoder layers of width 1,024) reads them and returns one score per
option, and a softmax per question turns the scores into probabilities.

Serving Clef from a vLLM server that runs Qwen3.8-27B therefore needs two things the server does
not do on its own: the forward pass has to use Clef's weights, and the hidden states have to reach
the scoring network, whereas vLLM's API returns only tokens.

## Clef is Qwen3.8-27B plus a rank-256 change

Clef's release has 1,184 weight tensors. Comparing each with the same tensor in Qwen3.8-27B (by
hashing its first 256 KiB, which HTTP range requests allow without downloading either model)
gives:

- the 333 tensors of the vision tower are identical;
- layers 0–39, the token embeddings, the output layer and every normalisation weight are
  identical;
- the 186 tensors that differ are the large matrices of layers 40–63: every attention projection,
  of both the linear-attention and the full-attention kind, and every feed-forward matrix. Each
  changed by between 3.6% and 22% of its own magnitude;
- Clef has no multi-token-prediction layers.

The differences are low-rank. In layer 43's query projection, for example, 99.95% of the
difference lies in its top 256 singular directions, and the singular values fall from 0.120 at
the 256th to 0.0063 at the 257th (`extract/delta_rank.py` prints this for any tensor). A LoRA
fine-tune merged into the weights leaves this pattern, because its change is the product of two
rank-256 matrices; the small remainder is the rounding added when the merged weights were stored
in bf16.

### Extracting the adapter

`extract/extract_adapter.py` downloads both versions of each changed tensor (37 GB in all, by
range requests), and factors the difference:

```
D = W_clef − W_base ≈ B · A,      A = √S · Vᵀ   (256 × in),     B = U · √S   (out × 256)
```

where U, S and V hold the top 256 singular vectors and values of D. A and B are rounded to bf16
and written in the PEFT format vLLM loads.

To check that rank 256 holds the whole change, the tool compares what the factorisation leaves
out with the rounding noise already present in Clef's own weights, which are stored in bf16 and
so are each off by up to half a step of 8-bit precision. For the 150 matrices larger than 256 in
both dimensions, what is left out measures 0.85 to 1.005 times that noise, and for the smaller
matrices a fraction of it. Weights rebuilt from the base and the adapter differ from Clef's by
0.17% at most. The adapter has 700,366,848 parameters, 1.40 GB in bf16.

The adapter reproduces Clef only on top of the bf16 base model. A server that holds the base model
quantized runs the quantized base plus a bf16 correction, and how far that moves the answers
depends on the quantization; `eval/compare_hidden.py` and `eval/test_service.py` measure it.

### Clef's tokenizer

Clef ships its own tokenizer file. Its vocabulary and merges are the base model's, but its rule for
splitting text before tokenizing treats combining accent marks the way older Qwen tokenizers did.
The decision service tokenizes with Clef's file and sends vLLM token ids, so vLLM's own tokenizer
is never used for decisions.

## The path of a decision

```
client ── POST /v1/systemone ──► decision service (service/service.py)
                                   │ 1. lay out and tokenize the record with Cloudflare's encode_record
                                   │ 2. POST /v1/completions: model "clef", the token ids, max_tokens 1,
                                   ▼    vllm_xargs {"clef_capture": KEY}
                                 vLLM, with the adapter and the clef_capture plugin
                                   │ the adapter is applied to this request only
                                   │ the plugin copies the prompt's last-layer rows to host memory
                                   ▼
                                 CAPTURE_DIR/KEY.bin
                                   │
                                 3. the service reads and deletes the file, runs the scoring network
                                    on the CPU, and answers in the SystemOne format
```

### Selecting the adapter per request

vLLM applies a LoRA adapter per request. Started with `--enable-lora` and
`--lora-modules clef=PATH`, it applies the adapter to requests that name the model `clef`, and
serves every other request from the base weights. Chat requests on the same server are
unaffected, apart from the cost of vLLM's LoRA support itself (see the README's results).

### The capture plugin

`vllm-plugin/clef_capture` is a vLLM general plugin, loaded in every vLLM process through the
`vllm.general_plugins` entry point. It wraps vLLM's GPU model runner. A decision request marks
itself with `vllm_xargs`, a field of vLLM's completion API whose contents reach the runner as
`sampling_params.extra_args`. After each forward pass the plugin looks for marked requests in the
batch, which costs one dictionary lookup when there are none. For a marked request it copies that
request's rows of the last layer's output to host memory and places them by prompt position,
because vLLM reads a long prompt in chunks and interleaves them with other requests' work. Once
every position has been seen, it writes the file.

The file is `CLEF_CAPTURE_DIR/KEY.bin` (default `/dev/shm/clef-hidden`):

| Part | Content |
|---|---|
| 8 bytes | `CLEFHS1\n` |
| 8 bytes | little-endian length of the JSON header |
| JSON header | `key`, `prompt_tokens`, `first_position`, `hidden_size`, `dtype` (`"bf16"`), `token_ids_sha256` (SHA-256 of all the prompt's token ids as little-endian int32) |
| data | (`prompt_tokens` − `first_position`) × `hidden_size` bf16 values, row-major |

`first_position` is above zero when vLLM served the start of the prompt from its prefix cache:
those positions were not computed and have no rows. The service checks the checksum against the
ids it sent, so a file can never be paired with another prompt. If the plugin cannot place a chunk,
it writes `KEY.err` with the reason instead. The service and vLLM therefore have to share a
filesystem: the same machine, and for vLLM in Docker, a shared `/dev/shm` (`--ipc=host`) or a
mounted directory.

vLLM has two GPU model runners, `vllm/v1/worker/gpu_model_runner.py` and
`vllm/v1/worker/gpu/model_runner.py`, and the plugin patches both. They keep per-request state in
different structures, so the plugin reads the mark from the runner's request table in the first,
and records it as each request is added in the second.

### Limiting adapter memory to the changed layers

vLLM reserves adapter buffers at the full `--max-lora-rank` for every module whose name matches
the adapter's target modules, in every layer, whether or not the adapter has weights there. For
Clef that would be 64 layers instead of 24, an estimated 3.7 GB instead of 1.4 GB. With
`CLEF_LORA_LAYERS=40-63` set, the plugin replaces vLLM's module matching so that only those
decoder layers get buffers and the vision tower's modules get none.

### The decision service

`service/service.py` checks the request and lays it out with Cloudflare's own `encode_record`,
loaded unchanged from the downloaded release, so the token ids and the position of every
question and option within them are exactly what Clef was trained on. A prompt over 16,384
tokens is refused with 400, because `encode_record` would otherwise shorten it by cutting off the
end of the state. Requests with pictures or video are refused as well (see the README's limits).

The scoring network runs in fp32 on the CPU, which keeps its 256 MB off the graphics card. On
8 threads it adds about 30 ms to a prompt under 600 tokens and about 130 ms to an 1,830-token
prompt with 77 options. Running it in bf16 instead moves probabilities by at most 0.003.

Each response has a `timings` object: `reused_tokens`, `backbone_seconds`, `head_seconds`,
`total_seconds`, and `read_twice` when vLLM had to be asked a second time.

## Reusing the start of long prompts

Some uses send prompts that share a long start: a game, for example, where each prompt holds a
large fixed text (rules and instructions), then a log that only grows at its end, then a question
about the next step. vLLM's prefix cache skips the shared start, but skipped positions are never
computed, so the plugin has no rows for them, while the scoring network reads every prompt
position.

### Why the skipped positions can be filled in exactly

The scoring network uses the positions before the first question in one way only: as the memory
read by its six attention layers (the two routing layers, and the cross-attention of the four
decoder layers). Each position's hidden state is normalised and projected to 1,024 numbers, and
each of the six layers projects that again into a key and a value, which makes twelve rows of
1,024 numbers per position. The backbone is causal, so a position's hidden state depends only on
the tokens up to it, and so do its twelve rows.

The service therefore keeps those rows (`service/reuse.py`). After each decision it keeps the rows
of the prompt's start, up to the first question, in one buffer per start. The buffer's blocks of 16
positions are named by a SHA-256 chain over all token ids up to the end of the block, so a block is
only ever used for a prompt that begins with exactly the same tokens. When vLLM skips positions,
the service takes a view of the kept rows (nothing is copied) and computes rows for the rest from
the new hidden states. The scoring network attends to the kept rows and the new rows as two parts,
so they are never joined into one tensor.

When the next prompt extends the kept start, as a growing game log does, the service appends the
new rows to the same buffer. Such a prompt departs from the previous one in its last block or two,
where the old state ended and Clef's fixed text after it began, so the store rewrites up to 64
blocks at the end of a buffer in place. That costs the older prompt nothing as long as vLLM
reads at least the last 1,552 tokens of a cached prompt again, as the measured server does (it
reads 3,104). A prompt that departs earlier gets a buffer of its own,
which copies the shared start once. Buffers have a quarter of spare room and grow when it runs
out. The rows are fp32, about 48 KB per token; the default `--reuse-memory-mb 2048` holds about
43,000 tokens, and over that the least recently used buffers are dropped whole. If vLLM skipped
positions the service no longer holds, after a restart for example, the service sends the request
again with a `cache_salt` unique to it, which makes vLLM compute every position.
`--reuse-memory-mb 0` turns reuse off.

On a 64,000-token prompt whose start was reused, keeping the rows in one buffer took the scoring
network's time from about 0.86 s to about 0.37 s: fetching the kept rows went from 317 ms to 3 ms,
joining them with the new rows (119 ms) disappeared, and attention over two parts took 113–148 ms
against 192 ms over one joined tensor.

`reuse.forward` follows Cloudflare's `JointSchemaHead.forward` step for step. On the 72 test
records its probabilities match Cloudflare's to 1.2 × 10⁻⁷, whether the start is computed or
taken from the store (`service/test_reuse.py`).

### How much vLLM reuses

For Qwen3.8-27B, vLLM's prefix cache works in blocks of 1,552 tokens, because the linear-attention
layers' state can only be saved at block boundaries. On the server this was measured on, vLLM
held back the last two matching blocks of every cache hit:

```
reused tokens = (whole 1,552-token blocks matching an earlier prompt − 2) × 1,552
```

One held-back block comes from speculative decoding: vLLM drops the last matching block of every
cache hit, because EAGLE-style draft models compute a block's draft cache from the token after it
(`vllm/v1/core/kv_cache_coordinator.py`). The other comes from the DFlash2 image the server runs,
whose release patch makes the linear-attention lookup search one block shorter, because the last
matched state may have been written over rejected draft tokens while generating
(`MambaManager.find_longest_cache_hit` in `vllm/v1/core/single_type_kv_cache_manager.py`). Stock
vLLM 0.31 has the first and not the second.

Neither applies to a decision: it generates one token and uses no draft tokens, and its prompt's
states are all written while reading prompts. With `CLEF_FULL_CACHE_HITS=1` in vLLM's environment,
the capture plugin changes three things for the requests it captures, and only for those:

- `KVCacheManager.get_computed_blocks`: for the length of the call, the coordinator gets copies of
  its cache groups with the drop flag off, so neither block is held back;
- the same call caps the reuse at the start of the first question, which the service passes as
  `clef_reuse_limit`, because the scoring network reads the hidden states of the question positions
  directly;
- `Scheduler._mamba_block_aligned_split`: with a draft model, the scheduler stops reading a prompt
  one block short of its last block boundary, so the linear-attention state there is never saved
  and the next, longer prompt cannot reuse it. For decisions it stops at the last boundary.

What remains is the block size itself: reuse grows one 1,552-token block at a time, so a prompt
that grows a little at a time re-reads its new part plus up to one block, and vLLM never treats the
whole prompt as cached.

Measured on the RTX 5090 setup (examples/rtx5090-nvfp4-dflash2), with the store that keeps each
start in one buffer:

| Prompt | Grown by four lines, option on | Identical again, option on | Grown, option off |
|---:|---:|---:|---:|
| 4,089 tokens | 230 ms | 223 ms | 731–837 ms (nothing reused) |
| 8,106 tokens | 199 ms | 195 ms | 701 ms |
| 16,089 tokens | 246 ms | 217 ms | 908 ms |
| 32,101 tokens | 395 ms | 387 ms | 1,209 ms |
| 64,102 tokens | 478 ms | 413 ms | 1,456 ms |

On a game of 14,400 to 15,800 tokens, every decision after the first reused exactly the whole
matching blocks; decisions differed from reading in full by at most 0.090 in probability while two
full reads differed by up to 0.161, and all twelve chose the same step. Twelve decisions sent while
a chat streamed took a median of 0.35 s, and the chat text was correct. Chat speed did not change
measurably: on this server it varies from one vLLM start to the next by about ±10% with either
image (205–253 tokens/s for the patched image over two starts, 219–235 for the unpatched one over
two starts), and the draft model's acceptance on a fixed prompt varied with it (54–62%).

The prefix cache is shared with chat, and the least recently used blocks go first. On the
measured server, with 194,648 token slots, a game's start survived 48,000 tokens of other traffic
between two decisions and was gone after 60,000. The next decision then reads in full and caches
the start again.

### Writing prompts that can be reused

Clef always lays out a prompt in the same order: a fixed 36-token preamble, the state, every
question with its options, and an 18-token ending. To be reused, a prompt should therefore:

- send the state as one string with the fixed text first, adding new events only at its end;
  anything rewritten each turn, such as a turn counter or a summary, also belongs at the end;
- avoid sending the state as a JSON object, because Clef's layout writes JSON with its keys sorted
  alphabetically, which would put a `history` key before a `rules` key;
- change the questions as needed, since they come after the state.

Asking the same questions about a new state each time, as a classifier does, reuses nothing,
because the state comes before the questions.

## The scoring network on the GPU

With `CLEF_GPU_HEAD=1` in vLLM's environment and `--gpu-head` on the decision service, Clef's scoring
network runs inside vLLM's process on the GPU, and the service only tokenizes requests and formats
answers. Nothing is sent back to the service but the option scores.

What happens for one decision:

1. The service sends, with the token ids, where the questions and options lie in the prompt
   (`vllm_xargs` `clef_head`, as JSON text, since vLLM accepts only flat values there: the first
   question position rounded down to 16, and each question's type, span and option spans).
2. Each time vLLM has computed a chunk of the prompt (at most 2,048 positions), the plugin turns the
   chunk's hidden states into the twelve key and value rows per position (`head.memory_kv`). Rows of
   positions before the first question are packed into NVFP4 (`packed.py`: 6,960 bytes per position
   instead of 49,152 in fp32) and kept in a store of prompt starts (`gpu_store.py`); the rows and hidden
   states of the question positions, a few hundred, are kept on the GPU for this request only.
3. If vLLM served the start of the prompt from its cache, the plugin finds that start's rows in its
   store by the chain of token-id hashes, as the CPU store does; if they are not there, it reports
   "not kept" and the service asks again with a `cache_salt`, which makes vLLM read the whole prompt.
4. When every position has been seen, the plugin runs the rest of the network (`head.forward`) over
   the stored rows and the kept ones, and writes the scores to `CAPTURE_DIR/KEY.json`. The stored rows
   reach the attention 4,096 positions at a time (copied up and unpacked per slice), with the softmax
   kept as a running maximum and sum in fp32, so graphics memory does not grow with the prompt.

Rows are computed 512 positions at a time, and packed rows are copied down to the store in RAM
without making vLLM wait; anything that reads them on the CPU side first waits for the copy.

The store follows the CPU store's rules (one buffer per prompt start, a growing prompt extended in
place with its last blocks rewritten, a departing one given its own buffer, least recently used
dropped first), within `CLEF_GPU_STORE_MB` (default 512). It lives in pinned system RAM by default
(`CLEF_STORE_DEVICE=cpu`) or in graphics memory (`CLEF_STORE_DEVICE=cuda`). The network runs in bf16 by
default (`CLEF_GPU_HEAD_DTYPE=bfloat16`), with its attention softmax in fp32; in Cloudflare's own run
on the 72 test records, the network in bf16 moved probabilities by at most 0.0027 against fp32 and
changed no answer. It is loaded from Clef's release directory, mounted into vLLM's container and
named by `CLEF_RELEASE`, on the first decision.

Storing the rows in NVFP4 was checked on real hidden states before it was built: on six prompts of
16,100 to 70,800 tokens, where 98% of each prompt came from stored rows, it moved probabilities by
at most 0.0003 and changed none of 18 answers (`eval/row_precision.py`); bf16 and FP8 moved nothing
measurable. On the CPU, the plugin's scorer in fp32 agrees with the service's CPU path given the same
NVFP4 rows to 5 × 10⁻⁷, and in bf16 to 0.005 (`vllm-plugin/test_gpu_head.py`).

Measured on the RTX 5090 setup, with `CLEF_FULL_CACHE_HITS=1` throughout, three ways of scoring:

| Prompt | On the CPU (service): read in full / start reused | GPU, fp32, store on the GPU | GPU, bf16, store in RAM (the default) |
|---:|---:|---:|---:|
| 4,002 tokens | 745 / 230 ms | 496 / 178 ms | 506 / 146 ms |
| 8,019 tokens | 1,598 / 199 ms | 1,036 / 175–178 ms | 1,035 / 170 ms |
| 16,002 tokens | 3,199 / 246 ms | 2,083–2,150 / 181–193 ms | 2,121 / 187 ms |
| 32,013 tokens | 6,953 / 395 ms | 4,824–4,828 / 310–311 ms | 4,762 / 293 ms |

| | On the CPU | GPU, fp32, store on the GPU | GPU, bf16, store in RAM |
|---|---|---|---|
| graphics memory for the scoring | none | about 1.6 GB on the card after a sweep to 32k | about 490 MiB on the card (PyTorch's own memory grew 350 MB at most), the same from 4k to 32k |
| vLLM's key-value reservation | 5 GiB | 4 GiB | 4.75 GiB |
| longest chat request | 194,000 tokens | 147,000 tokens | 181,000 tokens (182,279 slots) |
| system RAM | the service: 1.0 GB idle; 4.8 GB at most while reading a 32k prompt in full, 2.6 GB kept afterwards (2.9 and 1.8 GB at 16k) | the service: 0.57 GB | the service: 0.57 GB; vLLM's process about 1 GB more, mostly the store |
| known test answers right | 84 of 85 | 84 of 85 | 84 of 85 |

The network's own time on the GPU was 13–52 ms for a decision with a reused start; for one read in
full it was 108 ms at 32k with the store on the GPU and 195 ms with the store in RAM. On a 15,000-token
game, reused decisions differed from full reads by at most 0.158 in probability (fp32, store on the
GPU; 0.086 with the defaults), against 0.162 (0.114) between two full reads, and every decision chose
the same step; fifteen decisions sent while a chat streamed took a median of 0.27 s, and the chat text
was correct. Chat speed and speculative acceptance stayed within the ±10% that varies from one vLLM
start to the next (205–264 tokens/s, 57–67%); with the defaults at 4.75 GiB, chat decoded at 263
tokens/s, and PCIe traffic during it stayed at the level of the idle desktop.

On the measured card, which has little free memory, the scoring's graphics memory has to come out of
vLLM's key-value reservation, or Windows moves card memory to system RAM. Reading long prompts makes
vLLM itself hold more memory too: with the same decisions up to 16k, the production configuration (CPU
scoring, 5 GiB) went from 31,663 to 31,897 MiB on the card, and the defaults at 4.75 GiB from 31,183 to
31,675 MiB, about 220 MiB below it.

## Further speed-ups considered and not built

These were examined on the RTX 5090 setup and left aside, because a cached decision there already
takes about 0.2 s up to 16k tokens and about 0.3 s at 32k with `CLEF_FULL_CACHE_HITS=1` and the
scoring network on the GPU. They are recorded with what was measured and what is only estimated.

### Resuming inside a 1,552-token page

Reuse advances one 1,552-token page at a time, so a prompt that grows a little at a time re-reads
its new part plus up to one page; in a 15,000-token game that moved the decision time between 0.26
and 0.57 s. Three ways to resume closer to where the next prompt departs:

- **Finer matching and priming.** The DFlash2 fork of vLLM has `--prefix-match-unit N`, which lets a
  cache hit land on any N-token boundary inside a page, but it saves a linear-attention state there
  only at the end of a prompt. A decision prompt ends with its questions, after the point where the
  next prompt departs, so the service would send a throwaway "priming" request after each answer:
  the same prompt cut shortly before its first question, which makes vLLM save a point there. The
  service and plugin support this (`--prime-margin`, `vllm_xargs` `clef_prime`), tested only against
  the stand-in for vLLM (`eval/run-cpu-tests.sh`, step 4). On the GPU it cannot run: with
  `--prefix-match-unit 16` the engine died on its first chat request, because the DFlash2 draft
  model has sliding-window attention layers and the fork's `SlidingWindowManager` "does not support
  fine-grained (partial) cache hits". It would need that manager extended, or a server without such
  a draft model.
- **Padding to a page boundary and priming.** After each decision, the service would append filler
  to the state so that the state ends exactly on a page boundary (for example a line marked as a
  repeat, followed by part of the original description), prime the prompt up to that boundary in the
  background, and insert the same filler at the same point in every later request whose state starts
  with that text, so the client keeps sending its plain state. The next decision would then read
  only what changed and its questions: an estimated 0.2 s at 16k and 0.3 s at 64k, steady. The
  filler stays in the prompt for good, on average 776 tokens per step, so with about 200 tokens of
  change per step the prompt grows about five times faster and reaches a 16,384-token limit after
  about a fifth as many steps; a cap on the filler per step trades some of the gain for slower
  growth. Clef has not seen such filler, so its effect on the answers would have to be measured.
- **An FP8 cache.** Pages are as large as one linear-attention state; with keys and values in FP8
  instead of NVFP4 a page holds about 800 tokens instead of 1,552, halving the re-read, but the
  same 5 GiB holds about 100,000 tokens instead of 194,648, so a cached start survives about half
  as much other traffic. Untested.
