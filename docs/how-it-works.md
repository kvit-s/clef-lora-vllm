# How Clef runs as an adapter on a vLLM server

This document explains the mechanism behind this repository: why Cloudflare's Clef can be served
as a LoRA adapter on a vLLM server that already runs Qwen3.8-27B, how a decision request travels
through vLLM and the decision service, and how decisions reuse the start of long prompts. The
README covers installation, and `examples/rtx5090-nvfp4-dflash2/` the measured results.

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
the state, every question and every option as one prompt, runs the model over it once, and takes
the last layer's output at every prompt position (its hidden states: one vector of 5,120 numbers
per token). A scoring network of its own (`joint_head.safetensors`, 256 MB: two routing layers and
four decoder layers of width 1,024) reads them and returns one score per option, and a softmax per
question turns the scores into probabilities.

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
                                   ▼    vllm_xargs: a key, and where the questions and options lie
                                 vLLM, with the adapter and the clef_capture plugin
                                   │ the adapter is applied to this request only
                                   │ the plugin runs Clef's scoring network on each chunk's
                                   │ last-layer output, on the GPU
                                   ▼
                                 CAPTURE_DIR/KEY.json: one score per option
                                   │
                                 3. the service reads and deletes the file and answers in the
                                    SystemOne format
```

### Selecting the adapter per request

vLLM applies a LoRA adapter per request. Started with `--enable-lora` and
`--lora-modules clef=PATH`, it applies the adapter to requests that name the model `clef`, and
serves every other request from the base weights, in the same batches.

### Limiting adapter memory to the changed layers

vLLM reserves adapter buffers at the full `--max-lora-rank` for every module whose name matches
the adapter's target modules, in every layer, whether or not the adapter has weights there. For
Clef that would be 64 layers instead of 24, an estimated 3.7 GB instead of 1.4 GB. With
`CLEF_LORA_LAYERS=40-63` set, the plugin replaces vLLM's module matching so that only those
decoder layers get buffers and the vision tower's modules get none.

### The plugin

`vllm-plugin/clef_capture` is a vLLM general plugin, loaded in every vLLM process through the
`vllm.general_plugins` entry point. It wraps vLLM's GPU model runner, the part that executes each
forward pass. A decision request marks itself with `vllm_xargs`, a field of vLLM's completion API
whose contents reach the runner as `sampling_params.extra_args`: `clef_capture` names a file for the
answer, and `clef_head` says, as JSON text since vLLM accepts only flat values there, where the first
question starts (rounded down to 16) and each question's type, span and option spans. After each
forward pass the plugin looks for marked requests in the batch, which costs one dictionary lookup
when there are none.

vLLM reads a long prompt in chunks of at most 2,048 positions and interleaves them with other
requests' work. For each chunk of a marked request, the plugin:

1. takes that request's rows of the last layer's output and turns them into the twelve key and
   value rows per position that the scoring network's attention layers read (`head.memory_kv`),
   512 positions at a time;
2. packs the rows of positions before the first question into NVFP4 and adds them to a store of
   prompt starts in system RAM (below), copying them down without making vLLM wait;
3. keeps the rows and hidden states of the question positions, a few hundred, on the GPU for this
   request only.

When every position has been seen, it runs the rest of the network (`head.forward`) and writes the
scores to `CLEF_CAPTURE_DIR/KEY.json` (default `/dev/shm/clef-hidden`), or `KEY.err` with the reason
if it cannot. The stored rows reach the attention 4,096 positions at a time, copied up and unpacked
per slice, with the softmax kept as a running maximum and sum in fp32, so graphics memory does not
grow with the prompt: the network and its working memory take about 490 MiB, the same for 4,000-
and 32,000-token prompts. The network runs in bf16, loaded from Clef's release directory
(`CLEF_RELEASE`) on the first decision after vLLM starts, which therefore takes about 1.2 s.

The service and vLLM have to share the capture directory: the same machine, and for vLLM in
Docker, a mounted directory. The plugin creates the directory when it is missing and makes it
writable by everyone, since the service, which deletes each file after reading it, may run as
another user.

vLLM has two GPU model runners, `vllm/v1/worker/gpu_model_runner.py` and
`vllm/v1/worker/gpu/model_runner.py`, and the plugin patches both. They keep per-request state in
different structures, so the plugin reads the mark from the runner's request table in the first,
and records it as each request is added in the second.

### The decision service

`service/service.py`, started with `--gpu-head`, checks the request and lays it out with
Cloudflare's own `encode_record`, loaded unchanged from the downloaded release, so the token ids
and the position of every question and option within them are exactly what Clef was trained on. A
prompt over `--max-length` tokens (default 16,384) is refused with 400, because `encode_record`
would otherwise shorten it by cutting off the end of the state. Requests with pictures or video are
refused as well (see the README's limits).

Each response has a `timings` object: `reused_tokens`, `backbone_seconds` (vLLM, including the
scoring), `head_seconds` (the scoring network's share of it), `total_seconds`, and `read_twice` when
vLLM had to be asked a second time.

### Hidden states for checking

A request marked with `clef_capture` but without `clef_head` gets the prompt's hidden states
instead of scores, in `CAPTURE_DIR/KEY.bin`. `eval/compare_hidden.py` uses this to compare a
server's last-layer output with Clef's at full precision:

| Part | Content |
|---|---|
| 8 bytes | `CLEFHS1\n` |
| 8 bytes | little-endian length of the JSON header |
| JSON header | `key`, `prompt_tokens`, `first_position`, `hidden_size`, `dtype` (`"bf16"`), `token_ids_sha256` (SHA-256 of all the prompt's token ids as little-endian int32) |
| data | (`prompt_tokens` − `first_position`) × `hidden_size` bf16 values, row-major |

## Reusing the start of long prompts

Some uses send prompts that share a long start: a game, for example, where each prompt holds a
large fixed text (rules and instructions), then a log that only grows at its end, then a question
about the next step. vLLM's prefix cache skips the part of a prompt it has already read, so the
model produces no hidden states for it, while the scoring network reads every prompt position.

### Why the skipped positions can be filled in exactly

The scoring network uses the positions before the first question in one way only: as the memory
read by its six attention layers (the two routing layers, and the cross-attention of the four
decoder layers). Each position's hidden state is normalised and projected to 1,024 numbers, and
each of the six layers projects that again into a key and a value, which makes twelve rows of
1,024 numbers per position. The model reads left to right, so a position's hidden state depends
only on the tokens up to it, and so do its twelve rows, whatever follows.

### The store of prompt starts

The plugin therefore keeps those rows, not the hidden states, in its store (`gpu_store.py`). It
keeps them in blocks of 16 positions, each named by a SHA-256 chain over all token ids up to the
end of the block, the way vLLM names its own cache blocks, so a block can only be used for a prompt
that begins with exactly the same tokens. Each prompt start has one buffer. A prompt that extends
a kept start, as a growing game log does, appends to that buffer, rewriting up to 64 blocks at its
end, where the old state ended and Clef's fixed text after it began; a prompt that departs earlier
gets a buffer of its own. Over the budget, `CLEF_GPU_STORE_MB` (default 512), the least recently
used buffers are dropped whole. If vLLM skipped positions the store no longer has, after a restart
for example, the plugin writes `KEY.err` saying so, and the service asks again with a `cache_salt`
unique to the request, which makes vLLM read the whole prompt.

Each row is packed in NVFP4 (`packed.py`), the format of the model's weights: 4-bit codes with an
FP8 scale per 16 numbers and an fp32 scale per row, so a position's twelve rows take 6,960 bytes
instead of 49,152 in fp32, and 400 MB holds about 60,000 positions. On the hidden states of six real
prompts of 16,100 to 70,800 tokens, with 98% of each prompt taken from stored rows, the packing moved
probabilities by at most 0.0003 and changed none of 18 answers (`eval/row_precision.py`). The network
in bf16 moved probabilities by at most 0.0027 against fp32 on the 72 test records and changed no
answer.

The scoring code follows Cloudflare's `JointSchemaHead.forward` step for step. The service keeps a
CPU implementation of the same steps as a reference, whose probabilities match Cloudflare's to
1.2 × 10⁻⁷ on the 72 test records, whether the start is computed or taken from stored rows
(`service/test_reuse.py`); in fp32, given the same NVFP4 rows, the plugin's scorer agrees with it to
5 × 10⁻⁷ (`vllm-plugin/test_gpu_head.py`).

### How much vLLM reuses

For Qwen3.8-27B, vLLM's prefix cache works in blocks of 1,552 tokens, because the linear-attention
layers' state can only be saved at block boundaries. With a speculative draft model, vLLM also
holds back the last two matching blocks of every cache hit:

- vLLM drops the last matching block of every cache hit, because EAGLE-style draft models compute
  a block's draft cache from the token after it (`vllm/v1/core/kv_cache_coordinator.py`);
- the DFlash2 image's release patch makes the linear-attention lookup search one block shorter,
  because the last matched state may have been written over rejected draft tokens while
  generating (`MambaManager.find_longest_cache_hit`). Stock vLLM 0.31 has the first and not the
  second.

Neither applies to a decision: it generates one token and uses no draft tokens, and its prompt's
states are all written while reading prompts. With `CLEF_FULL_CACHE_HITS=1` in vLLM's environment,
the plugin changes three things for decision requests, and only for those:

- `KVCacheManager.get_computed_blocks`: for the length of the call, the coordinator gets copies of
  its cache groups with the drop flag off, so neither block is held back;
- the same call caps the reuse at the start of the first question, which the service passes as
  `clef_reuse_limit`, because the scoring network reads the hidden states of the question positions
  directly;
- `Scheduler._mamba_block_aligned_split`: with a draft model, the scheduler stops reading a prompt
  one block short of its last block boundary, so the linear-attention state there is never saved
  and the next, longer prompt cannot reuse it. For decisions it stops at the last boundary.

What remains is the block size itself: reuse grows one 1,552-token block at a time, so a decision
reads up to 1,551 tokens of the older prompt again, plus whatever is new.

The prefix cache is shared with chat, and the least recently used blocks go first. On the measured
server a game's start survived 48,000 tokens of other traffic between two decisions and was gone
after 60,000; the next decision then reads in full and caches the start again.

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

## Memory on a full card

On a card with no free memory, as on the measured RTX 5090, everything decisions need has to come
out of vLLM's key-value reservation, or Windows moves card memory to system RAM and chat slows to a
fraction of its speed. Decisions need the adapter's buffers, 1.4 GB plus vLLM's bookkeeping, and the
scoring network with its working memory, about 490 MiB; the store of prompt starts is in system RAM.
On the measured server the reservation went from 6.5 to 4.75 GiB (`--kv-cache-memory-bytes`), which
shortens the longest chat request from 262,144 to 181,000 tokens and lets six short requests run at
once instead of eight.

vLLM's step size matters on such a card as well. vLLM reads at most 2,048 tokens of one prompt per
step, but a step may hold up to `--max-num-batched-tokens` tokens. When two long prompts are read at
once with 4,096-token steps, PyTorch keeps the extra memory those steps needed, 600 MB after two
16,000-token prompts, and Windows moves part of vLLM's memory into system RAM until vLLM restarts.
With `--max-num-batched-tokens 2048`, long prompts arriving together are read one after another, and
several decisions about one new state share a single read of its start.
