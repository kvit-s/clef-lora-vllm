---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.8-27B
tags:
- clef
- lora
- peft
- vllm
- systemone
- decision-model
- qwen3.8
---

# Clef as a rank-256 LoRA adapter for Qwen3.8-27B

This adapter turns Qwen/Qwen3.8-27B into the backbone of Cloudflare's
[Clef](https://huggingface.co/Cloudflare/clef) decision model. Clef's weights differ from the base
model's only in the large matrices of decoder layers 40–63, by a change of rank 256, which is what
a LoRA fine-tune leaves when merged into the weights. This adapter is that change, factored back
into pairs of rank-256 matrices, in the PEFT format vLLM and PEFT load.

With it, a vLLM server that already runs Qwen3.8-27B for chat can also answer Clef's decision
requests, applying the adapter only to those requests. Clef's backbone alone does not answer
questions: its hidden states go to Clef's scoring network, which is in Cloudflare's release. The
code that connects the two, a vLLM plugin and a decision service answering `POST /v1/systemone`,
is at https://github.com/kvit-s/clef-lora-vllm.

## Files

| File | What it is |
|---|---|
| `adapter_model.safetensors` | 186 pairs of LoRA matrices, rank 256, bf16; 700,366,848 parameters, 1.40 GB |
| `adapter_config.json` | PEFT configuration: r = lora_alpha = 256, layers 40–63, twelve module types (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`) |
| `extraction-report.json` | per-matrix figures from the extraction (below) |
| `reference/answers/` | Clef's answers at full precision (bf16, on the CPU) to the 72 test records of the GitHub repository |
| `reference/hidden/` | the hidden states of that run, to compare a server's output against (`eval/compare_hidden.py`) |

## How it was made

From `Cloudflare/clef` at revision `2f3de3dd85f379784083b0814d997ab627200f0c` and
`Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`:

1. Every one of Clef's 1,184 tensors was compared with the base model's by hashing its first
   256 KiB. The vision tower, layers 0–39, the embeddings, the output layer and all normalisation
   weights are identical; 186 tensors differ, all of them attention and feed-forward matrices of
   layers 40–63.
2. For each, the difference D = W_clef − W_base was factored as D ≈ B·A from its top 256 singular
   vectors and values (A = √S·Vᵀ, B = U·√S) and rounded to bf16.

`extract/extract_adapter.py` in the GitHub repository repeats this with one command.

## Fidelity

Clef's weights are stored in bf16, so each is already off by up to half a step of 8-bit
precision. For the 150 matrices larger than 256 in both dimensions, what the factorisation leaves
out measures 0.85 to 1.005 times that rounding noise, and for the smaller matrices a fraction of
it. Weights rebuilt as base + B·A differ from Clef's by 0.17% at most.

Applied to a quantized base model, the adapter reproduces Clef less closely. On one RTX 5090
setup (a 4-bit NVFP4 checkpoint with a 4-bit cache, in vLLM) it answered 83 to 85 of 85 test
questions with known answers correctly across runs, where Clef at full precision answered 83, and
the median decision took 165 ms. The GitHub repository has the details and the scripts to measure other setups.

## Use with vLLM

```bash
export CLEF_LORA_LAYERS=40-63                  # read by the clef_capture plugin
vllm serve Qwen/Qwen3.8-27B ... \
    --enable-lora --max-lora-rank 256 --max-loras 1 \
    --lora-modules clef=kvit-s/clef-lora-r256
```

Requests to the model name `clef` then run Clef's backbone; the plugin and the decision service
from the GitHub repository turn them into decisions.

## License

Apache-2.0. The adapter is derived from the weights of Cloudflare's Clef and of Qwen3.8-27B, both
released under Apache-2.0, as the difference between the two in decoder layers 40–63, factored
into rank-256 matrices. No other change was made. The reference answers were computed with
Cloudflare's code on rows of ARC-Challenge (allenai/ai2_arc, CC BY-SA 4.0) and BANKING77 (PolyAI,
CC BY 4.0). The answer files hold record ids, option names, logits and probabilities. The
hidden-state files also hold each prompt's token ids, which decode back to the prompt text, so for
those records they contain text from the two datasets, under the datasets' licences.
