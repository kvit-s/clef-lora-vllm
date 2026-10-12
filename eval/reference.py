"""Run Cloudflare's Clef at full bf16 precision on the CPU and record what it computes.

This is the yardstick for the GPU path. For each test record it saves the prompt token ids, the
backbone's final hidden states (the numbers the scoring head reads), and the head's per-option
logits, so the vLLM path can later be compared both on the hidden states and on the answers.

It needs the whole release (52 GB), e.g. `hf download Cloudflare/clef --revision 2f3de3dd... --local-dir
DIR`. The backbone does not have to fit in memory: layers from --resident-layers onwards are read
from the downloaded safetensors files on every pass (transformers maps them in place). On an
8-core desktop CPU with 31 GB of memory this ran at about 13 tokens per second, 65 minutes for the
72 records; the answers it produced are in reference/, so it does not need to be run again.
Weights stay bf16; every linear layer is computed in fp32 and rounded back to bf16, which is what a
GPU's bf16 matrix multiply does and is about three times faster than bf16 arithmetic on this CPU.

usage: reference.py --release DIR RECORDS_JSONL OUT_DIR [--resident-layers N] [--batch-tokens N] [--only GROUP,...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
import clef_code  # noqa: E402

if __name__ == "__main__":
    clef_code.use_release()
from joint_schema_model import JointSchemaHead, collate_records, encode_record  # noqa: E402
from lm_rows import OutputEmbeddingRows  # noqa: E402

_linear = F.linear


def linear_fp32(x, weight, bias=None):
    if x.device.type == "cpu" and x.dtype == torch.bfloat16:
        out = _linear(x.float(), weight.float(), None if bias is None else bias.float())
        return out.to(torch.bfloat16)
    return _linear(x, weight, bias)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="the full Clef release directory (or CLEF_RELEASE)")
    parser.add_argument("records")
    parser.add_argument("out")
    parser.add_argument("--resident-layers", type=int, default=20)
    parser.add_argument("--batch-tokens", type=int, default=2400)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    F.linear = linear_fp32
    torch.set_num_threads(os.cpu_count() or 8)
    release, out = clef_code.release_dir(), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    config = json.loads((release / "config.json").read_text())
    layers = config["text_config"]["num_hidden_layers"]
    device_map = {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": "cpu",
        "model.language_model.norm": "cpu",
        "model.language_model.rotary_emb": "cpu",
        "lm_head": "disk",
    }
    for i in range(layers):
        device_map[f"model.language_model.layers.{i}"] = "cpu" if i < args.resident_layers else "disk"

    started = time.time()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        release, dtype=torch.bfloat16, device_map=device_map, offload_folder=str(out / "offload"),
    )
    backbone.eval()
    processor = AutoProcessor.from_pretrained(release)
    tokenizer = processor.tokenizer
    from safetensors.torch import load_file

    head_config = json.loads((release / "joint_head_config.json").read_text())
    head_state = load_file(release / "joint_head.safetensors")
    head_bf16 = JointSchemaHead(**head_config)
    head_bf16.load_state_dict(head_state, strict=True)
    head_bf16 = head_bf16.to(torch.bfloat16).eval()
    head_fp32 = JointSchemaHead(**head_config)
    head_fp32.load_state_dict(head_state, strict=True)
    head_fp32 = head_fp32.float().eval()
    rows = OutputEmbeddingRows(release)
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    only = set(filter(None, args.only.split(",")))
    pending = []
    for line in open(args.records):
        record = json.loads(line)
        if only and record["group"] not in only:
            continue
        if (out / f"{record['id']}.npz").exists():
            continue
        encoded = encode_record(tokenizer, record)
        pending.append((record, encoded))
    # similar lengths together, so a batch wastes little on padding
    pending.sort(key=lambda item: len(item[1].input_ids))
    print(f"{len(pending)} records to run", flush=True)

    text_model = backbone.model.language_model
    while pending:
        batch_items = [pending.pop(0)]
        while pending and (len(batch_items) + 1) * len(pending[0][1].input_ids) <= args.batch_tokens:
            batch_items.append(pending.pop(0))
        encoded = [item[1] for item in batch_items]
        batch = collate_records(encoded, tokenizer.pad_token_id, torch.device("cpu"))
        t0 = time.time()
        with torch.inference_mode():
            hidden = text_model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                use_cache=False, return_dict=True,
            ).last_hidden_state
            t1 = time.time()
            logits_bf16 = head_bf16(hidden, batch["input_ids"], batch["attention_mask"], encoded,
                                    rows.as_tensor_lookup(torch.bfloat16))
            logits_fp32 = head_fp32(hidden.float(), batch["input_ids"], batch["attention_mask"], encoded,
                                    rows.as_tensor_lookup(torch.float32))
        t2 = time.time()
        for b, (record, enc) in enumerate(batch_items):
            n = len(enc.input_ids)
            result = {
                "id": record["id"], "group": record["group"], "expected": record.get("expected", {}),
                "questions": [], "backbone_seconds_batch": t1 - t0, "head_seconds_batch": t2 - t1,
                "batch_size": len(batch_items),
            }
            for q, lb, lf in zip(enc.questions, logits_bf16[b], logits_fp32[b]):
                result["questions"].append({
                    "id": q.question_id, "options": list(q.option_ids),
                    "logits_bf16": lb.float().tolist(), "logits_fp32": lf.float().tolist(),
                    "probabilities_bf16": lb.float().softmax(-1).tolist(),
                    "probabilities_fp32": lf.float().softmax(-1).tolist(),
                })
            bits = hidden[b, :n].contiguous().view(torch.int16).numpy().view(np.uint16)
            np.savez(out / f"{record['id']}.npz", input_ids=np.array(enc.input_ids, dtype=np.int32), hidden_bf16=bits)
            (out / f"{record['id']}.json").write_text(json.dumps(result))
        print(f"batch of {len(batch_items)} ({max(len(e.input_ids) for e in encoded)} tokens max): "
              f"backbone {t1 - t0:.1f}s head {t2 - t1:.2f}s; {len(pending)} left", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
