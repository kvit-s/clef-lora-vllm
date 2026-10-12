"""How storing the kept key and value rows at lower precision changes Clef's answers, on real long prompts.

The decision service keeps, for the start of each prompt, the twelve key and value rows per position
that Clef's scoring network attends to (service/reuse.py), in fp32. This script measures what storing
them in bf16, FP8 or NVFP4 would change. For each prompt it asks vLLM (with the Clef adapter and the
clef_capture plugin) for the real hidden states, reading the whole prompt, computes the rows in fp32,
rounds the rows of the positions before the first question to each format and back, and compares the
scoring network's probabilities with the fp32 ones. The positions from the first question on are never
kept, so they stay fp32, as they would.

Prompts: a support transcript and a game log, each made long by repetition of varied lines, at the
lengths given; each asked a 6-option choice, a yes/no question and a 4-level score.

usage: row_precision.py VLLM_URL CAPTURE_DIR --release DIR [--sizes 16000,32000,64000] [--adapter clef]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "service"))
sys.path.insert(0, str(HERE))
import clef_code  # noqa: E402

clef_code.use_release()
import reuse  # noqa: E402
from joint_schema_model import JointSchemaHead, encode_record  # noqa: E402
from lm_rows import OutputEmbeddingRows  # noqa: E402
from service import VllmBackbone  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def nvfp4(x: torch.Tensor) -> torch.Tensor:
    """Round to NVFP4 and back: e2m1 values in blocks of 16 along the last dimension, an FP8 (e4m3)
    scale per block, and an fp32 scale per row. Done in slices to bound memory."""
    out = torch.empty_like(x)
    flat_in, flat_out = x.reshape(-1, x.shape[-1]), out.view(-1, x.shape[-1])
    for start in range(0, flat_in.shape[0], 8192):
        rows = flat_in[start:start + 8192]
        blocks = rows.reshape(rows.shape[0], -1, 16)
        row_scale = rows.abs().amax(-1).clamp(min=1e-12)[:, None, None] / (448 * 6)
        block_scale = (blocks.abs().amax(-1, keepdim=True) / 6 / row_scale).to(torch.float8_e4m3fn).float() * row_scale
        block_scale = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale)
        q = (blocks / block_scale).clamp(-6, 6)
        values = E2M1[torch.bucketize(q.abs(), E2M1_EDGES)] * q.sign()
        flat_out[start:start + 8192] = (values * block_scale).reshape(rows.shape)
    return out


def fp8(x: torch.Tensor) -> torch.Tensor:
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448
    return (x / s).to(torch.float8_e4m3fn).float() * s


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


QUESTIONS = {
    "transcript": {
        "intent": {"type": "choice", "instructions": "What is the primary thing the customer wants in the transcript?",
                   "criteria": {"refund": "money back or a charge reversed", "technical_help": "help fixing a bug or outage",
                                "billing_question": "a question about an invoice or a charge",
                                "information": "how a feature works", "cancellation": "wants to cancel the account",
                                "other": "none of the above"}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Right now"]},
    },
    "game": {
        "next_step": {"type": "choice", "instructions": "Which step should the explorer take next?",
                      "criteria": {"north": "Move north", "south": "Move south", "east": "Move east", "west": "Move west",
                                   "rest": "Rest and recover energy", "dig": "Dig for treasure"}},
        "low_energy": {"type": "noul", "instructions": "Is the explorer's energy dangerously low?"},
        "progress": {"type": "score", "criteria": ["Lost", "Wandering", "Getting closer", "Almost there"]},
    },
}


def state_text(kind: str, tokens: int, tokenizer, seed: int) -> str:
    rng = random.Random(seed)
    if kind == "transcript":
        import latency_sweep
        lines = latency_sweep.build_lines(tokenizer, tokens, seed)
        return "Support transcript, case 1\n" + "\n".join(lines)
    import game_test
    rules = game_test.made_up_rules(count=4000, seed=seed)
    log = game_test.game_log(400, seed=seed)
    text, parts = "GAME RULES AND NOTES\n\n", rules.split("\n")
    i = 0
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < tokens - 3000:
        text += "\n".join(parts[i:i + 200]) + "\n"
        i += 200
    return text + "\nGAME LOG\n" + "\n".join(log[:rng.randint(150, 400)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm")
    parser.add_argument("capture_dir")
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    parser.add_argument("--adapter", default="clef")
    parser.add_argument("--sizes", default="16000,32000,64000")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    release = clef_code.release_dir()
    tokenizer = AutoTokenizer.from_pretrained(release)
    head = JointSchemaHead(**json.loads((release / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(release / "joint_head.safetensors"))
    head = head.float().eval()
    rows = OutputEmbeddingRows(release).as_tensor_lookup(torch.float32)
    backbone = VllmBackbone(args.vllm, args.adapter, args.capture_dir, 900)
    formats = {"bf16": bf16, "FP8": fp8, "NVFP4": nvfp4}
    totals = {name: {"diff": 0.0, "flips": 0, "answers": 0} for name in formats}
    print(f"{'prompt':<22} {'tokens':>7} {'kept':>7} | " + " | ".join(f"{n:>5} max diff, flips" for n in formats))
    for size in [int(s) for s in args.sizes.split(",")]:
        for kind in ("transcript", "game"):
            record = {"model": "clef", "state": state_text(kind, size, tokenizer, seed=size), "questions": QUESTIONS[kind]}
            encoded = encode_record(tokenizer, record, max_length=1 << 40)
            ids = list(encoded.input_ids)
            cut = reuse.first_needed_position(encoded) // 16 * 16
            started = time.time()
            _, hidden, _ = backbone.hidden_states(ids, reuse_cache=False)
            with torch.inference_mode():
                hidden = hidden.float()
                kv = reuse.memory_kv(head, hidden)
                id_tensor = torch.tensor(ids)
                base = [l.softmax(-1) for l in reuse.forward(head, hidden, 0, [kv], id_tensor, encoded, rows)]
                cells = []
                for name, rounding in formats.items():
                    kept = rounding(kv[:, :cut])
                    out = [l.softmax(-1) for l in reuse.forward(head, hidden, 0, [kept, kv[:, cut:]], id_tensor, encoded, rows)]
                    diff = max(float((a - b).abs().max()) for a, b in zip(base, out))
                    flips = sum(int(a.argmax() != b.argmax()) for a, b in zip(base, out))
                    totals[name]["diff"] = max(totals[name]["diff"], diff)
                    totals[name]["flips"] += flips
                    totals[name]["answers"] += len(base)
                    cells.append(f"{diff:>14.4f}, {flips}")
                del kv, kept
            print(f"{kind + ' ' + str(size):<22} {len(ids):>7} {cut:>7} | " + " | ".join(cells)
                  + f"   ({time.time() - started:.0f} s)", flush=True)
    for name, t in totals.items():
        print(f"{name}: largest probability change {t['diff']:.4f}, answers changed {t['flips']} of {t['answers']}")


if __name__ == "__main__":
    main()
