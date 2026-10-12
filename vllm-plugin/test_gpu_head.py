"""Tests of the GPU scoring path (gpu_store.py, gpu_head.py), run on the CPU.

1. The store of prompt starts: kept rows found by their chain names, a growing prompt extended in
   place with its last blocks rewritten, a prompt departing earlier given its own segment, an
   unfinished segment released, the least recently used segment dropped first.
2. The scorer (in fp32 and in bf16, the plugin's default) against the decision service's CPU path on
   the 72 test records: the records' hidden
   states (saved by eval/reference.py, or made up) are fed to the scorer in chunks of 2,048 positions,
   as vLLM would, first from position 0 and then again with the start served from the store; the
   option probabilities are compared with the CPU path given the same NVFP4-rounded rows (they must
   agree to rounding) and with the CPU path at full precision (the change NVFP4 storage makes).

Run: python test_gpu_head.py --release DIR [--records ../eval/records.jsonl] [--reference-hidden DIR]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "service"))
from clef_capture import packed  # noqa: E402
from clef_capture.gpu_store import GpuStore  # noqa: E402

PLANES, WIDTH = 12, 64


def rows_for(ids):
    out = torch.zeros(PLANES, len(ids), WIDTH)
    running = 0.0
    for position, token in enumerate(ids):
        running = running * 0.5 + token
        out[:, position] = running + torch.arange(WIDTH) * 0.01
    return out


def stored(store, segment, upto):
    view = store.view(segment, upto)
    return torch.stack([torch.cat(list(view[p].slices(size=40))) for p in range(PLANES)])


def rounded(rows):
    return packed.unpack(*packed.pack(rows))


def test_store() -> None:
    per = PLANES * packed.bytes_per_row(WIDTH)
    store = GpuStore(1000 * per, torch.device("cpu"), PLANES, WIDTH)
    game = list(range(1000, 1100))
    seg, write_from = store.begin("a", game, 0, 96)
    assert write_from == 0
    store.write(seg, 0, rows_for(game)[:, :96])
    assert store.find(game, 48) is None, "findable before finish"
    store.finish(seg, 96)
    assert torch.equal(stored(store, store.find(game, 96), 96), rounded(rows_for(game)[:, :96]))

    # the log grows in front of fixed text: the kept start's last block departs, and is rewritten
    grown = game[:90] + list(range(2000, 2040))
    seg2, write_from = store.begin("b", grown, 80, 112)
    assert seg2 is seg and write_from == 80, (write_from,)
    store.write(seg2, 80, rows_for(grown)[:, 80:112])
    store.finish(seg2, 112)
    assert store.stats()["segments"] == 1
    assert torch.equal(stored(store, store.find(grown, 112), 112), rounded(rows_for(grown)[:, :112]))
    assert store.find(game, 96) is None and store.find(game, 80) is seg

    # a prompt departing after 32 tokens gets its own segment; the grown start stays usable
    store.rewrite_blocks = 2
    variant = game[:40] + [7] * 60
    seg3, write_from = store.begin("c", variant, 32, 96)
    assert seg3 is not seg and write_from == 32
    store.write(seg3, 32, rows_for(variant)[:, 32:96])
    store.finish(seg3, 96)
    assert torch.equal(stored(store, store.find(variant, 96), 96), rounded(rows_for(variant)[:, :96]))
    assert store.find(grown, 112) is seg

    # a start whose rows are not kept, an aborted segment, and the budget
    assert store.begin("d", list(range(9000, 9100)), 48, 96) is None
    seg4, _ = store.begin("e", list(range(9000, 9100)), 0, 96)
    store.abort(seg4)
    assert store.stats()["segments"] == 2
    store.find(grown, 112)
    store.segments.move_to_end(id(seg))   # the grown game is the most recently used
    other = list(range(5000, 5800))
    seg5, _ = store.begin("f", other, 0, 784)
    store.write(seg5, 0, rows_for(other)[:, :784])
    store.finish(seg5, 784)
    assert store.find(variant, 96) is None, "the least recently used segment should have gone"
    assert store.bytes <= store.budget or store.stats()["segments"] == 1
    print("GPU store: kept rows found by name, a growing prompt extended in place, a departing one given "
          "its own segment, unfinished segments released, least recently used dropped first")


def made_up_hidden(ids, size: int) -> torch.Tensor:
    """Made-up hidden states that, like a causal model's, depend only on the tokens up to each position:
    records sharing a start (all begin with Clef's fixed text) share its rows in the store, as real
    hidden states would allow."""
    import hashlib

    rows, running = [], b""
    for token in ids:
        running = hashlib.sha256(running + int(token).to_bytes(4, "little")).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(running[:8], "little"))
        rows.append(torch.randn(size, generator=generator) * 4)
    return torch.stack(rows).to(torch.bfloat16)


def test_streaming_attention() -> None:
    """Attention over parts given a slice at a time agrees with attention over whole tensors."""
    from types import SimpleNamespace

    from clef_capture import head

    torch.manual_seed(1)
    module = torch.nn.MultiheadAttention(64, 4, batch_first=True)
    query = torch.randn(9, 64)
    keys = [torch.randn(300, 64) * 3, torch.randn(77, 64)]
    values = [torch.randn(300, 64), torch.randn(77, 64)]

    def sliced(t, size):
        return SimpleNamespace(slices=lambda: (t[i:i + size] for i in range(0, t.shape[0], size)), shape=t.shape)

    with torch.inference_mode():
        whole = head._attend(module, query, keys, values)
        streamed = head._attend(module, query, [sliced(keys[0], 23), keys[1]], [sliced(values[0], 23), values[1]])
    difference = float((whole - streamed).abs().max())
    assert difference < 1e-5, difference
    print(f"attention streamed in slices of 23 positions agrees with the whole computation to {difference:.1e}")


def test_scorer(release: Path, records_path: Path, reference: Path | None, dtype: str) -> None:
    import clef_code

    clef_code.use_release(["--release", str(release)])
    import reuse
    from clef_capture.gpu_head import GpuScorer
    from joint_schema_model import JointSchemaHead, encode_record
    from lm_rows import OutputEmbeddingRows
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(release)
    head = JointSchemaHead(**json.loads((release / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(release / "joint_head.safetensors"))
    head = head.float().eval()
    lookup = OutputEmbeddingRows(release).as_tensor_lookup(torch.float32)
    size = json.loads((release / "joint_head_config.json").read_text())["hidden_size"]
    with tempfile.TemporaryDirectory() as directory:
        scorer = GpuScorer(str(release), 512 * 2**20, directory, store_device="cpu", dtype=dtype)
        worst_same, worst_full, flips, count, reused = 0.0, 0.0, 0, 0, 0
        for line in open(records_path):
            record = json.loads(line)
            encoded = encode_record(tokenizer, record)
            ids = list(encoded.input_ids)
            npz = reference / f"{record['id']}.npz" if reference else None
            if npz is not None and npz.exists():
                with np.load(npz) as data:
                    hidden = torch.from_numpy(data["hidden_bf16"].view(np.int16).copy()).view(torch.bfloat16)
            else:
                hidden = made_up_hidden(ids, size)
            needed = reuse.first_needed_position(encoded)
            cut = needed // 16 * 16
            spec = {"cut": cut, "questions": [{"type": int(q.question_type), "span": list(q.question_span),
                                                "options": [list(o) for o in q.option_spans]} for q in encoded.questions]}
            with torch.inference_mode():
                h = hidden.float()
                kv = reuse.memory_kv(head, h)
                full = [l.softmax(-1) for l in reuse.forward(head, h, 0, [kv], torch.tensor(ids), encoded, lookup)]
                same = [l.softmax(-1) for l in reuse.forward(head, h, 0, [rounded(kv[:, :cut]), kv[:, cut:]],
                                                             torch.tensor(ids), encoded, lookup)]
            for attempt, first in (("read", 0), ("reused", cut // 1552 * 1552 if cut >= 1552 else cut // 16 * 16)):
                key = f"key-{record['id'][:20]}-{attempt}".replace("&", "_")
                for start in range(first, len(ids), 2048):
                    scorer.add(f"{record['id']}-{attempt}", key, ids, spec, start, hidden[start:start + 2048])
                scorer.drop([f"{record['id']}-{attempt}"])
                result = json.loads((Path(directory) / f"{key}.json").read_text())
                got = [torch.tensor(l).softmax(-1) for l in result["logits"]]
                for a, b, c in zip(got, same, full):
                    worst_same = max(worst_same, float((a - b).abs().max()))
                    worst_full = max(worst_full, float((a - c).abs().max()))
                    flips += int(a.argmax() != c.argmax())
                reused += result["first_position"] > 0
            count += 1
    print(f"GPU scorer in {dtype} on {count} records, read in full and with the start from its store "
          f"({reused} reused): largest difference from the CPU path with the same NVFP4 rows {worst_same:.2e}, "
          f"from the CPU path at full precision {worst_full:.4f}; answers changed {flips}")
    # fp32 must match the CPU path to rounding; bf16 moved probabilities by up to 0.0027 in Cloudflare's own run
    assert count == 72 and flips == 0 and worst_same < (1e-4 if dtype == "float32" else 0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--records", default=str(HERE.parent / "eval" / "records.jsonl"))
    parser.add_argument("--reference-hidden")
    args = parser.parse_args()
    torch.set_num_threads(8)
    test_store()
    test_streaming_attention()
    for dtype in ("float32", "bfloat16"):
        test_scorer(Path(args.release), Path(args.records), Path(args.reference_hidden) if args.reference_hidden else None,
                    dtype)
    print("all passed")
