"""CPU tests for reusing the start of a decision prompt (reuse.py), without a GPU.

1. The scoring network, run by reuse.forward, against Cloudflare's JointSchemaHead on the 72 test
   records: once with every position computed, and once with the start (every whole 16-position
   block before the first question) supplied as stored key and value rows. The probabilities must
   agree to rounding. The hidden states are the ones reference.py saved, if --reference-hidden names
   a directory of them, and otherwise made-up ones of the right shape: the two computations must
   agree on any input, so the test needs neither a GPU nor Clef's backbone.
2. PrefixStore: blocks are shared between prompts with the same start, never used for a different
   start, and the oldest go first when the store is over budget.

Run: python test_reuse.py --release DIR [--records ../eval/records.jsonl] [--reference-hidden DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clef_code  # noqa: E402

clef_code.use_release()
import reuse  # noqa: E402
from joint_schema_model import JointSchemaHead, collate_records, encode_record  # noqa: E402
from lm_rows import OutputEmbeddingRows  # noqa: E402


def load_head(release: Path):
    from safetensors.torch import load_file

    head = JointSchemaHead(**json.loads((release / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(release / "joint_head.safetensors"), strict=True)
    return head.float().eval()


def hidden_states(record: dict, input_ids: list[int], reference: Path | None, size: int) -> tuple[torch.Tensor, str]:
    """Saved hidden states for the record if there are any, otherwise made-up ones."""
    npz = reference / f"{record['id']}.npz" if reference else None
    if npz is not None and npz.exists():
        with np.load(npz) as data:
            assert list(data["input_ids"]) == input_ids, f"{record['id']}: token ids differ from the reference run's"
            return torch.from_numpy(data["hidden_bf16"].view(np.int16).copy()).view(torch.bfloat16).float(), "saved"
    generator = torch.Generator().manual_seed(sum(map(ord, record["id"])))
    return torch.randn(len(input_ids), size, generator=generator).to(torch.bfloat16).float() * 4, "made-up"


def test_forward(release: Path, reference: Path | None, records_path: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(release)
    head = load_head(release)
    rows = OutputEmbeddingRows(release).as_tensor_lookup(torch.float32)
    size = json.loads((release / "joint_head_config.json").read_text())["hidden_size"]
    worst_full = worst_split = 0.0
    count, kinds = 0, set()
    with torch.inference_mode():
        for line in open(records_path):
            record = json.loads(line)
            encoded = encode_record(tokenizer, record)
            hidden, kind = hidden_states(record, list(encoded.input_ids), reference, size)
            kinds.add(kind)
            batch = collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
            original = head(hidden[None], batch["input_ids"], batch["attention_mask"], [encoded], rows)[0]
            ids = torch.tensor(encoded.input_ids)
            kv = reuse.memory_kv(head, hidden)
            full = reuse.forward(head, hidden, 0, kv, ids, encoded, rows)
            first = reuse.first_needed_position(encoded) // 16 * 16
            parts = [reuse.memory_kv(head, hidden[:first]), reuse.memory_kv(head, hidden[first:])]
            split = reuse.forward(head, hidden[first:], first, parts, ids, encoded, rows)
            for a, b, c in zip(original, full, split):
                pa, pb, pc = a.softmax(-1), b.softmax(-1), c.softmax(-1)
                worst_full = max(worst_full, float((pa - pb).abs().max()))
                worst_split = max(worst_split, float((pa - pc).abs().max()))
            count += 1
    print(f"scoring network on {count} records ({' and '.join(sorted(kinds))} hidden states): "
          f"largest probability difference from Cloudflare's "
          f"{worst_full:.2e} with every position computed, {worst_split:.2e} with the start from stored rows")
    assert count == 72 and worst_full < 1e-4 and worst_split < 1e-4


def test_store() -> None:
    width, block = 4, 16
    per_position = 12 * width * 4

    def rows_for(ids):
        """Rows that depend on the token ids up to each position, as a causal model's would."""
        out = torch.zeros(12, len(ids), width)
        running = 0.0
        for position, token in enumerate(ids):
            running = running * 0.5 + token
            out[:, position] = running
        return out

    store = reuse.PrefixStore(budget_bytes=300 * per_position, block=block, rewrite_blocks=2)
    game = list(range(1000, 1080))                      # a start of 80 tokens, then a question
    rows = rows_for(game)
    store.put(game, 0, None, rows, 70)                  # keeps whole blocks below 70: positions 0..63
    assert store.stats()["positions"] == 64
    view = store.get(game, 48)
    assert torch.equal(view, rows[:, :48])
    assert store.get(game, 40) is None, "a position count that is not a whole block"
    assert store.get(game, 80) is None, "positions that were never kept"

    # The game's log grows: the next prompt extends the start. vLLM skipped 48 positions, so the
    # service has the kept rows of 0..48 and new rows from 48 on; the segment grows in place.
    grown = game + list(range(2000, 2040))
    grown_rows = rows_for(grown)
    store.put(grown, 48, store.get(grown, 48), grown_rows[:, 48:], 110)  # needs a larger buffer
    assert store.stats()["segments"] == 1
    again = store.get(grown, 96)
    assert torch.equal(again, grown_rows[:, :96]), "rows after growing"
    storage = again.untyped_storage().data_ptr()
    assert store.get(grown, 64).untyped_storage().data_ptr() == storage, "kept rows copied rather than viewed"
    # One more turn fits in the spare room: appended in place, nothing reallocated.
    longer = grown + list(range(3000, 3016))
    longer_rows = rows_for(longer)
    store.put(longer, 96, store.get(longer, 96), longer_rows[:, 96:], 120)
    assert torch.equal(store.get(longer, 112), longer_rows[:, :112])
    assert store.get(longer, 112).untyped_storage().data_ptr() == storage, "grew by copying although it fitted"
    grown, grown_rows = longer, longer_rows

    # A game log grows in front of fixed text: the kept start ends with a few tokens of that text,
    # which the next prompt has further on. The departing last block is rewritten in place.
    fixed = [9, 9, 9, 9, 9, 9, 9, 9]
    log_a = list(range(4000, 4100)) + fixed                      # kept: whole blocks of 0..104 -> 0..96
    store_b = reuse.PrefixStore(budget_bytes=1000 * per_position, block=block)
    store_b.put(log_a, 0, None, rows_for(log_a), 104)
    log_b = list(range(4000, 4100)) + list(range(6000, 6020)) + fixed
    rows_b = rows_for(log_b)
    kept = store_b.get(log_b, 96)
    assert kept is not None and torch.equal(kept, rows_b[:, :96])
    storage_b = kept.untyped_storage().data_ptr()
    store_b.put(log_b, 96, kept, rows_b[:, 96:], 124)
    assert store_b.stats()["segments"] == 1, "a growing log should extend its segment, not copy it"
    assert torch.equal(store_b.get(log_b, 112), rows_b[:, :112])
    assert store_b.get(log_b, 112).untyped_storage().data_ptr() == storage_b
    assert store_b.get(log_a, 96) is not None and store_b.get(log_a, 112) is None

    # A prompt that shares only the first 40 tokens departs from the kept start: it gets its own
    # segment, and the original start stays usable.
    variant = game[:40] + [7] * 60
    variant_rows = rows_for(variant)
    store.put(variant, 32, store.get(variant, 32), variant_rows[:, 32:], 90)
    assert store.stats()["segments"] == 2
    assert torch.equal(store.get(variant, 80), variant_rows[:, :80])
    assert torch.equal(store.get(grown, 96), grown_rows[:, :96]), "original start lost"
    assert store.get(variant, 96) is None, "a block used for a different start"

    # Over the budget the least recently used segment goes, whole.
    store.get(grown, 96)                                # the grown game is now the most recently used
    other = list(range(5000, 5100))
    store.put(other, 0, None, rows_for(other), 100)
    assert store.get(variant, 80) is None, "the least recently used segment should have been dropped"
    assert torch.equal(store.get(grown, 96), grown_rows[:, :96])
    assert store.bytes <= store.budget
    print("prefix store: kept rows read as views and grown in place; a departing prompt gets its own "
          "segment; least recently used segments dropped first")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="the Clef release directory (or CLEF_RELEASE)")
    parser.add_argument("--records", default=str(HERE.parent / "eval" / "records.jsonl"),
                        help="the 72 test records, written by eval/build_records.py")
    parser.add_argument("--reference-hidden", help="hidden states saved by eval/reference.py (*.npz)")
    args = parser.parse_args()
    torch.set_num_threads(8)
    test_store()
    test_forward(clef_code.release_dir(), Path(args.reference_hidden) if args.reference_hidden else None,
                 Path(args.records))
    print("all passed")
