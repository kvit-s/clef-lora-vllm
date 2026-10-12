"""Reusing the already-read start of a decision prompt.

Clef's scoring network (JointSchemaHead in joint_schema_model.py) uses the prompt positions before
the first question in one way only: as the memory its six attention layers read (the two routing
layers, and the cross-attention of the four decoder layers). Each position's hidden state is
normalised and projected to 1,024 numbers, and every one of those six layers projects that row again
into a key and a value. The backbone is causal, so these 12 rows of 1,024 numbers depend only on the
tokens up to that position, and they are the same whatever follows.

When vLLM serves the start of a prompt from its prefix cache it skips those positions, so the capture
file only holds the positions from the first one vLLM computed. This module:

- computes the 12 key and value rows for positions (memory_kv), and runs the scoring network from the
  hidden states of the computed positions plus the stored rows of the skipped ones (forward); both
  live in the plugin package (vllm-plugin/clef_capture/head.py), shared with the plugin's GPU path,
  and are imported here. test_reuse.py checks them against the original on the 72 test records;
- keeps the key and value rows of the starts of recently read prompts (PrefixStore), one buffer per
  start. A prompt that extends a kept start, as a game's growing log does, appends its new rows to
  that buffer, and reading the kept rows returns a view of it, so nothing is copied per request.

joint_schema_model.py is Cloudflare's, loaded unchanged from the downloaded release (clef_code.py).
"""

from __future__ import annotations

import hashlib
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vllm-plugin"))
from clef_capture.head import attention_modules, first_needed_position, forward, memory_kv  # noqa: E402,F401


class _Segment:
    """The key and value rows of one prompt start: rows[:, :length] are valid, and names[i] is the
    chain name of the block at positions i*block..(i+1)*block."""

    def __init__(self, rows: torch.Tensor, length: int, names: list[bytes]) -> None:
        self.rows, self.length, self.names = rows, length, names


class PrefixStore:
    """Key and value rows of the starts of recently read prompts.

    Each start is kept as one buffer of shape [12, capacity, width] (a segment). Its blocks of
    `block` positions are named by the SHA-256 chain of all token ids up to their end, as vLLM names
    its cache blocks, so a name can only ever match a prompt that starts with exactly the same tokens,
    and finding a prompt's last needed block in the index finds a segment that holds all of its start.

    get() returns a view of that segment, so reading kept rows copies nothing. put() appends a
    prompt's new rows to the segment its start came from when the prompt extends it. A growing game
    log departs from the previous prompt's kept start in its last block or two, where the old state
    ended and Clef's fixed text after the state began; put() drops up to `rewrite_blocks` such
    blocks from the end of the segment and appends in their place. A prompt that departs earlier gets
    a new segment, which copies the shared part once. Capacity grows by a quarter at a time. Over the
    budget, the least recently used segments are dropped whole.

    The default allowance, 64 blocks of 16 positions, is below one vLLM cache block of this model
    (1,552 tokens). When vLLM holds back the last cached block of a prompt (it holds back two on the
    measured server), the older prompt whose last blocks are rewritten can still be served from the
    rest of its segment if it is sent again. With CLEF_FULL_CACHE_HITS=1 nothing is held back, and
    such an older prompt, sent again, may need a second read.

    Because put() may overwrite the last blocks of a segment, a caller must not use rows from get()
    while another thread calls put(); the decision service holds one lock from get() to put().
    """

    def __init__(self, budget_bytes: int, block: int = 16, rewrite_blocks: int = 64) -> None:
        self.budget, self.block, self.rewrite_blocks = budget_bytes, block, rewrite_blocks
        self.segments: OrderedDict[int, _Segment] = OrderedDict()  # id(segment) -> segment, oldest first
        self.index: dict[bytes, tuple[_Segment, int]] = {}         # block name -> (segment, block number)
        self.bytes = 0
        self.lock = threading.Lock()

    def _names(self, token_ids, count: int) -> list[bytes]:
        ids = np.asarray(token_ids[:count * self.block], dtype="<i4")
        names, name = [], b""
        for index in range(count):
            name = hashlib.sha256(name + ids[index * self.block:(index + 1) * self.block].tobytes()).digest()
            names.append(name)
        return names

    def get(self, token_ids, positions: int) -> torch.Tensor | None:
        """A view of the rows of positions 0..positions, shape [12, positions, width], or None if they
        are not all kept."""
        if positions <= 0 or positions % self.block:
            return None
        count = positions // self.block
        last = self._names(token_ids, count)[-1]
        with self.lock:
            entry = self.index.get(last)
            if entry is None or entry[1] != count - 1:
                return None
            segment = entry[0]
            self.segments.move_to_end(id(segment))
            return segment.rows[:, :positions]

    def put(self, token_ids, first: int, stored: torch.Tensor | None, new_rows: torch.Tensor, positions: int) -> None:
        """Keep the rows of the whole blocks within positions 0..positions.

        stored: the rows of positions 0..first that get() returned (None when first is 0);
        new_rows: the rows of positions first onwards, shape [12, n, width].
        """
        count = min(positions, first + new_rows.shape[1]) // self.block
        if count == 0:
            return
        names = self._names(token_ids, count)
        width = new_rows.shape[2]
        with self.lock:
            # The longest kept start of this prompt: names form a chain, so if block i is kept, so are
            # all blocks before it.
            low, high = 0, count
            while low < high:
                middle = (low + high + 1) // 2
                if names[middle - 1] in self.index:
                    low = middle
                else:
                    high = middle - 1
            matched = low
            segment = self.index[names[matched - 1]][0] if matched else None
            if matched == count:
                self.segments.move_to_end(id(segment))
                return
            end = count * self.block
            if segment is not None and matched * self.block >= first \
                    and 0 < len(segment.names) - matched <= self.rewrite_blocks:
                # The prompt departs from this start only in its last few blocks: drop them.
                for name in segment.names[matched:]:
                    if self.index.get(name, (None,))[0] is segment:
                        del self.index[name]
                segment.names = segment.names[:matched]
                segment.length = matched * self.block
            if segment is not None and len(segment.names) == matched and matched * self.block >= first:
                # The prompt extends the kept start: append the new blocks to its segment.
                start = matched * self.block
                self._reserve(segment, end, width)
                segment.rows[:, start:end] = new_rows[:, start - first:end - first]
            else:
                # Nothing kept, or the prompt departs from a kept start partway: a new segment.
                rows = self._allocate(end, width)
                if first:
                    rows[:, :first] = stored
                rows[:, first:end] = new_rows[:, :end - first]
                segment = _Segment(rows, 0, [])
                self.segments[id(segment)] = segment
                self.bytes += rows.numel() * rows.element_size()
                matched = 0
            for number in range(matched, count):
                self.index[names[number]] = (segment, number)
            segment.names = segment.names[:matched] + names[matched:count]
            segment.length = end
            self.segments.move_to_end(id(segment))
            self._evict(keep=segment)

    def _allocate(self, positions: int, width: int) -> torch.Tensor:
        """A buffer for `positions` rows plus a quarter more for growth, as far as the budget allows."""
        limit = self.budget // (12 * width * 4)
        room = max(positions, min(int(positions * 1.25), limit))
        capacity = -(-room // self.block) * self.block
        return torch.empty(12, capacity, width, dtype=torch.float32)

    def _reserve(self, segment: _Segment, positions: int, width: int) -> None:
        if positions <= segment.rows.shape[1]:
            return
        rows = self._allocate(positions, width)
        rows[:, :segment.length] = segment.rows[:, :segment.length]
        self.bytes += (rows.numel() - segment.rows.numel()) * rows.element_size()
        segment.rows = rows

    def _evict(self, keep: _Segment) -> None:
        while self.bytes > self.budget and len(self.segments) > 1:
            key, segment = next(iter(self.segments.items()))
            if segment is keep:
                self.segments.move_to_end(key)
                key, segment = next(iter(self.segments.items()))
            del self.segments[key]
            self.bytes -= segment.rows.numel() * segment.rows.element_size()
            for number, name in enumerate(segment.names):
                if self.index.get(name, (None,))[0] is segment:
                    del self.index[name]

    def stats(self) -> dict:
        with self.lock:
            return {"segments": len(self.segments), "positions": sum(s.length for s in self.segments.values()),
                    "megabytes": round(self.bytes / 2**20, 1), "budget_megabytes": round(self.budget / 2**20)}
