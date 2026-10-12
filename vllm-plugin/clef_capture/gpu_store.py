"""Key and value rows of recently read prompt starts, kept on the GPU in NVFP4 (packed.py).

The same rules as the decision service's CPU store (service/reuse.py PrefixStore): one buffer
(segment) per prompt start, with blocks of 16 positions named by the SHA-256 chain of all token ids up
to their end, so a name can only match a prompt that starts with exactly the same tokens; a prompt
that extends a kept start appends to its segment, rewriting up to `rewrite_blocks` blocks at its end
(where a growing log departs from the previous prompt); a prompt that departs earlier gets its own
segment, copying the shared part once; over the budget the least recently used segments go whole.

The difference is when rows arrive: a decision's rows are written chunk by chunk while vLLM reads the
prompt. begin() reserves the segment and says from which position rows must be written, write() packs
and stores a range, finish() makes the new blocks findable, abort() releases a segment that was never
finished. Everything runs in vLLM's engine thread, one call at a time; a segment being written by one
request is not extended by another, which gets a segment of its own.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import torch

from . import packed


class Segment:
    def __init__(self, capacity: int, planes: int, width: int, device: torch.device) -> None:
        self.codes = torch.empty(planes, capacity, width // 2, dtype=torch.uint8, device=device)
        self.scales = torch.empty(planes, capacity, width // packed.BLOCK, dtype=torch.float8_e4m3fn, device=device)
        self.row_scales = torch.empty(planes, capacity, dtype=torch.float32, device=device)
        self.length = 0                   # positions whose rows are valid and findable
        self.names: list[bytes] = []      # chain names of the findable blocks
        self.writer: str | None = None    # request appending to it, if any
        self.pending: list[bytes] = []    # names the writer will register on finish

    @property
    def capacity(self) -> int:
        return self.codes.shape[1]

    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel() + self.row_scales.numel() * 4


class RowsView:
    """Rows of positions 0..upto of a segment; indexing by plane unpacks that plane only."""

    def __init__(self, segment: Segment, upto: int, dtype: torch.dtype = torch.float32) -> None:
        self.segment, self.upto, self.dtype = segment, upto, dtype

    def __getitem__(self, plane: int) -> torch.Tensor:
        s, n = self.segment, self.upto
        return packed.unpack(s.codes[plane, :n], s.scales[plane, :n], s.row_scales[plane, :n], self.dtype)


class GpuStore:
    def __init__(self, budget_bytes: int, device: torch.device, planes: int = 12, width: int = 1024,
                 block: int = 16, rewrite_blocks: int = 64) -> None:
        self.budget, self.device, self.planes, self.width = budget_bytes, device, planes, width
        self.block, self.rewrite_blocks = block, rewrite_blocks
        self.per_position = planes * packed.bytes_per_row(width)
        self.segments: OrderedDict[int, Segment] = OrderedDict()   # oldest first
        self.index: dict[bytes, tuple[Segment, int]] = {}
        self.bytes = 0

    def _names(self, token_ids, count: int) -> list[bytes]:
        ids = np.asarray(token_ids[:count * self.block], dtype="<i4")
        names, name = [], b""
        for index in range(count):
            name = hashlib.sha256(name + ids[index * self.block:(index + 1) * self.block].tobytes()).digest()
            names.append(name)
        return names

    def find(self, token_ids, positions: int) -> Segment | None:
        """The segment holding the rows of positions 0..positions of this prompt, if any."""
        if positions <= 0 or positions % self.block:
            return None
        count = positions // self.block
        entry = self.index.get(self._names(token_ids, count)[-1])
        if entry is None or entry[1] != count - 1:
            return None
        return entry[0]

    def begin(self, request_id: str, token_ids, first: int, cut: int) -> tuple[Segment, int] | None:
        """Reserve a segment that will hold this prompt's rows of positions 0..cut. Returns it with the
        position from which rows must be written (rows before it are already there), or None if the
        rows of 0..first, which vLLM did not compute, are not kept."""
        count = cut // self.block
        names = self._names(token_ids, count)
        source = self.find(token_ids, first) if first else None
        if first and source is None:
            return None
        low, high = 0, count  # names form a chain: the longest kept start by binary search
        while low < high:
            middle = (low + high + 1) // 2
            if names[middle - 1] in self.index:
                low = middle
            else:
                high = middle - 1
        matched = low
        segment = self.index[names[matched - 1]][0] if matched else None
        if segment is not None and segment.writer is None and matched * self.block >= first:
            if matched == count:
                segment.writer, segment.pending = request_id, names
                return segment, cut
            if len(segment.names) - matched <= self.rewrite_blocks:
                self._truncate(segment, matched)
                self._reserve(segment, cut)
                segment.writer, segment.pending = request_id, names
                return segment, matched * self.block
        fresh = self._allocate(cut)
        if first:
            fresh.codes[:, :first] = source.codes[:, :first]
            fresh.scales[:, :first] = source.scales[:, :first]
            fresh.row_scales[:, :first] = source.row_scales[:, :first]
        fresh.writer, fresh.pending = request_id, names
        self.segments[id(fresh)] = fresh
        self.bytes += fresh.nbytes()
        return fresh, first

    def write(self, segment: Segment, start: int, rows: torch.Tensor) -> None:
        """Store rows [planes, n, width] for positions start..start+n."""
        codes, scales, row_scales = packed.pack(rows)
        end = start + rows.shape[1]
        segment.codes[:, start:end] = codes
        segment.scales[:, start:end] = scales
        segment.row_scales[:, start:end] = row_scales

    def view(self, segment: Segment, upto: int) -> RowsView:
        return RowsView(segment, upto)

    def finish(self, segment: Segment, cut: int) -> None:
        count = cut // self.block
        if count > len(segment.names):
            for number in range(len(segment.names), count):
                self.index[segment.pending[number]] = (segment, number)
            segment.names = segment.pending[:count]
            segment.length = count * self.block
        segment.writer, segment.pending = None, []
        self.segments.move_to_end(id(segment))
        self._evict(keep=segment)

    def abort(self, segment: Segment) -> None:
        segment.writer, segment.pending = None, []
        if not segment.names and id(segment) in self.segments:  # never finished
            del self.segments[id(segment)]
            self.bytes -= segment.nbytes()

    def _truncate(self, segment: Segment, blocks: int) -> None:
        for name in segment.names[blocks:]:
            if self.index.get(name, (None,))[0] is segment:
                del self.index[name]
        segment.names = segment.names[:blocks]
        segment.length = blocks * self.block

    def _allocate(self, positions: int) -> Segment:
        limit = self.budget // self.per_position
        room = max(positions, min(int(positions * 1.25), limit))
        capacity = -(-room // self.block) * self.block
        return Segment(capacity, self.planes, self.width, self.device)

    def _reserve(self, segment: Segment, positions: int) -> None:
        if positions <= segment.capacity:
            return
        grown = self._allocate(positions)
        n = segment.length
        grown.codes[:, :n], grown.scales[:, :n], grown.row_scales[:, :n] = (
            segment.codes[:, :n], segment.scales[:, :n], segment.row_scales[:, :n])
        self.bytes += grown.nbytes() - segment.nbytes()
        segment.codes, segment.scales, segment.row_scales = grown.codes, grown.scales, grown.row_scales

    def _evict(self, keep: Segment) -> None:
        for key in list(self.segments):
            if self.bytes <= self.budget:
                break
            segment = self.segments[key]
            if segment is keep or segment.writer is not None:
                continue
            del self.segments[key]
            self.bytes -= segment.nbytes()
            for name in segment.names:
                if self.index.get(name, (None,))[0] is segment:
                    del self.index[name]

    def stats(self) -> dict:
        return {"segments": len(self.segments), "positions": sum(s.length for s in self.segments.values()),
                "megabytes": round(self.bytes / 2**20, 1), "budget_megabytes": round(self.budget / 2**20)}
