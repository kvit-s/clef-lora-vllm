"""Reusing the already-read start of a decision prompt.

Clef's scoring network (JointSchemaHead in joint_schema_model.py) uses the prompt positions before
the first question in one way only: as the memory its six attention layers read (the two routing
layers, and the cross-attention of the four decoder layers). Each position's hidden state is
normalised and projected to 1,024 numbers, and every one of those six layers projects that row again
into a key and a value. The backbone is causal, so these 12 rows of 1,024 numbers depend only on the
tokens up to that position, and they are the same whatever follows.

When vLLM serves the start of a prompt from its prefix cache it skips those positions, so the capture
file only holds the positions from the first one vLLM computed. This module:

- computes the 12 key and value rows for positions (memory_kv);
- runs the scoring network from the hidden states of the computed positions plus the stored key and
  value rows of the skipped ones (forward). It follows JointSchemaHead.forward step for step and gives
  the same probabilities; test_reuse.py checks that against the original on the 72 test records. The
  rows can be given in parts (the stored start, then the new rows), which are attended to as one
  memory without being joined into one tensor;
- keeps the key and value rows of the starts of recently read prompts (PrefixStore), one buffer per
  start. A prompt that extends a kept start, as a game's growing log does, appends its new rows to
  that buffer, and reading the kept rows returns a view of it, so nothing is copied per request.

joint_schema_model.py is Cloudflare's, loaded unchanged from the downloaded release (clef_code.py).
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as functional

from joint_schema_model import EncodedRecord, JointSchemaHead


def attention_modules(head: JointSchemaHead) -> list[torch.nn.MultiheadAttention]:
    """The six attention layers that read the memory, in the order their key and value rows are stored."""
    modules = [layer.attention for layer in head.evidence_layers] + [layer.multihead_attn for layer in head.layers]
    for module in modules:
        if not module._qkv_same_embed_dim or module.in_proj_bias is None or module.bias_k is not None \
                or module.add_zero_attn or not module.batch_first:
            raise ValueError("unexpected attention layout in the scoring network")
    return modules


def first_needed_position(record: EncodedRecord) -> int:
    """The first position whose hidden state the scoring network reads directly: the start of the first
    question or option span. Positions before it are only read through their key and value rows."""
    starts = [q.question_span[0] for q in record.questions]
    starts += [span[0] for q in record.questions for span in q.option_spans]
    return min(starts)


def memory_kv(head: JointSchemaHead, hidden: torch.Tensor) -> torch.Tensor:
    """Key and value rows for positions, shape [12, positions, width], from fp32 hidden states."""
    memory = head.memory_projection(head.hidden_norm(hidden))
    rows = []
    for index, module in enumerate(attention_modules(head)):
        # the routing layers normalise the memory before attending; the decoder layers do not
        source = head.evidence_layers[index].memory_norm(memory) if index < len(head.evidence_layers) else memory
        width = module.embed_dim
        rows.append(functional.linear(source, module.in_proj_weight[width:2 * width], module.in_proj_bias[width:2 * width]))
        rows.append(functional.linear(source, module.in_proj_weight[2 * width:], module.in_proj_bias[2 * width:]))
    return torch.stack(rows)


def _attend(module: torch.nn.MultiheadAttention, query: torch.Tensor, keys: list[torch.Tensor],
            values: list[torch.Tensor]) -> torch.Tensor:
    """nn.MultiheadAttention(query, memory, memory) for one unbatched sequence, with the memory's key
    and value projections already applied. The memory may come in several parts (consecutive
    positions); one softmax runs over all of them, so they never have to be joined. With a few
    queries against tens of thousands of positions this is also faster on the CPU than
    scaled_dot_product_attention."""
    width, heads = module.embed_dim, module.num_heads
    depth = width // heads
    q = functional.linear(query, module.in_proj_weight[:width], module.in_proj_bias[:width])
    q = q.view(-1, heads, depth).transpose(0, 1)  # [heads, queries, depth]
    scores = torch.cat([torch.matmul(q, k.view(-1, heads, depth).permute(1, 2, 0)) for k in keys], dim=-1)
    weights = (scores / math.sqrt(depth)).softmax(-1)
    out, start = 0, 0
    for v in values:
        out = out + torch.matmul(weights[..., start:start + v.shape[0]], v.view(-1, heads, depth).transpose(0, 1))
        start += v.shape[0]
    out = out.transpose(0, 1).reshape(-1, width)
    return functional.linear(out, module.out_proj.weight, module.out_proj.bias)


def forward(
    head: JointSchemaHead,
    hidden: torch.Tensor,
    first: int,
    kv: torch.Tensor | list[torch.Tensor],
    input_ids: torch.Tensor,
    record: EncodedRecord,
    output_embedding_weight,
) -> list[torch.Tensor]:
    """One record's option logits, as JointSchemaHead.forward computes them.

    hidden: fp32 hidden states of positions first..end, shape [end - first, hidden_size];
    kv: key and value rows of all positions 0..end (memory_kv), shape [12, end, width], or a list of
        such tensors for consecutive ranges of positions that together cover 0..end;
    input_ids: all token ids, shape [end].
    """
    if first > first_needed_position(record):
        raise ValueError(f"positions up to {first} were skipped, but the scoring network reads position "
                         f"{first_needed_position(record)} directly")
    parts = list(kv) if isinstance(kv, (list, tuple)) else [kv]
    sequence_hidden = head.hidden_norm(hidden)
    routing_count = len(head.evidence_layers)

    def mean_span(span):
        start, end = span
        return sequence_hidden[start - first:end - first].mean(dim=0)

    global_vector = sequence_hidden[-1]
    question_vectors = torch.stack([mean_span(q.question_span) for q in record.questions])
    type_ids = torch.tensor([q.question_type for q in record.questions])
    option_contexts, lexical_options, option_counts = [], [], []
    for question in record.questions:
        option_contexts.append(torch.stack([mean_span(span) for span in question.option_spans]))
        lexical_options.append(torch.stack([
            output_embedding_weight[input_ids[start:end]].mean(dim=0) for start, end in question.option_spans]))
        option_counts.append(len(question.option_spans))

    option_queries = []
    for question_index, (context_vectors, lexical) in enumerate(zip(option_contexts, lexical_options)):
        option_queries.append(
            head.option_context_projection(context_vectors)
            + head.option_lexical_projection(lexical)
            + head.option_question_projection(question_vectors[question_index]).unsqueeze(0)
        )
    routed_options = torch.cat(option_queries, dim=0)
    for index, layer in enumerate(head.evidence_layers):
        routed = _attend(layer.attention, layer.query_norm(routed_options),
                         [p[2 * index] for p in parts], [p[2 * index + 1] for p in parts])
        routed_options = routed_options + routed
        routed_options = routed_options + layer.feedforward(layer.feedforward_norm(routed_options))
    split_options = list(torch.split(routed_options, option_counts, dim=0))

    base_fields = head.question_projection(question_vectors)
    option_summaries = []
    for field, options in zip(base_fields, split_options):
        routing_weights = torch.softmax(torch.matmul(options, field) / math.sqrt(options.shape[-1]), dim=0)
        option_summaries.append(torch.sum(routing_weights.unsqueeze(-1) * options, dim=0))
    fields = (
        base_fields
        + head.option_summary_norm(torch.stack(option_summaries))
        + head.global_projection(global_vector).unsqueeze(0)
        + head.type_embedding(type_ids)
    )
    fields = fields.unsqueeze(0)
    for index, layer in enumerate(head.layers):
        if not layer.norm_first:
            raise ValueError("unexpected decoder layout in the scoring network")
        slot = 2 * (routing_count + index)
        fields = fields + layer._sa_block(layer.norm1(fields), None, None)
        attended = _attend(layer.multihead_attn, layer.norm2(fields)[0], [p[slot] for p in parts], [p[slot + 1] for p in parts])
        fields = fields + layer.dropout2(attended).unsqueeze(0)
        fields = fields + layer._ff_block(layer.norm3(fields))
    fields = head.field_norm(fields[0])

    record_logits: list[torch.Tensor] = []
    for field, question, lexical, routed in zip(fields, record.questions, lexical_options, split_options):
        anchor = functional.normalize(question_vectors[len(record_logits)] + global_vector, dim=-1)
        lexical_anchor = functional.normalize(lexical, dim=-1)
        prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
        prior = prior_scale * torch.matmul(lexical_anchor, anchor)
        options = head.option_norm(routed)
        repeated_field = field.unsqueeze(0).expand_as(options)
        cosine = functional.cosine_similarity(repeated_field, options, dim=-1)
        features = torch.cat(
            [repeated_field, options, repeated_field * options, torch.abs(repeated_field - options)], dim=-1)
        residual = head.residual_scorer(features).squeeze(-1)
        joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
        joint = joint_scale * cosine + residual
        record_logits.append(prior + torch.sigmoid(head.residual_gate) * joint)
    return record_logits


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
