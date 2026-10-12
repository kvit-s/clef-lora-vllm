"""The parts of Clef's scoring network that run from stored key and value rows.

Clef's scoring network (JointSchemaHead in Cloudflare's joint_schema_model.py) uses the prompt
positions before the first question in one way only: as the memory its six attention layers read (the
two routing layers, and the cross-attention of the four decoder layers). Each position's hidden state
is normalised and projected to 1,024 numbers, and every one of those six layers projects that row again
into a key and a value. The backbone is causal, so these 12 rows of 1,024 numbers depend only on the
tokens up to that position, and they are the same whatever follows.

- memory_kv computes the 12 key and value rows for positions;
- forward runs the scoring network from the hidden states of the positions from some point on plus the
  key and value rows of all positions, which may come in parts (stored rows, then new ones). It follows
  JointSchemaHead.forward step for step; service/test_reuse.py checks it against the original.

The functions work on whatever device their tensors are on. They are used by the decision service on
the CPU (service/reuse.py) and by this plugin on the GPU (gpu_head.py). Importing this module does not
import vLLM or Cloudflare's code.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


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


def _attend(module: torch.nn.MultiheadAttention, query: torch.Tensor, keys: list, values: list) -> torch.Tensor:
    """nn.MultiheadAttention(query, memory, memory) for one unbatched sequence, with the memory's key
    and value projections already applied. The memory may come in several parts (consecutive
    positions); one softmax runs over all of them, so they never have to be joined. With a few
    queries against tens of thousands of positions this is also faster on the CPU than
    scaled_dot_product_attention. A part may be a tensor or an object with .slices() that yields the
    part's rows a slice at a time (gpu_store.Plane); then the softmax is computed as the slices go by
    (a running maximum and sum, in fp32), so only one slice is ever unpacked."""
    width, heads = module.embed_dim, module.num_heads
    depth = width // heads
    q = functional.linear(query, module.in_proj_weight[:width], module.in_proj_bias[:width])
    q = q.view(-1, heads, depth).transpose(0, 1)  # [heads, queries, depth]
    if not all(torch.is_tensor(k) for k in keys):
        out = _attend_streaming(q, keys, values, depth)
    else:
        scores = torch.cat([torch.matmul(q, k.view(-1, heads, depth).permute(1, 2, 0)) for k in keys], dim=-1)
        weights = (scores.float() / math.sqrt(depth)).softmax(-1).to(q.dtype)  # softmax in fp32 whatever the dtype
        out, start = 0, 0
        for v in values:
            out = out + torch.matmul(weights[..., start:start + v.shape[0]], v.view(-1, heads, depth).transpose(0, 1))
            start += v.shape[0]
    out = out.transpose(0, 1).reshape(-1, width)
    return functional.linear(out, module.out_proj.weight, module.out_proj.bias)


def _slices(part):
    return part.slices() if hasattr(part, "slices") else [part]


def _attend_streaming(q: torch.Tensor, keys: list, values: list, depth: int) -> torch.Tensor:
    heads = q.shape[0]
    peak = torch.full((*q.shape[:2], 1), float("-inf"), device=q.device)
    total = torch.zeros((*q.shape[:2], 1), device=q.device)
    acc = torch.zeros(q.shape, device=q.device)
    for key_part, value_part in zip(keys, values):
        for k, v in zip(_slices(key_part), _slices(value_part)):
            scores = torch.matmul(q, k.view(-1, heads, depth).permute(1, 2, 0)).float() / math.sqrt(depth)
            new_peak = torch.maximum(peak, scores.amax(-1, keepdim=True))
            weights = torch.exp(scores - new_peak)
            correction = torch.exp(peak - new_peak)
            total = total * correction + weights.sum(-1, keepdim=True)
            acc = acc * correction + torch.matmul(weights, v.view(-1, heads, depth).transpose(0, 1).float())
            peak = new_peak
    return (acc / total).to(q.dtype)


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
    type_ids = torch.tensor([q.question_type for q in record.questions], device=hidden.device)
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
