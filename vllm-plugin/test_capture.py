"""CPU tests for clef_capture's bookkeeping, with stand-ins for vLLM's runner state.

Run: python test_capture.py (needs torch and numpy; vLLM itself is not imported).
Hidden-state values encode (request, position, column) as small integers, which bf16 holds
exactly, so the written file can be compared value for value.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "service"))
import clef_capture  # noqa: E402
from capture_file import read  # noqa: E402

HIDDEN = 16


def rows_for(code, start, count):
    positions = torch.arange(start, start + count, dtype=torch.float32)[:, None]
    columns = torch.arange(HIDDEN, dtype=torch.float32)[None, :] % 7
    return (positions + 20 * code + columns).to(torch.bfloat16)


def expected(code, length):
    return rows_for(code, 0, length)


def v2_step(runner, batch):
    """batch: list of (req_id, code, start, count). Builds the step's hidden states in batch order."""
    hidden = torch.cat([rows_for(code, start, count) for _, code, start, count in batch])
    counts = np.array([c for *_, c in batch], dtype=np.int32)
    runner.execute_model_state = SimpleNamespace(
        hidden_states=hidden,
        input_batch=SimpleNamespace(
            req_ids=[r for r, *_ in batch], num_reqs=len(batch),
            num_computed_tokens_np=np.array([s for _, _, s, _ in batch], dtype=np.int32),
            query_start_loc_np=np.concatenate([[0], np.cumsum(counts)]).astype(np.int32),
            num_scheduled_tokens=counts,
        ),
    )
    clef_capture._capture_v2(runner)


def test_v2(directory):
    runner = SimpleNamespace(_clef_capture=clef_capture.Capture(directory))
    prompt_a = list(range(500, 510))
    runner.__dict__["_clef_marked"] = {"A": ("key-aaaaaaaa", prompt_a), "C": ("key-cccccccc", list(range(7)))}
    v2_step(runner, [("B", 2, 30, 3), ("A", 1, 0, 4)])
    assert not os.path.exists(os.path.join(directory, "key-aaaaaaaa.bin")), "written before the prompt was complete"
    v2_step(runner, [("A", 1, 4, 4), ("B", 2, 33, 1)])
    v2_step(runner, [("A", 1, 8, 3)])  # one position past the prompt: the token being sampled
    header, hidden = read(os.path.join(directory, "key-aaaaaaaa.bin"), prompt_a)
    assert header["prompt_tokens"] == 10 and header["hidden_size"] == HIDDEN
    assert torch.equal(hidden, expected(1, 10)), "rows out of place"
    # a marked request whose first chunk does not start at 0 came partly from the prefix cache:
    # the file holds the positions from the first computed one, and says where that is
    v2_step(runner, [("C", 3, 4, 3)])
    header, hidden = read(os.path.join(directory, "key-cccccccc.bin"), list(range(7)))
    assert header["first_position"] == 4 and header["prompt_tokens"] == 7
    assert torch.equal(hidden, rows_for(3, 4, 3)), "rows after a prefix-cache hit out of place"
    assert not os.path.exists(os.path.join(directory, "key-cccccccc.err"))
    # a chunk before the first computed position cannot be placed: reported, nothing written
    runner.__dict__["_clef_marked"]["D"] = ("key-dddddddd", list(range(40)))
    v2_step(runner, [("D", 4, 32, 4)])
    v2_step(runner, [("D", 4, 16, 4)])
    assert os.path.exists(os.path.join(directory, "key-dddddddd.err"))
    assert not os.path.exists(os.path.join(directory, "key-dddddddd.bin"))
    # a wrong prompt must be refused by the reader
    try:
        read(os.path.join(directory, "key-aaaaaaaa.bin"), prompt_a[:-1] + [0])
        raise AssertionError("reader accepted hidden states for a different prompt")
    except ValueError:
        pass
    print("v2: chunked prompt reassembled across batches; positions after a prefix-cache hit written with their start; prompt check works")


def test_v1(directory):
    prompt = list(range(900, 906))
    sp = SimpleNamespace(extra_args={"clef_capture": "key-v1v1v1v1"})
    requests = {
        "X": SimpleNamespace(sampling_params=sp, prompt_token_ids=prompt, num_prompt_tokens=len(prompt)),
        "Y": SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None), prompt_token_ids=[1], num_prompt_tokens=1),
    }
    runner = SimpleNamespace(_clef_capture=clef_capture.Capture(directory), requests=requests)

    def step(batch):
        runner.input_batch = SimpleNamespace(
            req_ids=[r for r, *_ in batch], num_reqs=len(batch),
            num_computed_tokens_cpu=np.array([s for _, _, s, _ in batch], dtype=np.int32))
        runner._clef_scheduler_output = SimpleNamespace(num_scheduled_tokens={r: c for r, _, _, c in batch})
        hidden = torch.cat([rows_for(code, s, c) for _, code, s, c in batch])
        clef_capture._capture_v1(runner, (hidden, ["aux"]))  # the DFlash form: (hidden, aux_hidden_states)

    step([("Y", 2, 10, 1), ("X", 1, 0, 2)])
    step([("X", 1, 2, 4), ("Y", 2, 11, 1)])
    header, hidden = read(os.path.join(directory, "key-v1v1v1v1.bin"), prompt)
    assert torch.equal(hidden, expected(1, 6)) and header["first_position"] == 0
    print("v1: chunked prompt reassembled; (hidden, aux) model output handled")


def test_bad_key():
    assert clef_capture.capture_key(SimpleNamespace(extra_args={"clef_capture": "../../etc/passwd"})) is None
    assert clef_capture.capture_key(SimpleNamespace(extra_args={"clef_capture": "short"})) is None
    assert clef_capture.capture_key(SimpleNamespace(extra_args=None)) is None
    print("keys that could name a path outside the directory are refused")


def test_directory(parent):
    # /dev/shm is empty after a restart: the capture directory may be missing when vLLM starts,
    # or removed while it runs
    directory = os.path.join(parent, "clef-hidden")
    capture = clef_capture.Capture(directory)
    assert os.path.isdir(directory) and os.stat(directory).st_mode & 0o777 == 0o777
    os.rmdir(directory)
    capture.add("Z", "key-zzzzzzzz", [1, 2], 0, rows_for(5, 0, 2))
    assert os.path.exists(os.path.join(directory, "key-zzzzzzzz.bin"))
    print("missing capture directory created, writable by everyone, and again if removed")


def test_lora_layers():
    """The layer limit, against stand-ins for the two ways vLLM chooses LoRA modules.

    vLLM 0.27: every module goes through LoRAModelManager._match_target_modules, which calls
    is_supported_lora_module(name, supported). vLLM 0.31: without --lora-target-modules,
    _create_lora_modules calls is_supported_lora_module(name, module, supported) directly.
    Both call the function by its name in vllm.lora.model_manager, as the stand-ins do.
    """
    import types

    def install(version):
        manager = types.ModuleType("vllm.lora.model_manager")
        exec(
            "def is_supported_lora_module(name, *rest):\n"
            "    return name.rsplit('.', 1)[-1] in ('q_proj', 'down_proj', 'proj')\n"
            "class LoRAModelManager:\n"
            "    target_modules = None\n"
            "    def _match_target_modules(self, name):\n"
            + ("        return is_supported_lora_module(name, [])\n" if version == "0.27" else "        return True\n")
            + "    def create_modules(self, names):\n"
            + ("        return [n for n in names if self._match_target_modules(n)]\n" if version == "0.27" else
               "        return [n for n in names if is_supported_lora_module(n, None, [])]\n"),
            manager.__dict__)
        lora = types.ModuleType("vllm.lora")
        lora.model_manager = manager
        sys.modules.update({"vllm.lora": lora, "vllm.lora.model_manager": manager})
        return manager

    names = ["language_model.model.layers.39.self_attn.q_proj", "language_model.model.layers.40.self_attn.q_proj",
             "language_model.model.layers.63.mlp.down_proj", "language_model.model.layers.40.mlp.act_fn",
             "visual.blocks.40.attn.proj", "lm_head"]
    os.environ["CLEF_LORA_LAYERS"] = "40-63"
    try:
        for version in ("0.27", "0.31"):
            manager = install(version)
            assert clef_capture._patch_lora_layers() is True
            chosen = manager.LoRAModelManager().create_modules(names)
            assert chosen == names[1:3], (version, chosen)
    finally:
        del os.environ["CLEF_LORA_LAYERS"]
        for name in ("vllm.lora", "vllm.lora.model_manager"):
            sys.modules.pop(name, None)
    print("LoRA buffers limited to layers 40-63, with vLLM 0.27's and 0.31's ways of choosing modules")


def test_cache_hits():
    """Decision requests see no held-back blocks and the reuse limit; everything else is unchanged."""
    import types
    from typing import NamedTuple

    class Group(NamedTuple):
        spec: str
        use_eagle: bool

    seen = []

    class Coordinator:
        def __init__(self):
            self.attention_groups = [Group("full", True), Group("mamba", True)]

        def find_longest_cache_hit(self, block_hashes, max_length):
            seen.append(([g.use_eagle for g in self.attention_groups], max_length))
            return "blocks", max_length

    class KVCacheManager:
        def __init__(self):
            self.coordinator = Coordinator()

        def get_computed_blocks(self, request):
            return self.coordinator.find_longest_cache_hit([], request.num_tokens - 1)

    class Scheduler:
        use_eagle = True

        def _mamba_block_aligned_split(self, request, num_new_tokens):
            # the real one backs off a block when use_eagle; record what it would see
            seen.append(("split", self.use_eagle))
            return num_new_tokens

    manager_module = types.ModuleType("vllm.v1.core.kv_cache_manager")
    manager_module.KVCacheManager = KVCacheManager
    scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = Scheduler
    sched_package = types.ModuleType("vllm.v1.core.sched")
    sched_package.scheduler = scheduler_module
    sys.modules.update({"vllm.v1.core": types.ModuleType("vllm.v1.core"), "vllm.v1.core.kv_cache_manager": manager_module,
                        "vllm.v1.core.sched": sched_package, "vllm.v1.core.sched.scheduler": scheduler_module})
    sys.modules["vllm.v1.core"].kv_cache_manager = manager_module
    sys.modules["vllm.v1.core"].sched = sched_package
    os.environ["CLEF_FULL_CACHE_HITS"] = "1"
    try:
        assert clef_capture._patch_decision_cache_hits() is True
        manager = KVCacheManager()
        decision = SimpleNamespace(num_tokens=5000, max_tokens=1, sampling_params=SimpleNamespace(
            extra_args={"clef_capture": "key-hits0001", "clef_reuse_limit": 4200}))
        chat = SimpleNamespace(num_tokens=5000, max_tokens=512, sampling_params=SimpleNamespace(extra_args=None))
        long_marked = SimpleNamespace(num_tokens=5000, max_tokens=8, sampling_params=decision.sampling_params)
        priming = SimpleNamespace(num_tokens=4000, max_tokens=1, sampling_params=SimpleNamespace(extra_args={"clef_prime": True}))
        assert clef_capture._is_decision(priming) and not clef_capture._is_decision(chat)
        manager.get_computed_blocks(decision)
        manager.get_computed_blocks(chat)
        manager.get_computed_blocks(long_marked)
        assert seen[0] == ([False, False], 4200), seen[0]
        assert seen[1] == ([True, True], 4999), seen[1]
        assert seen[2] == ([True, True], 4999), "a marked request that generates more than one token"
        assert [g.use_eagle for g in manager.coordinator.attention_groups] == [True, True]
        assert "find_longest_cache_hit" not in manager.coordinator.__dict__
        scheduler = Scheduler()
        scheduler._mamba_block_aligned_split(decision, 100)
        scheduler._mamba_block_aligned_split(chat, 100)
        assert seen[3:] == [("split", False), ("split", True)], seen[3:]
        assert scheduler.use_eagle is True
    finally:
        del os.environ["CLEF_FULL_CACHE_HITS"]
        for name in ("vllm.v1.core", "vllm.v1.core.kv_cache_manager", "vllm.v1.core.sched", "vllm.v1.core.sched.scheduler"):
            sys.modules.pop(name, None)
    print("decision requests: no held-back blocks, the reuse limit applied, and chunks that reach the last "
          "block boundary; other requests unchanged")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_v2(d)
        test_v1(d)
    with tempfile.TemporaryDirectory() as d:
        test_directory(d)
    test_bad_key()
    test_lora_layers()
    test_cache_hits()
    print("all passed")
