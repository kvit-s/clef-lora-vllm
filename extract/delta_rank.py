"""Print the singular values of the difference between Clef and the base model for chosen tensors.

This is the measurement that shows Clef's change is a merged rank-256 LoRA: for the matrices of
layers 40-63, 99.9% of the difference lies in its top 256 singular directions, and the singular
values drop sharply right after the 256th (layer 43's q_proj: 0.120 at the 256th, 0.0063 at the
257th).

usage: delta_rank.py TENSOR [TENSOR ...]
       e.g. delta_rank.py model.language_model.layers.40.mlp.down_proj.weight
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_adapter import Hub, load  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    hub = Hub()
    with tempfile.TemporaryDirectory() as work:
        for name in sys.argv[1:]:
            (clef_path, shape), (base_path, _) = hub.tensor("clef", name, Path(work)), hub.tensor("base", name, Path(work))
            a, b = load(clef_path, shape), load(base_path, shape)
            d = (a - b).astype(np.float64)
            sv = np.linalg.svd(d, compute_uv=False)
            print(f"{name} {tuple(shape)}: |delta|/|W| = {np.linalg.norm(d) / np.linalg.norm(b):.4f}, "
                  f"entries changed {np.mean(a != b):.1%}")
            for r in (64, 128, 255, 256, 257, 512):
                if r < len(sv):
                    print(f"  share of the difference in the top {r:>3} directions: "
                          f"{np.sum(sv[:r] ** 2) / np.sum(sv ** 2):.5f}   singular value {r:>3}: {sv[r - 1]:.5f}")
            clef_path.unlink()
            base_path.unlink()


if __name__ == "__main__":
    main()
