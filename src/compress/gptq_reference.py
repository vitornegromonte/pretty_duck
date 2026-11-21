"""The authors' GPTQ code as an independent cross-check of ``gptq.gptq_quantize``.

The reference code (IST-DASLab/gptq: ``gptq.py`` and ``quant.py``) is not part of this repository. Put the two files in
``third_party/gptq_ref/``. On the same grid and Hessian, both implementations give a similar layer output error
(see ``docs/adr``, E22). The reference has no protected columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "third_party" / "gptq_ref"


def reference_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group: int,
                       protect: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize ``W`` (out, in) with the reference GPTQ on the symmetric group grid; ``H`` is (in, in)."""
    if protect is not None and protect.numel():
        raise ValueError("the reference implementation has no protected columns")
    if not (REFERENCE_DIR / "gptq.py").exists():
        raise FileNotFoundError(f"put the authors' gptq.py and quant.py in {REFERENCE_DIR}")
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    from gptq import GPTQ  # type: ignore[import-not-found]

    class SymmetricGrid:
        """round(w/s) in [-2^(b-1), 2^(b-1)-1] with s = absmax / (2^(b-1)-1), as ``fake_quant_group``."""

        maxq = torch.tensor(2 ** bits - 1)
        zero = torch.tensor(float(2 ** (bits - 1)))

        def ready(self) -> bool:
            return False

        def find_params(self, x: torch.Tensor, weight: bool = True) -> None:
            self.scale = (x.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp_min(1e-12)
            self.zero = torch.full_like(self.scale, float(2 ** (bits - 1)))
            self.maxq = self.maxq.to(x.device)

    layer = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device)
    layer.weight.data = W.float().clone()
    gptq = GPTQ(layer)
    gptq.quantizer = SymmetricGrid()
    gptq.H = 2 * H.to(device=W.device, dtype=torch.float32)  # the reference scales H by 2/n
    gptq.fasterquant(blocksize=128, percdamp=0.01, groupsize=group)
    return layer.weight.data.float()
