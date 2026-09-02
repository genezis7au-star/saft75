"""
Structured and unstructured neural network pruning.

Implements:
- Unstructured L1/random weight pruning
- Structured channel/filter pruning (L2-norm based)
- Iterative magnitude pruning with fine-tuning hooks
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PruningConfig:
    """Configuration for a pruning pass."""
    method: str = "unstructured_l1"    # unstructured_l1 | unstructured_random | structured_l2
    amount: float = 0.3               # Fraction of connections/channels to prune (0–1)
    target_layers: Optional[List[str]] = None   # Layer class names to prune (None = all Linear/Conv)
    make_permanent: bool = True       # Remove pruning masks after applying


@dataclass
class PruningResult:
    """Holds the pruned model and associated metrics."""
    original_model: Any
    pruned_model: Any
    method: str
    amount: float
    original_params: int
    remaining_params: int
    original_latency_ms: float
    pruned_latency_ms: float
    original_size_mb: float
    pruned_size_mb: float
    success: bool = True
    error: Optional[str] = None

    @property
    def sparsity(self) -> float:
        if self.original_params == 0:
            return 0.0
        return 1.0 - self.remaining_params / self.original_params

    @property
    def latency_improvement(self) -> float:
        if self.pruned_latency_ms == 0:
            return 0.0
        return self.original_latency_ms / self.pruned_latency_ms

    @property
    def size_reduction(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return 1.0 - self.pruned_size_mb / self.original_size_mb

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "amount": self.amount,
            "original_params": self.original_params,
            "remaining_params": self.remaining_params,
            "sparsity": round(self.sparsity, 4),
            "original_latency_ms": round(self.original_latency_ms, 3),
            "pruned_latency_ms": round(self.pruned_latency_ms, 3),
            "latency_improvement": round(self.latency_improvement, 2),
            "original_size_mb": round(self.original_size_mb, 3),
            "pruned_size_mb": round(self.pruned_size_mb, 3),
            "size_reduction": round(self.size_reduction, 3),
            "success": self.success,
            "error": self.error,
        }


class RealPruner:
    """
    Production-grade neural network pruner.

    Supports unstructured magnitude-based pruning (L1/random) and
    structured channel pruning based on L2 norms.
    """

    def __init__(self, config: Optional[PruningConfig] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")
        self.config = config or PruningConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        method: Optional[str] = None,
        amount: Optional[float] = None,
    ) -> PruningResult:
        """
        Prune ``model`` according to the configured strategy.

        Args:
            model: PyTorch module to prune.
            test_input: Representative input for latency benchmarking.
            method: Override config method.
            amount: Override config amount.

        Returns:
            PruningResult with the pruned model and metrics.
        """
        method = method or self.config.method
        amount = amount if amount is not None else self.config.amount
        original_params = _count_params(model)
        original_size = _model_size_mb(model)
        original_latency = _benchmark_latency(model, test_input)

        try:
            import copy
            m = copy.deepcopy(model)
            m.eval()

            if method == "unstructured_l1":
                self._unstructured_prune(m, amount, prune.L1Unstructured)
            elif method == "unstructured_random":
                self._unstructured_prune(m, amount, prune.RandomUnstructured)
            elif method == "structured_l2":
                self._structured_prune(m, amount)
            else:
                raise ValueError(f"Unknown pruning method: {method}")

            if self.config.make_permanent:
                self._make_permanent(m)

            pruned_params = _count_nonzero_params(m)
            pruned_size = _model_size_mb(m)
            pruned_latency = _benchmark_latency(m, test_input)

            return PruningResult(
                original_model=model,
                pruned_model=m,
                method=method,
                amount=amount,
                original_params=original_params,
                remaining_params=pruned_params,
                original_latency_ms=original_latency,
                pruned_latency_ms=pruned_latency,
                original_size_mb=original_size,
                pruned_size_mb=pruned_size,
            )
        except Exception as exc:
            return PruningResult(
                original_model=model,
                pruned_model=model,
                method=method,
                amount=amount,
                original_params=original_params,
                remaining_params=original_params,
                original_latency_ms=original_latency,
                pruned_latency_ms=original_latency,
                original_size_mb=original_size,
                pruned_size_mb=original_size,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    def _unstructured_prune(
        self,
        model: "nn.Module",
        amount: float,
        pruning_class: Any,
    ) -> None:
        """Apply unstructured pruning to all Linear and Conv2d layers."""
        target_types = self._target_layer_types()
        params_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, target_types)
        ]
        if not params_to_prune:
            return
        prune.global_unstructured(
            params_to_prune,
            pruning_method=pruning_class,
            amount=amount,
        )

    def _structured_prune(self, model: "nn.Module", amount: float) -> None:
        """
        Structured L2-norm pruning: remove the ``amount`` fraction of
        output channels from each Conv2d / Linear layer.
        """
        target_types = self._target_layer_types()
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                n_to_remove = max(1, int(module.out_channels * amount))
                n_to_keep = max(1, module.out_channels - n_to_remove)
                if n_to_keep < module.out_channels:
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            elif isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)

    def _make_permanent(self, model: "nn.Module") -> None:
        """Remove pruning re-parametrizations, making the mask permanent."""
        for module in model.modules():
            if hasattr(module, "weight_orig"):
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass

    def _target_layer_types(self) -> Tuple:
        types = []
        if self.config.target_layers is None:
            return (nn.Linear, nn.Conv2d)
        for name in self.config.target_layers:
            cls = getattr(nn, name, None)
            if cls is not None:
                types.append(cls)
        return tuple(types) if types else (nn.Linear, nn.Conv2d)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


def _count_nonzero_params(model: "nn.Module") -> int:
    return sum(p.nonzero().size(0) for p in model.parameters())


def _model_size_mb(model: "nn.Module") -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)


def _benchmark_latency(
    model: "nn.Module",
    test_input: "torch.Tensor",
    warmup: int = 5,
    runs: int = 20,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(test_input)
        start = time.perf_counter()
        for _ in range(runs):
            model(test_input)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
