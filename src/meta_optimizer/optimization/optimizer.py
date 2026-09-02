"""
Real model optimization: quantization, pruning, and knowledge distillation.

Implements actual compression algorithms (not placeholders):
- INT8 / INT4 quantization via PyTorch
- Structured and unstructured pruning
- Knowledge distillation (student/teacher)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as torch_prune

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    bits: int = 8  # 8 = INT8, 4 = INT4
    symmetric: bool = True
    per_channel: bool = False
    calibration_samples: int = 100


@dataclass
class PruningConfig:
    """Configuration for pruning."""

    amount: float = 0.3  # fraction of weights to prune (0.0 – 1.0)
    structured: bool = False  # True = structured (channel), False = unstructured
    dim: int = 0  # dimension for structured pruning


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.7  # weight for distillation loss vs task loss
    epochs: int = 5
    learning_rate: float = 1e-4


@dataclass
class OptimizationResult:
    """Result of an optimization step."""

    success: bool
    method: str
    original_params: int = 0
    compressed_params: int = 0
    compression_ratio: float = 1.0
    sparsity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_reduction_pct(self) -> float:
        if self.original_params == 0:
            return 0.0
        return (1.0 - self.compressed_params / self.original_params) * 100


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


def _count_params(model: Any) -> int:
    """Count total parameters in a PyTorch model."""
    if not _TORCH_AVAILABLE:
        return 0
    return sum(p.numel() for p in model.parameters())


class Quantizer:
    """
    Real INT8/INT4 quantization using PyTorch's quantization API.

    For INT8, uses ``torch.quantization.quantize_dynamic`` which replaces
    supported layers (Linear, LSTM, etc.) with quantized counterparts.

    For INT4-style, applies manual min-max symmetric quantization to
    weight tensors and stores them as INT8 (PyTorch does not have native
    INT4 storage, so INT4 is simulated via clamped INT8 scaling).
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()

    def quantize(self, model: Any) -> Tuple[Any, OptimizationResult]:
        """
        Quantize the model.

        Args:
            model: A ``torch.nn.Module``.

        Returns:
            Tuple of (quantized_model, OptimizationResult).
        """
        if not _TORCH_AVAILABLE:
            return model, OptimizationResult(
                success=False,
                method="quantization",
                details={"error": "PyTorch not available"},
            )

        original_params = _count_params(model)

        if self.config.bits == 8:
            quantized = self._quantize_int8(model)
        elif self.config.bits == 4:
            quantized = self._quantize_int4(model)
        else:
            raise ValueError(f"Unsupported bits={self.config.bits}. Use 8 or 4.")

        compressed_params = _count_params(quantized)

        return quantized, OptimizationResult(
            success=True,
            method=f"quantization_int{self.config.bits}",
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / max(compressed_params, 1),
            details={
                "bits": self.config.bits,
                "symmetric": self.config.symmetric,
                "per_channel": self.config.per_channel,
            },
        )

    def _quantize_int8(self, model: Any) -> Any:
        import torch

        model_copy = copy.deepcopy(model)
        model_copy.eval()
        return torch.quantization.quantize_dynamic(
            model_copy,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

    def _quantize_int4(self, model: Any) -> Any:
        """Simulate INT4 via min-max symmetric quantization of weights."""
        import torch

        model_copy = copy.deepcopy(model)
        model_copy.eval()

        max_val = 2 ** (4 - 1) - 1  # 7

        with torch.no_grad():
            for module in model_copy.modules():
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight.data
                    scale = w.abs().max() / max_val if w.abs().max() > 0 else 1.0
                    w_quant = torch.clamp(torch.round(w / scale), -max_val, max_val)
                    module.weight.data = (w_quant * scale).to(w.dtype)

        return model_copy


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------


class Pruner:
    """
    Real structured and unstructured pruning using PyTorch prune API.

    Unstructured pruning removes individual weights (L1-norm based).
    Structured pruning removes entire channels/filters (L2-norm based).
    """

    def __init__(self, config: Optional[PruningConfig] = None) -> None:
        self.config = config or PruningConfig()

    def prune(self, model: Any) -> Tuple[Any, OptimizationResult]:
        """
        Prune the model.

        Args:
            model: A ``torch.nn.Module``.

        Returns:
            Tuple of (pruned_model, OptimizationResult).
        """
        if not _TORCH_AVAILABLE:
            return model, OptimizationResult(
                success=False,
                method="pruning",
                details={"error": "PyTorch not available"},
            )

        model_copy = copy.deepcopy(model)
        original_params = _count_params(model_copy)

        if self.config.structured:
            self._structured_prune(model_copy)
        else:
            self._unstructured_prune(model_copy)

        sparsity = self._compute_sparsity(model_copy)

        # Make pruning permanent
        for module in model_copy.modules():
            if hasattr(module, "weight_mask"):
                try:
                    torch_prune.remove(module, "weight")
                except ValueError:
                    pass

        compressed_params = _count_params(model_copy)

        return model_copy, OptimizationResult(
            success=True,
            method="structured_pruning" if self.config.structured else "unstructured_pruning",
            original_params=original_params,
            compressed_params=compressed_params,
            compression_ratio=original_params / max(compressed_params, 1),
            sparsity=sparsity,
            details={
                "amount": self.config.amount,
                "structured": self.config.structured,
            },
        )

    def _unstructured_prune(self, model: Any) -> None:
        import torch.nn as nn

        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]
        if parameters_to_prune:
            torch_prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=self.config.amount,
            )

    def _structured_prune(self, model: Any) -> None:
        import torch.nn as nn

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch_prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.config.amount,
                    n=2,
                    dim=self.config.dim,
                )

    def _compute_sparsity(self, model: Any) -> float:
        import torch

        total, zeros = 0, 0
        for module in model.modules():
            if hasattr(module, "weight") and module.weight is not None:
                mask_attr = "weight_mask"
                if hasattr(module, mask_attr):
                    mask = getattr(module, mask_attr)
                    total += mask.numel()
                    zeros += (mask == 0).sum().item()
                else:
                    w = module.weight.data
                    total += w.numel()
                    zeros += (w == 0).sum().item()
        return zeros / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------


class KnowledgeDistillation:
    """
    Knowledge distillation: train a small student model using a large teacher.

    Uses soft targets (KL-divergence on logits) combined with hard target
    cross-entropy loss, weighted by ``alpha``.

    Reference: Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        self.config = config or DistillationConfig()

    def distill(
        self,
        teacher: Any,
        student: Any,
        train_loader: Any,
        device: str = "cpu",
    ) -> Tuple[Any, OptimizationResult]:
        """
        Distill knowledge from teacher into student.

        Args:
            teacher: Large teacher ``torch.nn.Module`` (frozen).
            student: Smaller student ``torch.nn.Module`` (trainable).
            train_loader: PyTorch DataLoader yielding (inputs, labels).
            device: Device string, e.g. ``"cpu"`` or ``"cuda"``.

        Returns:
            Tuple of (trained_student, OptimizationResult).
        """
        if not _TORCH_AVAILABLE:
            return student, OptimizationResult(
                success=False,
                method="knowledge_distillation",
                details={"error": "PyTorch not available"},
            )

        import torch
        import torch.nn.functional as F

        teacher_params = _count_params(teacher)
        student_params = _count_params(student)

        teacher = teacher.to(device).eval()
        student = student.to(device).train()

        optimizer = torch.optim.Adam(student.parameters(), lr=self.config.learning_rate)
        T = self.config.temperature
        alpha = self.config.alpha

        epoch_losses: List[float] = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batches = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(inputs)

                student_logits = student(inputs)

                # Soft loss: KL-divergence on temperature-scaled logits
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)

                # Hard loss: cross-entropy with true labels
                hard_loss = F.cross_entropy(student_logits, labels)

                loss = alpha * soft_loss + (1 - alpha) * hard_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            epoch_losses.append(avg_loss)

        return student, OptimizationResult(
            success=True,
            method="knowledge_distillation",
            original_params=teacher_params,
            compressed_params=student_params,
            compression_ratio=teacher_params / max(student_params, 1),
            details={
                "temperature": T,
                "alpha": alpha,
                "epochs": self.config.epochs,
                "final_loss": epoch_losses[-1] if epoch_losses else None,
                "epoch_losses": epoch_losses,
                "student_teacher_ratio": student_params / max(teacher_params, 1),
            },
        )
