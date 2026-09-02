"""
PyTorch INT8 and INT4 quantization for production edge deployment.

Implements:
- Dynamic INT8 quantization (no calibration data needed)
- Static INT8 quantization (with calibration)
- INT4 weight-only quantization (via manual rounding)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.quantization as tq
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class QuantizationConfig:
    """Configuration for a quantization pass."""
    method: str = "dynamic_int8"      # dynamic_int8 | static_int8 | int4_weight
    dtype: str = "int8"               # int8 | int4
    per_channel: bool = True
    calibration_batches: int = 10
    backend: str = "fbgemm"           # fbgemm (x86) | qnnpack (ARM)


@dataclass
class QuantizationResult:
    """Holds the quantized model and associated metrics."""
    original_model: Any
    quantized_model: Any
    method: str
    original_size_mb: float
    quantized_size_mb: float
    original_latency_ms: float
    quantized_latency_ms: float
    accuracy_delta: float = 0.0
    success: bool = True
    error: Optional[str] = None

    @property
    def size_reduction(self) -> float:
        if self.original_size_mb == 0:
            return 0.0
        return 1.0 - self.quantized_size_mb / self.original_size_mb

    @property
    def latency_improvement(self) -> float:
        if self.quantized_latency_ms == 0:
            return 0.0
        return self.original_latency_ms / self.quantized_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "original_size_mb": round(self.original_size_mb, 3),
            "quantized_size_mb": round(self.quantized_size_mb, 3),
            "size_reduction": round(self.size_reduction, 3),
            "original_latency_ms": round(self.original_latency_ms, 3),
            "quantized_latency_ms": round(self.quantized_latency_ms, 3),
            "latency_improvement": round(self.latency_improvement, 2),
            "accuracy_delta": round(self.accuracy_delta, 4),
            "success": self.success,
            "error": self.error,
        }


class RealQuantizer:
    """
    Production-grade PyTorch quantization.

    Supports dynamic INT8 (fastest to apply), static INT8 (best accuracy/
    speed tradeoff), and INT4 weight-only (smallest model size).
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required. Install with: pip install torch")
        self.config = config or QuantizationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self,
        model: "nn.Module",
        test_input: "torch.Tensor",
        method: Optional[str] = None,
    ) -> QuantizationResult:
        """
        Quantize ``model`` using the specified method.

        Args:
            model: PyTorch module to quantize.
            test_input: Representative input tensor for latency benchmarking.
            method: Override for self.config.method.

        Returns:
            QuantizationResult with metrics and quantized model.
        """
        method = method or self.config.method
        original_size = _model_size_mb(model)
        original_latency = _benchmark_latency(model, test_input)

        try:
            if method == "dynamic_int8":
                q_model = self._dynamic_int8(model)
            elif method == "static_int8":
                q_model = self._static_int8(model, test_input)
            elif method == "int4_weight":
                q_model = self._int4_weight_only(model)
            else:
                raise ValueError(f"Unknown quantization method: {method}")

            q_size = _model_size_mb(q_model)
            q_latency = _benchmark_latency(q_model, test_input)

            return QuantizationResult(
                original_model=model,
                quantized_model=q_model,
                method=method,
                original_size_mb=original_size,
                quantized_size_mb=q_size,
                original_latency_ms=original_latency,
                quantized_latency_ms=q_latency,
            )
        except Exception as exc:
            return QuantizationResult(
                original_model=model,
                quantized_model=model,
                method=method,
                original_size_mb=original_size,
                quantized_size_mb=original_size,
                original_latency_ms=original_latency,
                quantized_latency_ms=original_latency,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal quantization implementations
    # ------------------------------------------------------------------

    def _dynamic_int8(self, model: "nn.Module") -> "nn.Module":
        """
        Dynamic INT8 quantization — weights quantized ahead of time,
        activations quantized dynamically at inference.
        No calibration data required.
        """
        import copy
        m = copy.deepcopy(model)
        m.eval()
        quantized = tq.quantize_dynamic(
            m,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )
        return quantized

    def _static_int8(
        self,
        model: "nn.Module",
        calibration_input: "torch.Tensor",
    ) -> "nn.Module":
        """
        Static INT8 quantization — both weights and activations are
        quantized using scale factors calibrated from representative data.
        """
        import copy
        m = copy.deepcopy(model)
        m.eval()

        backend = self.config.backend
        m.qconfig = tq.get_default_qconfig(backend)
        tq.prepare(m, inplace=True)

        # Calibration pass
        with torch.no_grad():
            for _ in range(self.config.calibration_batches):
                m(calibration_input)

        tq.convert(m, inplace=True)
        return m

    def _int4_weight_only(self, model: "nn.Module") -> "nn.Module":
        """
        INT4 weight-only quantization via manual rounding to 4-bit
        precision. Activations remain in float32 at inference time.
        """
        import copy
        m = copy.deepcopy(model)
        m.eval()

        with torch.no_grad():
            for module in m.modules():
                if isinstance(module, nn.Linear) and module.weight is not None:
                    w = module.weight.data.float()
                    w_min, w_max = w.min(), w.max()
                    scale = (w_max - w_min) / 15.0  # 4-bit: 0..15
                    if scale > 0:
                        w_q = torch.clamp(torch.round((w - w_min) / scale), 0, 15)
                        module.weight.data = (w_q * scale + w_min).to(module.weight.dtype)
        return m


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _model_size_mb(model: "nn.Module") -> float:
    """Return the in-memory parameter size of ``model`` in megabytes."""
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in model.parameters()
    )
    total_bytes += sum(
        b.nelement() * b.element_size()
        for b in model.buffers()
    )
    return total_bytes / (1024 ** 2)


def _benchmark_latency(
    model: "nn.Module",
    test_input: "torch.Tensor",
    warmup: int = 5,
    runs: int = 20,
) -> float:
    """Return mean inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(test_input)
        start = time.perf_counter()
        for _ in range(runs):
            model(test_input)
        elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
