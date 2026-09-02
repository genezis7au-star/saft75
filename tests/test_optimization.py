"""Tests for model optimization: quantization, pruning, knowledge distillation."""

import pytest

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from meta_optimizer.optimization import (
    DistillationConfig,
    KnowledgeDistillation,
    OptimizationResult,
    PruningConfig,
    Pruner,
    QuantizationConfig,
    Quantizer,
)

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")


def make_model():
    return nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


class TestQuantizer:
    def test_int8_quantization(self):
        model = make_model()
        quantizer = Quantizer(QuantizationConfig(bits=8))
        q_model, result = quantizer.quantize(model)
        assert result.success
        assert result.method == "quantization_int8"
        assert result.original_params > 0

    def test_int4_quantization(self):
        model = make_model()
        quantizer = Quantizer(QuantizationConfig(bits=4))
        q_model, result = quantizer.quantize(model)
        assert result.success
        assert result.method == "quantization_int4"

    def test_invalid_bits_raises(self):
        model = make_model()
        quantizer = Quantizer(QuantizationConfig(bits=3))
        with pytest.raises(ValueError, match="Unsupported bits"):
            quantizer.quantize(model)


class TestPruner:
    def test_unstructured_pruning(self):
        model = make_model()
        pruner = Pruner(PruningConfig(amount=0.5, structured=False))
        p_model, result = pruner.prune(model)
        assert result.success
        assert result.method == "unstructured_pruning"
        assert result.sparsity > 0.0

    def test_structured_pruning(self):
        model = make_model()
        pruner = Pruner(PruningConfig(amount=0.3, structured=True, dim=0))
        p_model, result = pruner.prune(model)
        assert result.success
        assert result.method == "structured_pruning"

    def test_pruning_respects_amount(self):
        model = make_model()
        amount = 0.5
        pruner = Pruner(PruningConfig(amount=amount, structured=False))
        _, result = pruner.prune(model)
        # Sparsity should be close to the requested amount (within 15%)
        assert abs(result.sparsity - amount) < 0.15

    def test_optimization_result_fields(self):
        model = make_model()
        pruner = Pruner(PruningConfig(amount=0.3))
        _, result = pruner.prune(model)
        assert result.original_params > 0
        assert isinstance(result.details, dict)


class TestKnowledgeDistillation:
    def test_distillation_runs(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        teacher = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 4))
        student = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))

        x = torch.randn(20, 8)
        y = torch.randint(0, 4, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=10)

        config = DistillationConfig(temperature=4.0, alpha=0.7, epochs=2, learning_rate=1e-3)
        kd = KnowledgeDistillation(config)
        trained, result = kd.distill(teacher, student, loader)

        assert result.success
        assert result.method == "knowledge_distillation"
        assert result.compression_ratio > 1.0
        assert "epoch_losses" in result.details
        assert len(result.details["epoch_losses"]) == 2

    def test_compression_ratio(self):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        teacher = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
        student = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10))

        x = torch.randn(10, 64)
        y = torch.randint(0, 10, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        kd = KnowledgeDistillation(DistillationConfig(epochs=1))
        _, result = kd.distill(teacher, student, loader)
        # Teacher is much larger than student
        assert result.compression_ratio > 5.0
