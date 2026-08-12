import torch

from ml_engine.models.generator import CompactRRDBNet, output_4_by_3


def test_model_native_and_output_geometry():
    model = CompactRRDBNet().eval()
    source = torch.rand(1, 3, 36, 48)
    with torch.no_grad():
        assert model(source).shape == (1, 3, 72, 96)
        assert output_4_by_3(model, source).shape == (1, 3, 48, 64)
