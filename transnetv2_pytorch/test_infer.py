def test_import_and_init():
    from transnetv2_pytorch.inference import TransNetV2Torch

    model = TransNetV2Torch()
    assert model is not None
