import torch
from model import build_model


def main():
    model = build_model(
        num_classes=7,
        pretrained_backbone=False,
        embedding_dimension=64,
        se_reduction=8,
        se_levels=["fine", "mid", "global"],
        token_size=2,
        transformer_depth=1,
        transformer_heads=4,
        dropout=0.1,
        landmark_checkpoint=None,
        freeze_landmark_encoder=False
    )
    images = torch.randn(2, 3, 64, 64)
    landmarks = torch.rand(2, 3, 64, 64)
    output = model(images, landmarks, return_features=True)
    assert output["logits"].shape == (2, 7)
    assert len(output["refined_features"]) == 3
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
