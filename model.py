import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, enabled=True):
        super().__init__()
        self.enabled = enabled
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.last_weights = None

    def forward(self, x):
        if not self.enabled:
            self.last_weights = torch.ones(
                x.shape[0], x.shape[1], 1, 1, device=x.device, dtype=x.dtype
            )
            return x
        weights = self.pool(x)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights))
        self.last_weights = weights
        return x * weights


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class LandmarkFeatureEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 256)):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Sequential(
            ConvBNAct(3, c1, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        self.stage1 = nn.Sequential(ConvBNAct(c1, c1), ConvBNAct(c1, c1))
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, stride=2), ConvBNAct(c2, c2))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, stride=2), ConvBNAct(c3, c3))

    def forward(self, x):
        x = self.stem(x)
        fine = self.stage1(x)
        mid = self.stage2(fine)
        global_feature = self.stage3(mid)
        return fine, mid, global_feature


class ResNet18FeatureBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.stage1 = network.layer1
        self.stage2 = network.layer2
        self.stage3 = network.layer3
        self.out_channels = (64, 128, 256)

    def forward(self, x):
        x = self.stem(x)
        fine = self.stage1(x)
        mid = self.stage2(fine)
        global_feature = self.stage3(mid)
        return fine, mid, global_feature


class ScaleTokenizer(nn.Module):
    def __init__(self, dimension, token_size=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((token_size, token_size))
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class SEPoster(nn.Module):
    level_names = ("fine", "mid", "global")

    def __init__(
        self,
        num_classes=7,
        pretrained_backbone=True,
        embedding_dimension=256,
        se_reduction=16,
        se_levels=("fine", "mid", "global"),
        token_size=7,
        transformer_depth=4,
        transformer_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        landmark_checkpoint=None,
        freeze_landmark_encoder=False
    ):
        super().__init__()
        selected_levels = set(se_levels)
        invalid_levels = selected_levels.difference(self.level_names)
        if invalid_levels:
            raise ValueError(f"Invalid SE levels: {sorted(invalid_levels)}")
        self.image_backbone = ResNet18FeatureBackbone(pretrained_backbone)
        channels = self.image_backbone.out_channels
        self.landmark_encoder = LandmarkFeatureEncoder(channels)
        self.fusion = nn.ModuleList(
            [nn.Conv2d(channel * 2, embedding_dimension, 1, bias=False) for channel in channels]
        )
        self.fusion_norm = nn.ModuleList([nn.BatchNorm2d(embedding_dimension) for _ in channels])
        self.se_blocks = nn.ModuleList(
            [
                SEBlock(embedding_dimension, se_reduction, enabled=level in selected_levels)
                for level in self.level_names
            ]
        )
        self.tokenizers = nn.ModuleList(
            [ScaleTokenizer(embedding_dimension, token_size) for _ in channels]
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=transformer_heads,
            dim_feedforward=int(embedding_dimension * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=transformer_depth)
        total_tokens = len(channels) * token_size * token_size
        self.position_embedding = nn.Parameter(torch.zeros(1, total_tokens, embedding_dimension))
        self.scale_embedding = nn.Parameter(torch.zeros(1, len(channels), 1, embedding_dimension))
        self.output_norm = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dimension, num_classes)
        )
        self._initialize_weights()
        if landmark_checkpoint:
            checkpoint = torch.load(landmark_checkpoint, map_location="cpu")
            state = checkpoint.get("model", checkpoint)
            landmark_state = {
                key.replace("landmark_encoder.", "", 1): value
                for key, value in state.items()
                if key.startswith("landmark_encoder.")
            }
            if not landmark_state:
                landmark_state = state
            self.landmark_encoder.load_state_dict(landmark_state, strict=True)
        if freeze_landmark_encoder:
            if not landmark_checkpoint:
                raise ValueError("landmark_checkpoint is required when freeze_landmark_encoder is true")
            for parameter in self.landmark_encoder.parameters():
                parameter.requires_grad = False
            self.landmark_encoder.eval()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.scale_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def train(self, mode=True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.landmark_encoder.parameters()):
            self.landmark_encoder.eval()
        return self

    def forward_features(self, image, landmark_heatmap):
        image_features = self.image_backbone(image)
        landmark_features = self.landmark_encoder(landmark_heatmap)
        tokens = []
        refined_features = []
        for index, (image_feature, landmark_feature) in enumerate(zip(image_features, landmark_features)):
            if landmark_feature.shape[-2:] != image_feature.shape[-2:]:
                landmark_feature = F.interpolate(
                    landmark_feature,
                    size=image_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            fused = torch.cat([image_feature, landmark_feature], dim=1)
            fused = F.gelu(self.fusion_norm[index](self.fusion[index](fused)))
            refined = self.se_blocks[index](fused)
            refined_features.append(refined)
            scale_tokens = self.tokenizers[index](refined)
            scale_tokens = scale_tokens + self.scale_embedding[:, index]
            tokens.append(scale_tokens)
        sequence = torch.cat(tokens, dim=1)
        sequence = sequence + self.position_embedding[:, :sequence.shape[1]]
        sequence = self.transformer(sequence)
        sequence = self.output_norm(sequence)
        representation = sequence.mean(dim=1)
        return representation, refined_features

    def forward(self, image, landmark_heatmap, return_features=False):
        representation, refined_features = self.forward_features(image, landmark_heatmap)
        logits = self.classifier(representation)
        if return_features:
            return {
                "logits": logits,
                "refined_features": refined_features,
                "se_weights": [block.last_weights for block in self.se_blocks]
            }
        return logits


def build_model(num_classes=7, **kwargs):
    return SEPoster(num_classes=num_classes, **kwargs)
