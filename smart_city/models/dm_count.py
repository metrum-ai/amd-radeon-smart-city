# Created by Metrum AI for AMD

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

VGG19_CFG = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, 256, "M",
    512, 512, 512, 512, "M",
    512, 512, 512, 512,
]


def _make_vgg_layers(cfg: list, batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG19DensityModel(nn.Module):
    """VGG19-based density estimation matching DM-Count architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.features = _make_vgg_layers(VGG19_CFG)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view(B, -1).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


class DMCountEstimator:
    """DM-Count density estimation wrapper."""

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.weights_path = weights_path
        self.device = torch.device(device)
        self._model: Optional[VGG19DensityModel] = None
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        logger.info("Loading DM-Count model on %s", self.device)
        self._model = VGG19DensityModel()

        if self.weights_path:
            state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Loaded DM-Count weights from %s", self.weights_path)
        else:
            logger.warning(
                "No DM-Count weights provided — random init. "
                "Place weights at models/pretrained/DM-Count_pretrained_models/model_sh_B.pth"
            )

        self._model.to(self.device)
        self._model.half()
        self._model.eval()
        logger.info("DM-Count model ready (FP16) on %s", self.device)

    def infer_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, float]]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        batch_tensors = torch.stack(
            [self._transform(f) for f in frames]
        ).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            density_tensors, _ = self._model(batch_tensors)

        results = []
        for i in range(len(frames)):
            density_map = density_tensors[i, 0].cpu().numpy()
            count = float(density_map.sum())
            results.append((density_map, count))
        return results

    @staticmethod
    def make_heatmap(density_map: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Convert density map to a JET colormap BGR heatmap resized to target_size (w, h)."""
        dmin, dmax = density_map.min(), density_map.max()
        if dmax - dmin > 1e-8:
            normalized = (density_map - dmin) / (dmax - dmin)
        else:
            normalized = np.zeros_like(density_map)
        gray = (normalized * 255).astype(np.uint8)
        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap(resized, cv2.COLORMAP_JET)
