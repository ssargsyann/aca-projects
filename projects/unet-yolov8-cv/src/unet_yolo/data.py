"""Index-stable synthetic shapes, with separate deterministic seed namespaces."""

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

SPLITS = {"train": 0, "validation": 1, "test": 2}


class Shapes(Dataset):
    def __init__(self, count, image_size=(64, 64), seed=42, split="train"):
        if count < 1 or min(image_size) < 32 or split not in SPLITS or seed < 0:
            raise ValueError(
                "Positive count, H/W >=32, nonnegative seed and known split required"
            )
        self.count, self.image_size, self.seed, self.split = (
            count,
            tuple(image_size),
            seed,
            split,
        )

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, SPLITS[self.split], index])
        )
        h, w = self.image_size
        base = rng.integers(30, 200, 3)
        gradient = np.linspace(0.5, 1, w)[None, :, None]
        background = np.clip(
            base * gradient + rng.normal(0, 12, (h, w, 3)), 0, 255
        ).astype(np.uint8)
        image = Image.fromarray(background)
        mask = Image.new("L", (w, h))
        draw, target = ImageDraw.Draw(image), ImageDraw.Draw(mask)
        size = int(
            rng.integers(max(2, int(0.15 * min(h, w))), max(3, int(0.35 * min(h, w))))
        )
        cx, cy = (
            int(rng.integers(int(0.15 * w), int(0.85 * w))),
            int(rng.integers(int(0.15 * h), int(0.85 * h))),
        )
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        kind = int(rng.integers(0, 3))
        if kind < 2:
            method = "ellipse" if kind == 0 else "rectangle"
            getattr(draw, method)(
                (cx - size, cy - size, cx + size, cy + size), fill=color
            )
            getattr(target, method)(
                (cx - size, cy - size, cx + size, cy + size), fill=1
            )
        else:
            points = [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
            draw.polygon(points, fill=color)
            target.polygon(points, fill=1)
        return torch.from_numpy(np.array(image)).permute(
            2, 0, 1
        ).float() / 255, torch.from_numpy(np.array(mask)).unsqueeze(0).float()
