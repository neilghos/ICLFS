from __future__ import annotations

from dataclasses import dataclass

import torch


def _sample_keep_mask(n_patients: int, keep_ratio: float) -> torch.Tensor:
    keep_count = max(1, min(n_patients, int(round(n_patients * keep_ratio))))
    perm = torch.randperm(n_patients)
    mask = torch.zeros(n_patients, dtype=torch.float32)
    mask[perm[:keep_count]] = 1.0
    return mask


def _sample_subset_pair(
    n_patients: int,
    keep_ratio: float,
    overlap_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_count = max(1, min(n_patients, int(round(n_patients * keep_ratio))))
    overlap_count = max(0, min(keep_count, int(round(keep_count * overlap_ratio))))

    perm = torch.randperm(n_patients)
    overlap_idx = perm[:overlap_count]
    remaining = perm[overlap_count:]

    half = max(0, keep_count - overlap_count)
    first_unique = remaining[:half]
    second_unique = remaining[half : half + half]

    mask1 = torch.zeros(n_patients, dtype=torch.float32)
    mask2 = torch.zeros(n_patients, dtype=torch.float32)
    mask1[overlap_idx] = 1.0
    mask2[overlap_idx] = 1.0
    mask1[first_unique] = 1.0
    mask2[second_unique] = 1.0

    # If the second side runs short due to rounding, top it up from unused indices.
    if int(mask2.sum().item()) < keep_count:
        used = (mask1 + mask2) > 0
        unused = (~used).nonzero(as_tuple=False).flatten()
        needed = keep_count - int(mask2.sum().item())
        mask2[unused[:needed]] = 1.0

    return mask1, mask2


@dataclass
class RandomMaskAugmentor:
    """
    Default two-view augmentor for inverted feature profiles.

    The current task remains unchanged: for each feature profile across
    patients, generate two independently masked views. All future masking/task
    redesigns should happen in this module so the training path stays agnostic.
    """

    n_patients: int
    mask_prob: float = 0.0

    def _sample_mask(self) -> torch.Tensor:
        return (torch.rand(self.n_patients) > self.mask_prob).float()

    def __call__(self, feat_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask1 = self._sample_mask()
        mask2 = self._sample_mask()
        return feat_vector * mask1, feat_vector * mask2


@dataclass
class MultiViewMaskLibrary:
    """
    Library of candidate positive-view maskers for future contrastive-task
    redesigns. This does not yet change the main training path; it only exposes
    named mask families we can sample from later.
    """

    n_patients: int
    light_keep_ratio: float = 0.90
    heavy_keep_ratio: float = 0.60
    subset_keep_ratio: float = 0.50
    complementary_overlap_ratio: float = 0.10

    def light_mask(self, feat_vector: torch.Tensor) -> torch.Tensor:
        mask = _sample_keep_mask(self.n_patients, keep_ratio=self.light_keep_ratio)
        return feat_vector * mask

    def heavy_mask(self, feat_vector: torch.Tensor) -> torch.Tensor:
        mask = _sample_keep_mask(self.n_patients, keep_ratio=self.heavy_keep_ratio)
        return feat_vector * mask

    def subset_mask(self, feat_vector: torch.Tensor) -> torch.Tensor:
        mask = _sample_keep_mask(self.n_patients, keep_ratio=self.subset_keep_ratio)
        return feat_vector * mask

    def complementary_subset_pair(
        self,
        feat_vector: torch.Tensor,
        overlap_ratio: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask1, mask2 = _sample_subset_pair(
            self.n_patients,
            keep_ratio=self.subset_keep_ratio,
            overlap_ratio=overlap_ratio,
        )
        return feat_vector * mask1, feat_vector * mask2

    def build_four_positive_views(self, feat_vector: torch.Tensor) -> list[torch.Tensor]:
        comp_a, comp_b = self.complementary_subset_pair(
            feat_vector,
            overlap_ratio=self.complementary_overlap_ratio,
        )
        return [
            self.light_mask(feat_vector),
            self.heavy_mask(feat_vector),
            comp_a,
            comp_b,
        ]


@dataclass
class FourViewMaskAugmentor:
    """
    Anchor + 4 positive masked views:
    - original feature profile
    - light mask
    - heavy mask
    - complementary subset A
    - complementary subset B
    """

    n_patients: int
    light_keep_ratio: float = 0.90
    heavy_keep_ratio: float = 0.60
    subset_keep_ratio: float = 0.50
    complementary_overlap_ratio: float = 0.10

    def __post_init__(self):
        self.library = MultiViewMaskLibrary(
            n_patients=self.n_patients,
            light_keep_ratio=self.light_keep_ratio,
            heavy_keep_ratio=self.heavy_keep_ratio,
            subset_keep_ratio=self.subset_keep_ratio,
            complementary_overlap_ratio=self.complementary_overlap_ratio,
        )

    def __call__(self, feat_vector: torch.Tensor):
        views = self.library.build_four_positive_views(feat_vector)
        return (feat_vector, *views)


def build_augmentor(
    *,
    n_patients: int,
    mask_prob: float = 0.0,
    strategy: str = "four_view_mask",
    config: dict | None = None,
):
    config = config or {}
    if strategy == "random_mask":
        random_mask_cfg = config.get("random_mask", {})
        return RandomMaskAugmentor(
            n_patients=n_patients,
            mask_prob=random_mask_cfg.get("mask_prob", mask_prob),
        )
    if strategy == "four_view_mask":
        four_view_cfg = config.get("four_view_mask", {})
        return FourViewMaskAugmentor(
            n_patients=n_patients,
            light_keep_ratio=four_view_cfg.get("light_keep_ratio", 0.90),
            heavy_keep_ratio=four_view_cfg.get("heavy_keep_ratio", 0.60),
            subset_keep_ratio=four_view_cfg.get("subset_keep_ratio", 0.50),
            complementary_overlap_ratio=four_view_cfg.get("complementary_overlap_ratio", 0.10),
        )
    raise ValueError(f"Unknown augmentor strategy '{strategy}'")
