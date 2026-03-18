"""DAOD-specific image-view transforms for object detection analysis and adaptation."""

from __future__ import annotations

import random
from typing import Any, Callable

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


def build_weak_view_transform() -> Callable[[Image.Image], Image.Image]:
    """Build the weak target-view transform used for detection self-training.

    The weak view only applies a random horizontal flip. This keeps the view
    simple enough for reliable pseudo-label generation while still exposing the
    model to a light appearance/layout change.
    """

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )


def build_strong_view_transform() -> Callable[[Image.Image], Image.Image]:
    """Build the strong target-view transform used for detection self-training.

    The strong view is photometric only: color jitter, grayscale, and Gaussian
    blur. This avoids introducing extra box-remapping complexity from geometric
    transforms while still producing a noticeably harder view.
    """

    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ]
    )


def make_weak_view(
    image: Image.Image,
    *,
    rng: random.Random | None = None,
    flip_prob: float = 0.5,
) -> tuple[Image.Image, dict[str, Any]]:
    """Create one weak view and return the applied view metadata.

    Inputs:
    - `image`: a PIL RGB image in original-image coordinates
    - `rng`: optional random generator for reproducible notebook analysis
    - `flip_prob`: probability of horizontal flip

    Output:
    - transformed PIL image
    - metadata dict describing the applied geometric transform

    Assumption:
    - the weak view uses only horizontal flip, so the only geometry metadata we
      need to preserve is whether a flip happened and the image width.
    """

    rng = rng or random
    did_hflip = bool(rng.random() < flip_prob)
    if did_hflip:
        image = F.hflip(image)
    return image, {"hflip": did_hflip, "width": int(image.width)}


def make_strong_view(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Create one strong view and return the applied view metadata.

    Inputs:
    - `image`: a PIL RGB image in original-image coordinates

    Output:
    - transformed PIL image
    - metadata dict describing the applied transform

    Assumption:
    - the strong view is photometric only, so it does not change the image
      coordinate system and therefore needs no box remapping metadata.
    """

    return build_strong_view_transform()(image), {"hflip": False, "width": int(image.width)}


def map_boxes_to_original_view(
    boxes_xyxy: list[list[float]],
    view_meta: dict[str, Any],
) -> list[list[float]]:
    """Map boxes from a transformed weak/strong view back to original coordinates.

    Inputs:
    - `boxes_xyxy`: list of `[x0, y0, x1, y1]` boxes in the transformed view
    - `view_meta`: metadata returned by `make_weak_view` or `make_strong_view`

    Output:
    - list of `[x0, y0, x1, y1]` boxes in original-image coordinates

    Assumption:
    - only horizontal flip changes geometry in the current DAOD view policy.
    """

    if not view_meta.get("hflip", False):
        return [list(box) for box in boxes_xyxy]

    width = float(view_meta["width"])
    mapped: list[list[float]] = []
    for x0, y0, x1, y1 in boxes_xyxy:
        mapped.append([width - float(x1), float(y0), width - float(x0), float(y1)])
    return mapped
