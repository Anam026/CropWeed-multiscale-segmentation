"""
Augmentation / Transform pipelines using albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 512, config: dict = None) -> A.Compose:
    cfg = config or {}
    aug = cfg.get("train", {})
    norm = aug.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

    transforms = [
        A.Resize(image_size, image_size),
    ]

    if aug.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))
    if aug.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))
    if aug.get("random_rotation", 0):
        transforms.append(A.Rotate(limit=aug["random_rotation"], p=0.5))

    # Replace RandomResizedCrop with simple random crop — avoids all API version issues
    transforms.append(A.RandomCrop(height=image_size, width=image_size, p=0.0))  # no-op placeholder
    transforms.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5))

    if aug.get("color_jitter"):
        cj = aug["color_jitter"]
        transforms.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=0.1,
                p=0.5,
            )
        )

    if aug.get("gaussian_blur", False):
        transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))

    transforms += [
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def get_val_transforms(image_size: int = 512, config: dict = None) -> A.Compose:
    cfg = config or {}
    aug = cfg.get("val", {})
    norm = aug.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=norm["mean"], std=norm["std"]),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: int = 512, config: dict = None) -> A.Compose:
    return get_val_transforms(image_size, config)