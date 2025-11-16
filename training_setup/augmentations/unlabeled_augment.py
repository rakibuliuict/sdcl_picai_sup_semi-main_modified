


# def get_unlabeled_strong_transforms(patch_size: Sequence[int] = (144, 128, 16)):
#     """
#     Strong augmentation for unlabeled data.
#     """
#     return Compose(
#         [
#             LoadImaged(keys=["t2w", "adc", "hbv"]),
#             EnsureChannelFirstd(keys=["t2w", "adc", "hbv"]),
#             NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
#             ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),
#             RandSpatialCropd(keys=["image"], roi_size=patch_size, random_center=True, random_size=False),
#             RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
#             RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
#             RandRotate90d(keys=["image"], prob=0.5, max_k=3),
#             RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.02),
#             RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.8)),
#             EnsureTyped(keys=["image"]),
#         ]
#     )

# from monai.transforms import (
#     Compose,
#     LoadImaged,
#     EnsureChannelFirstd,
#     NormalizeIntensityd,
#     ConcatItemsd,
#     Orientationd,
#     ResizeWithPadOrCropd,
#     RandFlipd,
#     RandRotate90d,
#     RandGaussianNoised,
#     RandAdjustContrastd,
#     EnsureTyped,
# )

# def get_unlabeled_weak_transforms():
#     patch_size = (144, 128, 16)

#     return Compose([
#         LoadImaged(keys=["t2w", "adc", "hbv"]),
#         EnsureChannelFirstd(keys=["t2w", "adc", "hbv"]),
#         Orientationd(keys=["t2w", "adc", "hbv"], axcodes="RAS"),
#         NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
#         ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),

#         ResizeWithPadOrCropd(keys=["image"], spatial_size=patch_size),

#         RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
#         RandRotate90d(keys=["image"], prob=0.5, max_k=3),
#         EnsureTyped(keys=["image"]),
#     ])


# def get_unlabeled_strong_transforms():
#     patch_size = (144, 128, 16)

#     return Compose([
#         LoadImaged(keys=["t2w", "adc", "hbv"]),
#         EnsureChannelFirstd(keys=["t2w", "adc", "hbv"]),
#         Orientationd(keys=["t2w", "adc", "hbv"], axcodes="RAS"),
#         NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),
#         ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),

#         ResizeWithPadOrCropd(keys=["image"], spatial_size=patch_size),

#         RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
#         RandRotate90d(keys=["image"], prob=0.5, max_k=3),
#         RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.02),
#         RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 1.8)),
#         EnsureTyped(keys=["image"]),
#     ])

# src/training_setup/augmentations/unlabeled_augment.py
from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged,
    Resized, ToTensord, RandGaussianNoised, RandAffined,
    RandFlipd, ConcatItemsd, NormalizeIntensityd,
    RandScaleIntensityd, RandShiftIntensityd, Orientationd
)

IMG_KEYS = ["t2w", "adc", "hbv"]

def get_unlabeled_weak_transforms():
    """
    Weak (light) transforms for unlabeled data.
    - Minimal augmentations, just reorientation, normalization, etc.
    """
    return Compose([
        LoadImaged(keys=IMG_KEYS),
        EnsureChannelFirstd(keys=IMG_KEYS),
        ConcatItemsd(keys=IMG_KEYS, name="img"),
        Resized(keys=["img"], spatial_size=(144, 128, 16)),
        Orientationd(keys=["img"], axcodes="RAS"),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        ToTensord(keys=["img"]),
    ])


def get_unlabeled_strong_transforms():
    """
    Strong transforms for unlabeled data.
    - Adds noise, affine jitter, intensity scaling/shifting, and random flips.
    """
    return Compose([
        LoadImaged(keys=IMG_KEYS),
        EnsureChannelFirstd(keys=IMG_KEYS),
        ConcatItemsd(keys=IMG_KEYS, name="img"),
        Resized(keys=["img"], spatial_size=(144, 128, 16)),
        RandAffined(keys=["img"], prob=0.2, translate_range=10.0),
        Orientationd(keys=["img"], axcodes="RAS"),
        RandFlipd(keys=["img"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["img"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["img"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="img", prob=0.4),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
        RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
        ToTensord(keys=["img"]),
    ])
