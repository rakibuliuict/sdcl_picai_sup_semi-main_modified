

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

# def get_train_transforms():
#     patch_size = (144, 128, 16)  # target (H, W, D)

#     return Compose([
#         LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
#         EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),

#         # Optional but recommended: unify orientation
#         Orientationd(keys=["t2w", "adc", "hbv", "seg"], axcodes="RAS"),

#         NormalizeIntensityd(keys=["t2w", "adc", "hbv"], nonzero=True, channel_wise=True),

#         # stack modalities into a single 3-channel image
#         ConcatItemsd(keys=["t2w", "adc", "hbv"], name="image", dim=0),

#         #  This is the important part: force SAME spatial size for all
#         ResizeWithPadOrCropd(keys=["image", "seg"], spatial_size=patch_size),

#         # data augmentation
#         RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
#         RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
#         RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
#         RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),

#         EnsureTyped(keys=["image", "seg"]),
#     ])

from monai.transforms import (
    EnsureChannelFirstd, Compose, LoadImaged, 
    Resized, ToTensord, RandGaussianNoised, RandAffined, 
    RandFlipd, ConcatItemsd, NormalizeIntensityd, 
    RandScaleIntensityd, RandShiftIntensityd, Orientationd
)

def get_train_transforms():
    return Compose([
        LoadImaged(keys=["t2w", "adc", "hbv", "seg"]),
        EnsureChannelFirstd(keys=["t2w", "adc", "hbv", "seg"]),
        ConcatItemsd(keys=["t2w", "adc", "hbv"], name="img"),
        Resized(keys=["img", "seg"], spatial_size=(144, 128, 16)),
        RandAffined(keys=["img", "seg"], prob=0.2, translate_range=10.0),
        Orientationd(keys=["img", "seg"], axcodes="RAS"),
        RandFlipd(keys=["img", "seg"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["img", "seg"], spatial_axis=[2], prob=0.5),
        RandGaussianNoised(keys="img", prob=0.4),
        NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="img", factors=0.1, prob=0.4),
        RandShiftIntensityd(keys="img", offsets=0.1, prob=0.4),
        ToTensord(keys=["img", "seg"]),
    ])
