from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ToTensord,
    NormalizeIntensityd,
)

# Transforms to be applied on training instances
train_transform = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ]
)

# Transforms to be applied on validation instances
val_transform = Compose(
    [
        EnsureChannelFirstd(keys=["image", "label"], channel_dim=0),
        NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        ToTensord(keys=['image', 'label'])
    ]
)
