import albumentations as A
from albumentations.pytorch import ToTensorV2


train_transforms = A.Compose([
    A.RandomResizedCrop(height=224, width=224),
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.CoarseDropout(max_holes=12, min_holes=6, max_height=12, max_width=12, min_height=8, min_width=8, p=0.25),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ], p=0.5),
    A.OneOf([
        A.Blur(p=0.1),
        A.GaussianBlur(p=0.1),
        A.MotionBlur(p=0.1),
    ], p=0.1),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
])
