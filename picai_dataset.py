import os
from glob import glob
from typing import List, Dict, Optional, Callable

from monai.data import DataLoader, Dataset, CacheDataset
from torch.utils.data import Dataset as TorchDataset
from monai.utils import set_determinism

from training_setup.augmentations.train_augment import get_train_transforms
from training_setup.augmentations.test_augment import get_test_transforms
from training_setup.augmentations.unlabeled_augment import (
    get_unlabeled_weak_transforms,
    get_unlabeled_strong_transforms,
)


def _pid_from_path(p: str) -> str:
    return os.path.basename(p).split("_")[0]


def _index_by_pid(paths: List[str]) -> Dict[str, str]:
    return {_pid_from_path(p): p for p in paths}


def _collect_labeled(in_dir: str) -> List[Dict]:
    t2w = sorted(glob(os.path.join(in_dir, "train", "images", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "train", "images", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "train", "images", "hbv", "*.nii.gz")))
    seg = sorted(glob(os.path.join(in_dir, "train", "labels", "seg", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)
    i_seg = _index_by_pid(seg)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv) & set(i_seg))
    samples = []
    for pid in common:
        samples.append(
            {
                "patient_id": pid,
                "t2w": i_t2w[pid],
                "adc": i_adc[pid],
                "hbv": i_hbv[pid],
                "seg": i_seg[pid],
            }
        )
    return samples


def _collect_val(in_dir: str) -> List[Dict]:
    t2w = sorted(glob(os.path.join(in_dir, "valid", "images", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "valid", "images", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "valid", "images", "hbv", "*.nii.gz")))
    seg = sorted(glob(os.path.join(in_dir, "valid", "labels", "seg", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)
    i_seg = _index_by_pid(seg)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv) & set(i_seg))
    samples = []
    for pid in common:
        samples.append(
            {
                "patient_id": pid,
                "t2w": i_t2w[pid],
                "adc": i_adc[pid],
                "hbv": i_hbv[pid],
                "seg": i_seg[pid],
            }
        )
    return samples


def _collect_unlabeled(in_dir: str) -> List[Dict]:
    t2w = sorted(glob(os.path.join(in_dir, "unlab_data", "t2w", "*.nii.gz")))
    adc = sorted(glob(os.path.join(in_dir, "unlab_data", "adc", "*.nii.gz")))
    hbv = sorted(glob(os.path.join(in_dir, "unlab_data", "hbv", "*.nii.gz")))

    i_t2w = _index_by_pid(t2w)
    i_adc = _index_by_pid(adc)
    i_hbv = _index_by_pid(hbv)

    common = sorted(set(i_t2w) & set(i_adc) & set(i_hbv))
    samples = []
    for pid in common:
        samples.append(
            {
                "patient_id": pid,
                "t2w": i_t2w[pid],
                "adc": i_adc[pid],
                "hbv": i_hbv[pid],
            }
        )
    return samples


class DualViewUnlabeledDataset(TorchDataset):
    """
    Returns a dict:
      {
        "weak":   {"image": tensor, "patient_id": str, ...},
        "strong": {"image": tensor, "patient_id": str, ...},
        "patient_id": str
      }
    """

    def __init__(
        self,
        data: List[Dict],
        weak_transform: Callable,
        strong_transform: Callable,
        cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 0,
    ):
        self.data = data
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        base_cls = CacheDataset if cache else Dataset
        base_kwargs = dict(data=data, transform=None)
        if cache:
            base_kwargs.update(cache_rate=cache_rate, num_workers=num_workers)
        self.base_ds = base_cls(**base_kwargs)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        sample = self.base_ds[idx]
        pid = sample.get("patient_id", None)

        weak = self.weak_transform({**sample})
        strong = self.strong_transform({**sample})

        weak["patient_id"] = pid
        strong["patient_id"] = pid

        return {"weak": weak, "strong": strong, "patient_id": pid}


class LabeledPairDataset(TorchDataset):
    """
    Wrap MONAI Dataset (dict) -> (image, label)
    Expects keys: "image", "seg"
    """

    def __init__(self, base_ds, image_key: str = "image", label_key: str = "seg"):
        self.base_ds = base_ds
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        img = sample[self.image_key]
        lab = sample[self.label_key]
        return img, lab


class UnlabeledPairDataset(TorchDataset):
    """
    Wrap DualViewUnlabeledDataset -> (image, dummy_label)
    """

    def __init__(self, base_ds: DualViewUnlabeledDataset, view: str = "weak", image_key: str = "image"):
        assert view in ("weak", "strong")
        self.base_ds = base_ds
        self.view = view
        self.image_key = image_key

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        view_dict = sample[self.view]
        img = view_dict[self.image_key]
        dummy_label = img.new_zeros((1,) + img.shape[1:])
        return img, dummy_label


def prepare_semi_supervised(
    in_dir: str,
    *,
    cache: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 2,
    batch_size_labeled: int = 1,
    batch_size_unlabeled: int = 2,
    batch_size_val: int = 1,
    labeled_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    unlabeled_weak_transform: Optional[Callable] = None,
    unlabeled_strong_transform: Optional[Callable] = None,
    seed: int = 0,
):
    set_determinism(seed=seed)

    labeled_samples = _collect_labeled(in_dir)
    val_samples = _collect_val(in_dir)
    unlabeled_samples = _collect_unlabeled(in_dir)

    labeled_transform = labeled_transform or get_train_transforms()
    val_transform = val_transform or get_test_transforms()
    unlabeled_weak_transform = unlabeled_weak_transform or get_unlabeled_weak_transforms()
    unlabeled_strong_transform = unlabeled_strong_transform or get_unlabeled_strong_transforms()

    if cache:
        labeled_ds = CacheDataset(
            labeled_samples,
            transform=labeled_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        val_ds = CacheDataset(
            val_samples,
            transform=val_transform,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    else:
        labeled_ds = Dataset(labeled_samples, transform=labeled_transform)
        val_ds = Dataset(val_samples, transform=val_transform)

    unlabeled_dual_ds = DualViewUnlabeledDataset(
        data=unlabeled_samples,
        weak_transform=unlabeled_weak_transform,
        strong_transform=unlabeled_strong_transform,
        cache=cache,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    labeled_loader = DataLoader(
        labeled_ds, batch_size=batch_size_labeled, shuffle=True, num_workers=num_workers
    )
    unlabeled_loader = DataLoader(
        unlabeled_dual_ds, batch_size=batch_size_unlabeled, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers
    )

    print(f"[SemiSup] Labeled: {len(labeled_ds)}, Unlabeled: {len(unlabeled_dual_ds)}, Val: {len(val_ds)}")
    return labeled_loader, unlabeled_loader, val_loader


def get_sdcl_dataloaders(
    in_dir: str,
    *,
    cache: bool = False,
    cache_rate: float = 1.0,
    num_workers: int = 2,
    batch_size: int = 2,
    seed: int = 0,
):
    """
    Returns:
      lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, val_loader
    """
    labeled_loader, unlabeled_loader, val_loader = prepare_semi_supervised(
        in_dir,
        cache=cache,
        cache_rate=cache_rate,
        num_workers=num_workers,
        batch_size_labeled=batch_size,
        batch_size_unlabeled=batch_size,
        batch_size_val=1,
        seed=seed,
    )

    labeled_ds_m = labeled_loader.dataset
    unlabeled_ds_dual = unlabeled_loader.dataset
    val_ds_m = val_loader.dataset

    lab_ds_a = LabeledPairDataset(labeled_ds_m, image_key="img", label_key="seg")
    lab_ds_b = LabeledPairDataset(labeled_ds_m, image_key="img", label_key="seg")

    unlab_ds_a = UnlabeledPairDataset(unlabeled_ds_dual, view="weak", image_key="img")
    unlab_ds_b = UnlabeledPairDataset(unlabeled_ds_dual, view="strong", image_key="img")

    lab_loader_a = DataLoader(
        lab_ds_a,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    lab_loader_b = DataLoader(
        lab_ds_b,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    unlab_loader_a = DataLoader(
        unlab_ds_a,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    unlab_loader_b = DataLoader(
        unlab_ds_b,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    val_sdcl_ds = LabeledPairDataset(val_ds_m, image_key="img", label_key="seg")
    val_loader_sdcl = DataLoader(
        val_sdcl_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    print(
        f"[PiCAI-SDCL] Labeled: {len(lab_ds_a)}, Unlabeled: {len(unlab_ds_a)}, Val: {len(val_sdcl_ds)}"
    )

    return lab_loader_a, lab_loader_b, unlab_loader_a, unlab_loader_b, val_loader_sdcl
