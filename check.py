# check.py
import torch
from picai_dataset import prepare_semi_supervised

def main():
    # 👇 change this to your real PiCAI root
    root = r"E:\Dataset\PICAI\ssl_final_data\picai_144_128_16"

    print("\n===== Building loaders with prepare_semi_supervised() =====")
    lab_loader, unlab_loader, val_loader = prepare_semi_supervised(
        root,
        batch_size_labeled=1,
        batch_size_unlabeled=1,
        batch_size_val=1,
    )

    print(f"\nLabeled train cases: {len(lab_loader.dataset)}")
    print(f"Validation cases:    {len(val_loader.dataset)}")
    print(f"Unlabeled cases:     {len(unlab_loader.dataset)}")

    # ---------- LABELED SAMPLE ----------
    print("\n===== One LABELED sample =====")
    lab_sample = next(iter(lab_loader))   # this is a dict, NOT (img, lab)

    print("Labeled sample keys:", lab_sample.keys())
    for k, v in lab_sample.items():
        try:
            # MONAI MetaTensor / Tensor usually has .shape
            print(f"  {k}: shape = {tuple(v.shape)}")
        except AttributeError:
            print(f"  {k}: (no .shape, type={type(v)})")

    # ---------- UNLABELED SAMPLE ----------
    print("\n===== One UNLABELED sample =====")
    unlab_sample = next(iter(unlab_loader))  # dict with 'weak', 'strong', 'patient_id'

    print("Unlabeled sample keys:", unlab_sample.keys())
    weak = unlab_sample["weak"]
    strong = unlab_sample["strong"]

    print("\n  Weak keys:", weak.keys())
    for k, v in weak.items():
        try:
            print(f"    weak[{k}]: shape = {tuple(v.shape)}")
        except AttributeError:
            print(f"    weak[{k}]: (no .shape, type={type(v)})")

    print("\n  Strong keys:", strong.keys())
    for k, v in strong.items():
        try:
            print(f"    strong[{k}]: shape = {tuple(v.shape)}")
        except AttributeError:
            print(f"    strong[{k}]: (no .shape, type={type(v)})")

    # ---------- VALIDATION SAMPLE ----------
    print("\n===== One VALIDATION sample =====")
    val_sample = next(iter(val_loader))
    print("Val sample keys:", val_sample.keys())
    for k, v in val_sample.items():
        try:
            print(f"  {k}: shape = {tuple(v.shape)}")
        except AttributeError:
            print(f"  {k}: (no .shape, type={type(v)})")


if __name__ == "__main__":
    main()
