# SDCL for PiCAI (3D Prostate MRI)

This repo is a **self-contained implementation** of a simple SDCL-style
(semi-supervised dual-student with discrepancy correction) framework
for the PiCAI dataset with 3D volumes of shape `[3, 144, 128, 16]`.

## Folder structure

- `train_sdcl_picai.py` : main training script
- `picai_dataset.py`    : semi-supervised dataloaders for PiCAI
- `models/`             : 3D UNet and 3D ResUNet
- `training_setup/augmentations/` : MONAI transform pipelines
- `losses.py`           : Dice + CE + SDCL (MSE + KL) losses

## Requirements

Install with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA/CPU
pip install monai[torch] numpy tqdm
```

## Data layout

Expected PiCAI folder structure:

```text
/your/picai_root/
  train/
    images/
      t2w/*.nii.gz
      adc/*.nii.gz
      hbv/*.nii.gz
    labels/
      seg/*.nii.gz        # binary mask, 0 = background, 1 = lesion
  valid/
    images/
      t2w/*.nii.gz
      adc/*.nii.gz
      hbv/*.nii.gz
    labels/
      seg/*.nii.gz
  unlab_data/
    t2w/*.nii.gz
    adc/*.nii.gz
    hbv/*.nii.gz
```

## Run training

```bash
python train_sdcl_picai.py   --root_path /your/picai_root   --gpu 0   --batch_size 2   --max_epoch 300   --num_classes 2
```

Models and logs will be saved into `./runs_sdcl_picai/`.
