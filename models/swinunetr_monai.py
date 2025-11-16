"""
Thin wrapper around MONAI's SwinUNETR so it can be imported from the local models package.
"""

from monai.networks.nets import SwinUNETR

__all__ = ["SwinUNETR"]
