"""
Thin wrapper around MONAI's UNETR so it can be imported from the local models package.
"""

from monai.networks.nets import UNETR

__all__ = ["UNETR"]
