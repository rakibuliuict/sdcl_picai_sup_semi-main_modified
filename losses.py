import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Combined Dice + CrossEntropy for multi-class segmentation.
    """
    def __init__(self, num_classes: int = 2, dice_weight: float = 1.0, ce_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: [B, C, ...]
        targets: [B, ...] long
        """
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_onehot = to_one_hot(targets.unsqueeze(1), self.num_classes)

        dims = tuple(range(2, probs.dim()))
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(targets_onehot, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def to_one_hot(tensor, nClasses):
    """
    tensor: Nx1x... with integer labels
    """
    assert tensor.max().item() < nClasses
    assert tensor.min().item() >= 0

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size, device=tensor.device)
    return one_hot.scatter_(1, tensor.long(), 1)


def softmax_mse_loss(input_logits, target_probs):
    """
    Takes softmax on inputs and MSE to given probabilities.
    """
    assert input_logits.size() == target_probs.size()
    input_softmax = F.softmax(input_logits, dim=1)
    mse_loss = (input_softmax - target_probs) ** 2
    return mse_loss


voxel_kl_loss = nn.KLDivLoss(reduction="none")


def mix_loss(output, img_l, patch_l, mask, num_classes: int, l_weight: float = 1.0, u_weight: float = 0.5, unlab: bool = False):
    """
    Bidirectional copy-paste segmentation loss:
    - img_l: labels in region where mask==1
    - patch_l: labels in region where mask==0
    mask: [B, ...] float/binary
    """
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l = img_l.long()
    patch_l = patch_l.long()

    output_soft = F.softmax(output, dim=1)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1.0 - mask

    # Dice on masked regions
    img_onehot = to_one_hot(img_l.unsqueeze(1), num_classes)
    patch_onehot = to_one_hot(patch_l.unsqueeze(1), num_classes)

    dims = tuple(range(2, output_soft.dim()))
    prob_img = output_soft
    prob_patch = output_soft

    # image region
    inter_img = torch.sum(prob_img * img_onehot, dim=dims)
    denom_img = torch.sum(prob_img, dim=dims) + torch.sum(img_onehot, dim=dims)
    dice_img = (2.0 * inter_img + 1e-5) / (denom_img + 1e-5)

    # patch region
    inter_patch = torch.sum(prob_patch * patch_onehot, dim=dims)
    denom_patch = torch.sum(prob_patch, dim=dims) + torch.sum(patch_onehot, dim=dims)
    dice_patch = (2.0 * inter_patch + 1e-5) / (denom_patch + 1e-5)

    dice_img = (1.0 - dice_img) * (mask.view(mask.size(0), -1).mean(dim=1))
    dice_patch = (1.0 - dice_patch) * (patch_mask.view(patch_mask.size(0), -1).mean(dim=1))

    loss_dice = image_weight * dice_img.mean() + patch_weight * dice_patch.mean()

    # CE on both regions
    ce_all = CE(output, img_l) * mask + CE(output, patch_l) * patch_mask
    loss_ce = (image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) +
               patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16))

    return loss_dice, loss_ce


def mix_mse_loss(net_output, img_l, patch_l, mask, num_classes: int, l_weight: float = 1.0, u_weight: float = 0.5,
                 unlab: bool = False, diff_mask=None):
    img_l = img_l.long()
    patch_l = patch_l.long()

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1.0 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), num_classes)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), num_classes)

    mse_img = torch.mean(softmax_mse_loss(net_output, img_l_onehot), dim=1) * mask * image_weight
    mse_patch = torch.mean(softmax_mse_loss(net_output, patch_l_onehot), dim=1) * patch_mask * patch_weight
    mse_loss = mse_img + mse_patch

    if diff_mask is None:
        diff_mask = torch.ones_like(mask, dtype=torch.float32)

    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss


def mix_max_kl_loss(net_output, img_l, patch_l, mask, num_classes: int, l_weight: float = 1.0, u_weight: float = 0.5,
                    unlab: bool = False, diff_mask=None):
    img_l = img_l.long()
    patch_l = patch_l.long()

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1.0 - mask

    with torch.no_grad():
        s1 = torch.softmax(net_output, dim=1)
        l1 = torch.argmax(s1, dim=1)
        img_diff_mask = (l1 != img_l)
        patch_diff_mask = (l1 != patch_l)

        uniform_distri = torch.ones_like(net_output) / float(num_classes)

    kl_img = torch.mean(
        voxel_kl_loss(F.log_softmax(net_output, dim=1), uniform_distri), dim=1
    ) * mask * img_diff_mask * image_weight
    kl_patch = torch.mean(
        voxel_kl_loss(F.log_softmax(net_output, dim=1), uniform_distri), dim=1
    ) * patch_mask * patch_diff_mask * patch_weight

    kl_loss = kl_img + kl_patch

    if diff_mask is None:
        diff_mask = torch.ones_like(mask, dtype=torch.float32)

    sum_diff = torch.sum(mask * img_diff_mask * diff_mask) + torch.sum(patch_mask * patch_diff_mask * diff_mask)
    loss = torch.sum(diff_mask * kl_loss) / (sum_diff + 1e-16)
    return loss


def get_XOR_region(mixout1, mixout2):
    s1 = torch.softmax(mixout1, dim=1)
    l1 = torch.argmax(s1, dim=1)
    s2 = torch.softmax(mixout2, dim=1)
    l2 = torch.argmax(s2, dim=1)
    diff_mask = (l1 != l2).float()
    return diff_mask
