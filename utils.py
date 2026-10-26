import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
from PIL import Image
import copy
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from torch import einsum

def one_hot2dist(seg: torch.Tensor) -> torch.Tensor:
    assert seg.dim() == 4, "Input must have shape (B, C, H, W)"
    B, C, H, W = seg.shape  # Batch size, number of classes, height, width

    device = seg.device  # Get the device of the input tensor
    res = torch.zeros_like(seg, dtype=torch.float32, device=device)

    for c in range(C):
        posmask = seg[:, c:c+1]  # Foreground mask for class 'c', shape (B, 1, H, W)
        negmask = 1 - posmask     # Background mask for class 'c', shape (B, 1, H, W)

        if posmask.any():
            # Convert to float for distance calculation
            posmask = posmask.float()
            negmask = negmask.float()

            # Ensure dist_pos and dist_neg are on the same device as the input tensor
            dist_pos = distance_transform(negmask.to(device))  # Background distances
            dist_neg = distance_transform(posmask.to(device))  # Foreground distances

            # Calculate the result as the distance map
            res[:, c:c+1] = dist_pos * negmask - (dist_neg - 1) * posmask

    return res


def distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean distance transform for the input binary mask tensor.
    This is a simplified version of distance transform using pixel-wise distances.
    """
    # Ensure mask is float type for distance calculation
    mask = mask.float()

    # Create a grid of pixel coordinates with explicit indexing="ij"
    grid_x, grid_y = torch.meshgrid(torch.arange(mask.shape[2], device=mask.device),
                                     torch.arange(mask.shape[3], device=mask.device),
                                     indexing="ij")
    grid_x = grid_x.float().unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
    grid_y = grid_y.float().unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
    grid_x = grid_x.expand(mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])
    grid_y = grid_y.expand(mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3])

    # Compute squared Euclidean distance to the nearest foreground pixel
    dist = (grid_x - grid_x.transpose(2, 3))**2 + (grid_y - grid_y.transpose(2, 3))**2
    dist = dist.sqrt()

    return dist

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res



def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1]  # 这里忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3

    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multiplied.mean()

        return loss


def one_hot_targets(labels: Tensor, logits: Tensor) -> Tensor:
    """Match class-index masks to the raw class logits for per-channel BCE.

    The current binary model keeps its two output channels (background and
    foreground). BCE supervises these channels independently; Dice and argmax
    inference retain their existing behavior.
    """
    if labels.ndim != 3 or labels.dtype != torch.long:
        raise ValueError("BCE requires integer class-index labels [N, H, W]")
    if (
        logits.ndim != 4
        or logits.shape[0] != labels.shape[0]
        or logits.shape[2:] != labels.shape[1:]
    ):
        raise ValueError("BCE logits [N, C, H, W] must match label dimensions")
    return F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(
        device=logits.device, dtype=logits.dtype
    )


class EdgePreservingLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        # 使用 register_buffer 注册 Sobel 滤波器，这样会自动随模型移动到相同的设备上
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - 模型输出的 logits
        target: (B, H, W) - 真实标签，类别索引
        """
        # 将 logits 转为概率，并取最大类别作为单通道预测
        pred_probs = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_probs, dim=1, keepdim=True).float()

        # 真实标签转换为单通道 float 类型
        target = target.unsqueeze(1).float()

        # 计算预测边缘
        pred_edge_x = F.conv2d(pred_class, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_class, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)

        # 计算真实边缘（不反向传播）
        with torch.no_grad():
            target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
            target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
            target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)

        # 边缘损失（L1 损失）
        edge_loss = F.l1_loss(pred_edge, target_edge)

        # 主分割损失：组合 BCEWithLogitsLoss 与 DiceLoss
        labels = target.squeeze(1).long()
        main_loss = 0.3 * F.binary_cross_entropy_with_logits(
            pred, one_hot_targets(labels, pred)
        ) + 0.7 * DiceLoss(2)(pred, labels)

        return self.alpha * main_loss + (1 - self.alpha) * edge_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'Predict {inputs.size()} & Target {target.size()} shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_iou(pred, gt, smooth=1e-5):
    """
    Calculate Intersection over Union (IoU).

    Args:
        pred (numpy.ndarray): Predicted binary mask (0 or 1).
        gt (numpy.ndarray): Ground truth binary mask (0 or 1).
        smooth (float): A small value to avoid division by zero.

    Returns:
        float: IoU value.
    """
    pred = np.asarray(pred).astype(np.bool_)
    gt = np.asarray(gt).astype(np.bool_)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou



def calculate_metric_percase(pred, gt):
    """
    Calculate metrics (Dice, HD95, IoU, and HD) for a single prediction and ground truth pair.

    Args:
        pred (numpy.ndarray): Predicted binary mask (0 or 1).
        gt   (numpy.ndarray): Ground truth binary mask (0 or 1).

    Returns:
        tuple: (dice, hd95, iou, hd)
    """
    # Ensure binary masks
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt   > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice  = metric.binary.dc(pred, gt)
        hd95  = metric.binary.hd95(pred, gt)
        hd    = metric.binary.hd(pred, gt)      # 新增：最大 Hausdorff Distance
        iou   = calculate_iou(pred, gt)
        return dice, hd95, iou, hd
    elif pred.sum() > 0 and gt.sum() == 0:
        # False positive: prediction contains foreground but ground truth does not.
        return 0.0, 0.0, 0.0, 0.0
    else:
        # 真值无目标：视为完美（无分割需求）
        return 0.0, 0.0, 0.0, 0.0


def test_single_volume(image, label, net, classes, patch_size=[512, 512], test_save_path=None, case=None, z_spacing=1):
    """
    Test a single volume and compute metrics.

    Args:
        image (torch.Tensor): Input image tensor.
        label (torch.Tensor): Ground truth label tensor.
        net (torch.nn.Module): Trained segmentation network.
        classes (int): Number of classes.
        patch_size (list): Input size for the network.
        test_save_path (str): Path to save predictions.
        case (str): Case name.
        z_spacing (float): Z-axis spacing.

    Returns:
        list: Metrics for all classes.
    """
    # Squeeze batch dimension and convert to numpy
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    # Check input shape and ensure it is 3D (Channels, Height, Width)
    if len(image.shape) == 3:  # If RGB image [H, W, C]
        image = image.transpose(2, 0, 1)  # Convert to [C, H, W]
    elif len(image.shape) == 2:  # If grayscale image [H, W]
        image = image[np.newaxis, :, :]  # Add channel dimension [1, H, W]

    _, x, y = image.shape

    # Resize image if it doesn't match the patch size
    if x != patch_size[0] or y != patch_size[1]:
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
        image = F.interpolate(image, size=patch_size, mode="bilinear", align_corners=False).squeeze(0).numpy()

    # Convert image to PyTorch tensor and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)  # Shape: [1, C, H, W]

    # Perform model inference
    net.eval()
    with torch.no_grad():
        output = net(input_tensor)  # Model output logits
        out = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0)  # Shape: [H, W]
        out = out.cpu().detach().numpy()

    # Resize prediction back to original size if resized earlier
    if x != patch_size[0] or y != patch_size[1]:
        prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    else:
        prediction = out

    # Calculate metrics for each class
    metric_list = []
    for i in range(1, classes):  # Skip background (class 0)
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # Save predictions as an RGB image
    if test_save_path is not None:
        # Create RGB channels for visualization
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)

        # Set all predicted areas to white
        a1[a1 == 1] = 255
        a2[a2 == 1] = 255
        a3[a3 == 1] = 255

        # Convert to PIL Image and save
        a1 = Image.fromarray(np.uint8(a1)).convert('L')
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction_img = Image.merge('RGB', [a1, a2, a3])
        prediction_img.save(test_save_path + '/' + case + '.png')

    return metric_list
