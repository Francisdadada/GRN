import numpy as np

def compute_iou(pred_mask, true_mask, num_classes):
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        pred_mask (numpy.ndarray): Predicted mask, shape [H, W].
        true_mask (numpy.ndarray): Ground truth mask, shape [H, W].
        num_classes (int): Number of classes.

    Returns:
        list: IoU for each class.
    """
    ious = []
    for cls in range(num_classes):
        true_cls = (true_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append(intersection / union)
    return ious

def compute_dice(pred_mask, true_mask, num_classes):
    """
    Compute Dice Coefficient for each class.

    Args:
        pred_mask (numpy.ndarray): Predicted mask, shape [H, W].
        true_mask (numpy.ndarray): Ground truth mask, shape [H, W].
        num_classes (int): Number of classes.

    Returns:
        list: Dice coefficient for each class.
    """
    dices = []
    for cls in range(num_classes):
        true_cls = (true_mask == cls)
        pred_cls = (pred_mask == cls)
        intersection = np.logical_and(true_cls, pred_cls).sum()
        total = pred_cls.sum() + true_cls.sum()
        if total == 0:
            dices.append(float('nan'))  # Avoid division by zero
        else:
            dices.append(2 * intersection / total)
    return dices