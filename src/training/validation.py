import logging

import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric


@torch.no_grad()
def validation(net, testLoader, args):
    net.eval()

    dice_scores = []
    hd_scores = []
    asd_scores = []
    false_positive_scans = 0
    negative_scans = 0

    dice_metric = DiceMetric(include_background=True, reduction='mean')
    hd_metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction='mean')
    asd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction='mean')

    for batch in testLoader:
        img, label = batch['image'], batch['label']
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).float()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            logits = sliding_window_inference(
                inputs=img,
                roi_size=args.spatial_size,
                sw_batch_size=getattr(args, 'sw_batch_size', 4),
                predictor=net,
                overlap=getattr(args, 'sw_overlap', 0.5),
            )

        pred = (torch.sigmoid(logits.float()) > 0.5).float()

        gt_empty = label.sum().item() == 0
        pred_empty = pred.sum().item() == 0

        if gt_empty:
            # true/false-positive-on-negative-scan: no tumor surface exists, so HD/ASD
            # are undefined (MONAI returns nan/inf here) -- score Dice directly and
            # track false positives separately instead of polluting the HD/ASD average.
            negative_scans += 1
            dice_scores.append(1.0 if pred_empty else 0.0)
            if not pred_empty:
                false_positive_scans += 1
            continue

        dice_scores.append(dice_metric(pred, label).item())

        if pred_empty:
            # missed the tumor entirely: Dice already captured this as 0 via the metric
            # above, but HD/ASD have no predicted surface to measure against, so skip them.
            continue

        hd_scores.append(hd_metric(pred, label).item())
        asd_scores.append(asd_metric(pred, label).item())

    dice_list = np.array([np.mean(dice_scores)])
    hd_list = np.array([np.mean(hd_scores) if len(hd_scores) > 0 else np.nan])
    asd_list = np.array([np.mean(asd_scores) if len(asd_scores) > 0 else np.nan])

    if negative_scans > 0:
        logging.info(f"False positives on {false_positive_scans}/{negative_scans} true-negative scans")

    return dice_list, asd_list, hd_list
