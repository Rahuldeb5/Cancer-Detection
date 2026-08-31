import logging
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric, SurfaceDistanceMetric

# NSD tolerance: how close a predicted/GT surface point must be to the other
# surface to count as "matched". No official value for pancreatic tumors ships
# with this dataset -- 2mm is a common default for small/irregular lesions at
# this dataset's ~1x1x1.5mm spacing (see TARGET_SPACING in pants_preprocessing.py).
# Override via `nsd_tolerance_mm` in the yaml config if you want to tune it.
DEFAULT_NSD_TOLERANCE_MM = 2.0


@torch.no_grad()
def validation(net, testLoader, args):
    """
    Returns a dict of validation metrics for one fold:
      - 'dice', 'nsd', 'hd95', 'asd': per-class np.array (voxel/surface overlap quality,
        averaged only over scans where both GT and prediction have foreground -- see below)
      - 'dice_positive_cases': like 'dice', but restricted to scans with real tumor --
        excludes the trivial 1.0 scores negative scans get from an empty-vs-empty match.
        Use THIS for "did this checkpoint improve" comparisons, not 'dice': with ~25% of
        cases scan-level-negative, a model collapsed to predicting nothing everywhere can
        outscore a model that's genuinely trying, on 'dice' alone.
      - 'sensitivity', 'specificity', 'f1', 'auc': scalars, case-level tumor
        DETECTION metrics (does this scan contain a tumor at all), not segmentation
        quality. A case counts as "predicted positive" if any voxel survives the
        0.5 sigmoid threshold; AUC uses the scan's maximum predicted probability
        as the continuous score. This mirrors how R-Super's F1/AUC/Sens/Spec are
        computed (thresholded predicted volume vs. GT lesion presence, probability-based
        AUROC) -- it is a much easier task than voxel-level DSC/NSD, which is why it
        reads far higher; don't compare it against DSC as if they measured the same thing.
    """
    net.eval()

    dice_scores = []
    positive_dice_scores = []  # dice_scores restricted to cases with real tumor -- see below
    nsd_scores = []
    hd95_scores = []
    asd_scores = []
    false_positive_scans = 0
    negative_scans = 0

    gt_present, pred_present, max_prob = [], [], []  # case-level detection bookkeeping

    dice_metric = DiceMetric(include_background=True, reduction='mean')
    nsd_tolerance = getattr(args, 'nsd_tolerance_mm', DEFAULT_NSD_TOLERANCE_MM)
    nsd_metric = SurfaceDiceMetric(class_thresholds=[nsd_tolerance], include_background=True, reduction='mean')
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=True, reduction='mean')
    asd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction='mean')

    # sliding-window inference over full volumes is slow with no other progress signal --
    # without this, a multi-hour validation pass looks identical to a hang
    n_cases = len(testLoader)
    t_start = time.time()
    logging.info(f"Starting validation over {n_cases} cases")

    for i, batch in enumerate(testLoader):
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

        if isinstance(logits, (tuple, list)):
            # aux_loss=True makes the model return [main_out, aux_out]; the aux head is
            # only a deep-supervision training signal, inference/eval uses the main output
            logits = logits[0]

        probs = torch.sigmoid(logits.float())
        pred = (probs > 0.5).float()

        gt_empty = label.sum().item() == 0
        pred_empty = pred.sum().item() == 0

        gt_present.append(0 if gt_empty else 1)
        pred_present.append(0 if pred_empty else 1)
        # fp16/AMP can occasionally produce a NaN activation somewhere in a sliding-window
        # patch (rare, usually self-recovering during training -- see the transient
        # "Loss (nan)" artifact seen in training logs); torch.max propagates a single NaN to
        # the whole reduction, and sklearn's roc_auc_score hard-crashes on NaN input, so one
        # bad voxel out of hundreds of thousands would otherwise take down the entire
        # multi-hour validation pass at the very last step. Treat a NaN voxel as 0 probability.
        max_prob.append(torch.nan_to_num(probs, nan=0.0).max().item())

        if (i + 1) % 10 == 0 or (i + 1) == n_cases:
            elapsed = time.time() - t_start
            rate = elapsed / (i + 1)
            eta = rate * (n_cases - (i + 1))
            logging.info(f"Validation {i+1}/{n_cases} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        if gt_empty:
            # true/false-positive-on-negative-scan: no tumor surface exists, so surface-based
            # metrics (NSD/HD95/ASD) are undefined (MONAI returns nan/inf here) -- score Dice
            # directly and track false positives separately instead of polluting those averages
            negative_scans += 1
            dice_scores.append(1.0 if pred_empty else 0.0)
            if not pred_empty:
                false_positive_scans += 1
            continue

        d = dice_metric(pred, label).item()
        dice_scores.append(d)
        positive_dice_scores.append(d)

        if pred_empty:
            # missed the tumor entirely: Dice already captured this as 0 via the metric
            # above, but surface metrics have no predicted surface to measure against
            continue

        nsd_scores.append(nsd_metric(pred, label).item())
        hd95_scores.append(hd95_metric(pred, label).item())
        asd_scores.append(asd_metric(pred, label).item())

    dice_list = np.array([np.mean(dice_scores)])
    # 'dice' includes trivial 1.0 scores for scan-level-negative cases (empty prediction
    # correctly matching empty ground truth) -- with ~25% of cases negative, a model that
    # has collapsed to predicting nothing everywhere can score dice_list.mean() *higher*
    # than a model that's actually trying, which silently defeats "is this checkpoint
    # better" comparisons. dice_positive_cases only covers cases with real tumor, so a
    # collapsed model scores 0 here regardless of how many negatives it gets "right" --
    # use THIS for checkpoint selection, not 'dice'.
    dice_positive_list = np.array([np.mean(positive_dice_scores) if len(positive_dice_scores) > 0 else np.nan])
    nsd_list = np.array([np.mean(nsd_scores) if len(nsd_scores) > 0 else np.nan])
    hd95_list = np.array([np.mean(hd95_scores) if len(hd95_scores) > 0 else np.nan])
    asd_list = np.array([np.mean(asd_scores) if len(asd_scores) > 0 else np.nan])

    gt_arr, pred_arr = np.array(gt_present), np.array(pred_present)
    tp = int(np.sum((gt_arr == 1) & (pred_arr == 1)))
    fn = int(np.sum((gt_arr == 1) & (pred_arr == 0)))
    fp = int(np.sum((gt_arr == 0) & (pred_arr == 1)))
    tn = int(np.sum((gt_arr == 0) & (pred_arr == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
    auc = roc_auc_score(gt_arr, np.array(max_prob)) if len(np.unique(gt_arr)) > 1 else np.nan

    if negative_scans > 0:
        logging.info(f"False positives on {false_positive_scans}/{negative_scans} true-negative scans")

    return {
        'dice': dice_list,
        'dice_positive_cases': dice_positive_list,
        'nsd': nsd_list,
        'hd95': hd95_list,
        'asd': asd_list,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
    }
