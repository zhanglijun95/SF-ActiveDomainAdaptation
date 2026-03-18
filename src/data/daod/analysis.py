"""Small DAOD target-analysis helpers used by the target-val notebook.

The goal of this module is to keep the notebook readable while avoiding a large
analysis pipeline. The functions here are intentionally simple and explicit:
they operate on one image at a time and return plain Python dict/list outputs
that are easy to inspect in a notebook.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two boxes in `[x0, y0, x1, y1]` format.

    Inputs:
    - `box_a`, `box_b`: boxes in absolute pixel coordinates

    Output:
    - scalar IoU in `[0, 1]`

    Coordinate convention:
    - boxes are expected in the original image coordinate system unless the
      caller explicitly works in another shared coordinate frame
    """

    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def instances_to_prediction_rows(
    instances: Any,
    thing_classes: tuple[str, ...] | list[str],
    *,
    score_thresh: float,
) -> list[dict[str, Any]]:
    """Convert detectron2 `Instances` predictions into plain row dicts.

    Inputs:
    - `instances`: detectron2 prediction output with `pred_boxes`, `scores`,
      and `pred_classes`
    - `thing_classes`: class-name lookup for contiguous category ids
    - `score_thresh`: predictions below this score are dropped

    Output:
    - a list of rows, one per retained prediction, each with:
      `bbox`, `score`, `category_id`, and `category_name`

    Coordinate convention:
    - returned boxes stay in the coordinate system used by the incoming
      `instances` object; callers are responsible for remapping boxes if a view
      transform changed geometry before inference
    """

    if hasattr(instances, "to"):
        instances = instances.to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    rows: list[dict[str, Any]] = []
    for box, score, category_id in zip(boxes, scores, classes):
        score = float(score)
        if score < score_thresh:
            continue
        category_id = int(category_id)
        rows.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "score": score,
                "category_id": category_id,
                "category_name": thing_classes[category_id],
            }
        )
    return rows


def greedy_match_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
    class_aware: bool,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Greedily match two detection sets by highest IoU.

    Inputs:
    - `left_rows`, `right_rows`: prediction rows with `bbox` and `category_id`
    - `iou_thresh`: minimum IoU required for a match
    - `class_aware`: if true, only same-class pairs may match

    Output:
    - list of matched pair dicts with indices and IoU
    - unmatched indices from `left_rows`
    - unmatched indices from `right_rows`

    Assumptions:
    - matching is intentionally simple and explainable
    - greedy highest-IoU matching is good enough for notebook-level analysis
      even though it is not the only possible assignment rule
    - both input row lists must already use the same coordinate system
    """

    candidates: list[tuple[float, int, int]] = []
    for left_idx, left in enumerate(left_rows):
        for right_idx, right in enumerate(right_rows):
            if class_aware and left["category_id"] != right["category_id"]:
                continue
            iou = xyxy_iou(left["bbox"], right["bbox"])
            if iou >= iou_thresh:
                candidates.append((iou, left_idx, right_idx))

    candidates.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, left_idx, right_idx in candidates:
        if left_idx in used_left or right_idx in used_right:
            continue
        used_left.add(left_idx)
        used_right.add(right_idx)
        matches.append(
            {
                "left_idx": left_idx,
                "right_idx": right_idx,
                "iou": float(iou),
                "left": left_rows[left_idx],
                "right": right_rows[right_idx],
            }
        )

    unmatched_left = [idx for idx in range(len(left_rows)) if idx not in used_left]
    unmatched_right = [idx for idx in range(len(right_rows)) if idx not in used_right]
    return matches, unmatched_left, unmatched_right


def match_predictions_to_gt(
    gt_annotations: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    iou_thresh: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Match predictions to GT with class-aware greedy IoU matching.

    Inputs:
    - `gt_annotations`: GT annotations with `bbox` and `category_id`
    - `pred_rows`: prediction rows with `bbox`, `score`, and `category_id`
    - `iou_thresh`: minimum IoU required for a GT/prediction match

    Output:
    - `tp_rows`: matched GT/prediction pairs
    - `fp_rows`: predictions left unmatched after GT matching
    - `fn_rows`: GT objects left unmatched after prediction matching

    Score meaning:
    - this produces the simple TP / FP / FN counts used to define notebook
      ground-truth difficulty scores at the image level.
    """

    gt_rows = [
        {
            "bbox": [float(v) for v in ann["bbox"]],
            "category_id": int(ann["category_id"]),
            "area": float(ann.get("area", 0.0)),
        }
        for ann in gt_annotations
    ]

    matches, unmatched_gt, unmatched_pred = greedy_match_rows(
        gt_rows,
        pred_rows,
        iou_thresh=iou_thresh,
        class_aware=True,
    )

    tp_rows = [
        {
            "gt_idx": match["left_idx"],
            "pred_idx": match["right_idx"],
            "iou": match["iou"],
            "gt": match["left"],
            "pred": match["right"],
        }
        for match in matches
    ]
    fp_rows = [pred_rows[idx] for idx in unmatched_pred]
    fn_rows = [gt_rows[idx] for idx in unmatched_gt]
    return tp_rows, fp_rows, fn_rows


def classify_detection_errors(
    gt_annotations: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
    *,
    match_iou_thresh: float,
    nearby_iou_thresh: float,
) -> dict[str, Any]:
    """Split GT/prediction errors into more interpretable detection modes.

    Inputs:
    - `gt_annotations`: GT annotation dicts with `bbox` and `category_id`
    - `pred_rows`: prediction dicts with `bbox`, `score`, and `category_id`
    - `match_iou_thresh`: IoU threshold for a correct GT/prediction match
    - `nearby_iou_thresh`: lower IoU threshold used to define a nearby object

    Output:
    - a dict containing typed GT and FP error counts plus a continuous
      localization error summary

    Error modes:
    - `miss_count`: GT object has no nearby prediction at all
    - `wrong_class_count`: GT object has a nearby prediction, but not with the
      correct class
    - `localization_error_count`: GT object has a nearby same-class prediction
      that does not reach the correct-match IoU threshold
    - `background_fp_count`: unmatched prediction is not near any GT object
    - `duplicate_fp_count`: unmatched prediction is near a same-class GT object,
      so it is more plausibly a duplicate / extra detection
    - `localization_error_mean`: mean `(1 - IoU)` over localization-error GTs

    Assumptions:
    - this function keeps the aspects separate instead of collapsing them into
      one scalar difficulty target
    """

    gt_rows = [
        {
            "bbox": [float(v) for v in ann["bbox"]],
            "category_id": int(ann["category_id"]),
        }
        for ann in gt_annotations
    ]
    _, unmatched_gt_indices, unmatched_pred_indices = greedy_match_rows(
        gt_rows,
        pred_rows,
        iou_thresh=match_iou_thresh,
        class_aware=True,
    )

    miss_count = 0
    wrong_class_count = 0
    localization_error_count = 0
    localization_error_values: list[float] = []
    for gt_index in unmatched_gt_indices:
        gt_row = gt_rows[gt_index]
        nearby_iou = 0.0
        same_class_nearby_iou = 0.0
        for pred_row in pred_rows:
            iou = xyxy_iou(gt_row["bbox"], pred_row["bbox"])
            nearby_iou = max(nearby_iou, iou)
            if pred_row["category_id"] == gt_row["category_id"]:
                same_class_nearby_iou = max(same_class_nearby_iou, iou)

        if same_class_nearby_iou >= nearby_iou_thresh:
            localization_error_count += 1
            localization_error_values.append(1.0 - same_class_nearby_iou)
        elif nearby_iou >= nearby_iou_thresh:
            wrong_class_count += 1
        else:
            miss_count += 1

    background_fp_count = 0
    duplicate_fp_count = 0
    for pred_index in unmatched_pred_indices:
        pred_row = pred_rows[pred_index]
        max_iou_any_gt = 0.0
        max_iou_same_class_gt = 0.0
        for gt_row in gt_rows:
            iou = xyxy_iou(pred_row["bbox"], gt_row["bbox"])
            max_iou_any_gt = max(max_iou_any_gt, iou)
            if pred_row["category_id"] == gt_row["category_id"]:
                max_iou_same_class_gt = max(max_iou_same_class_gt, iou)
        if max_iou_any_gt < nearby_iou_thresh:
            background_fp_count += 1
        elif max_iou_same_class_gt >= nearby_iou_thresh:
            duplicate_fp_count += 1
        else:
            background_fp_count += 1

    return {
        "miss_count": float(miss_count),
        "wrong_class_count": float(wrong_class_count),
        "localization_error_count": float(localization_error_count),
        "localization_error_mean": float(np.mean(localization_error_values)) if localization_error_values else 0.0,
        "localization_error_sum": float(np.sum(localization_error_values)) if localization_error_values else 0.0,
        "background_fp_count": float(background_fp_count),
        "duplicate_fp_count": float(duplicate_fp_count),
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    denom = np.sum(exp_x)
    return exp_x / denom if denom > 0 else np.zeros_like(x)


def compute_logit_proxy_summary(selected_detections: list[dict[str, Any]]) -> dict[str, float]:
    """Compute image-level classification uncertainty proxies from detection logits.

    Inputs:
    - `selected_detections`: per-detection rows from the adapter raw-output path
      containing `class_logits` and `selected_logit`

    Output:
    - a flat dict of image-level logit uncertainty summaries

    Score meanings:
    - entropy proxies: larger means more class ambiguity per selected detection
    - margin proxies: smaller top-1 vs top-2 separation means more ambiguity
    - selected-logit proxies: lower selected logits mean weaker class evidence
    """

    if not selected_detections:
        return {
            "proxy_logit_entropy_mean": 0.0,
            "proxy_logit_entropy_high_frac": 0.0,
            "proxy_logit_margin_mean": 0.0,
            "proxy_logit_low_margin_frac": 0.0,
            "proxy_selected_logit_mean": 0.0,
        }

    entropies = []
    margins = []
    selected_logits = []
    for det in selected_detections:
        logits = det["class_logits"].detach().cpu().numpy().astype(float)
        probs = _softmax(logits)
        probs = np.clip(probs, 1e-12, 1.0)
        entropies.append(float(-(probs * np.log(probs)).sum() / np.log(len(probs))))
        top2 = np.sort(probs)[-2:]
        margins.append(float(top2[-1] - top2[-2]))
        selected_logits.append(float(det["selected_logit"]))

    entropies_np = np.asarray(entropies)
    margins_np = np.asarray(margins)
    selected_logits_np = np.asarray(selected_logits)
    return {
        "proxy_logit_entropy_mean": float(entropies_np.mean()),
        "proxy_logit_entropy_high_frac": float((entropies_np > 0.5).mean()),
        "proxy_logit_margin_mean": float(margins_np.mean()),
        "proxy_logit_low_margin_frac": float((margins_np < 0.2).mean()),
        "proxy_selected_logit_mean": float(selected_logits_np.mean()),
    }


def compute_decoder_proxy_summary(selected_detections: list[dict[str, Any]]) -> dict[str, float]:
    """Compute image-level decoder instability proxies from DINO aux outputs.

    Inputs:
    - `selected_detections`: per-detection rows from the adapter raw-output path
      containing `aux_selected_logits`, `aux_class_logits`, and `aux_bbox_xyxy`

    Output:
    - a flat dict of image-level decoder-instability summaries

    Score meanings:
    - class-instability proxies: how much selected-query logits or top classes
      change across decoder layers
    - box-instability proxies: how much selected-query boxes drift across
      decoder layers in original image coordinates
    """

    detections_with_aux = [det for det in selected_detections if "aux_selected_logits" in det]
    if not detections_with_aux:
        return {
            "proxy_decoder_logit_std_mean": 0.0,
            "proxy_decoder_top_class_flip_frac": 0.0,
            "proxy_decoder_box_iou_gap_mean": 0.0,
            "proxy_decoder_box_center_shift_mean": 0.0,
        }

    logit_stds = []
    class_flip_flags = []
    box_iou_gaps = []
    center_shifts = []
    for det in detections_with_aux:
        aux_selected_logits = np.asarray(det["aux_selected_logits"], dtype=float)
        logit_stds.append(float(aux_selected_logits.std()))

        aux_top_classes = []
        for aux_logits in det["aux_class_logits"]:
            aux_logits_np = aux_logits.detach().cpu().numpy()
            aux_top_classes.append(int(np.argmax(aux_logits_np)))
        class_flip_flags.append(float(len(set(aux_top_classes)) > 1))

        final_box = det["bbox_xyxy"].detach().cpu().numpy().astype(float).tolist()
        aux_boxes = [aux_box.detach().cpu().numpy().astype(float).tolist() for aux_box in det["aux_bbox_xyxy"]]
        aux_ious = [xyxy_iou(aux_box, final_box) for aux_box in aux_boxes]
        box_iou_gaps.append(float(np.mean([1.0 - iou for iou in aux_ious])))

        fx0, fy0, fx1, fy1 = final_box
        final_center = np.asarray([(fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0], dtype=float)
        diag = max(np.hypot(fx1 - fx0, fy1 - fy0), 1e-6)
        shifts = []
        for aux_box in aux_boxes:
            ax0, ay0, ax1, ay1 = aux_box
            aux_center = np.asarray([(ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0], dtype=float)
            shifts.append(float(np.linalg.norm(aux_center - final_center) / diag))
        center_shifts.append(float(np.mean(shifts)))

    return {
        "proxy_decoder_logit_std_mean": float(np.mean(logit_stds)),
        "proxy_decoder_top_class_flip_frac": float(np.mean(class_flip_flags)),
        "proxy_decoder_box_iou_gap_mean": float(np.mean(box_iou_gaps)),
        "proxy_decoder_box_center_shift_mean": float(np.mean(center_shifts)),
    }


def compute_proxy_summary(
    original_rows: list[dict[str, Any]],
    weak_rows: list[dict[str, Any]],
    strong_rows: list[dict[str, Any]],
    *,
    weak_strong_iou_thresh: float,
) -> dict[str, float]:
    """Compute image-level no-GT proxy scores from model outputs only.

    Inputs:
    - prediction rows from the original, weak, and strong views for one image
    - `weak_strong_iou_thresh`: IoU threshold used for weak/strong matching

    Output:
    - a flat dict of image-level proxy scores

    Score meanings:
    - confidence proxies: how uncertain or unstable the original-view detector
      looks based on confidence magnitudes and spread
    - disagreement proxies: how inconsistent weak/strong predictions look after
      both views are expressed in the same image coordinate system

    Assumptions:
    - weak and strong boxes are already expressed in the original image
      coordinate system before calling this function
    - no GT is used here; every returned score is intended to be a candidate
      proxy that could still be computed when target labels are unavailable
    """

    orig_scores = np.asarray([row["score"] for row in original_rows], dtype=float)
    low_conf_mask = orig_scores < 0.5 if len(orig_scores) else np.asarray([], dtype=bool)

    matches, unmatched_weak, unmatched_strong = greedy_match_rows(
        weak_rows,
        strong_rows,
        iou_thresh=weak_strong_iou_thresh,
        class_aware=False,
    )

    matched_ious = np.asarray([match["iou"] for match in matches], dtype=float)
    matched_score_diffs = np.asarray(
        [abs(match["left"]["score"] - match["right"]["score"]) for match in matches],
        dtype=float,
    )
    matched_center_shifts = []
    for match in matches:
        lx0, ly0, lx1, ly1 = match["left"]["bbox"]
        rx0, ry0, rx1, ry1 = match["right"]["bbox"]
        left_center = np.asarray([(lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0], dtype=float)
        right_center = np.asarray([(rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0], dtype=float)
        diag = max(np.hypot(lx1 - lx0, ly1 - ly0), 1e-6)
        matched_center_shifts.append(float(np.linalg.norm(left_center - right_center) / diag))
    matched_center_shifts = np.asarray(matched_center_shifts, dtype=float)
    class_disagree_count = sum(
        int(match["left"]["category_id"] != match["right"]["category_id"])
        for match in matches
    )

    pred_count = float(len(original_rows))
    match_count = float(len(matches))
    disagreement_count = float(len(unmatched_weak) + len(unmatched_strong) + class_disagree_count)
    low_conf_count = float(low_conf_mask.sum()) if len(orig_scores) else 0.0

    return {
        "proxy_pred_count": pred_count,
        "proxy_mean_score": float(orig_scores.mean()) if len(orig_scores) else 0.0,
        "proxy_score_std": float(orig_scores.std()) if len(orig_scores) else 0.0,
        "proxy_low_conf_count": low_conf_count,
        "proxy_low_conf_frac": float(low_conf_count / pred_count) if pred_count else 0.0,
        "proxy_low_conf_mass": float(np.maximum(0.0, 0.5 - orig_scores).sum()) if len(orig_scores) else 0.0,
        "proxy_ws_match_count": match_count,
        "proxy_ws_unmatched_count": float(len(unmatched_weak) + len(unmatched_strong)),
        "proxy_ws_unmatched_frac": float((len(unmatched_weak) + len(unmatched_strong)) / max(1.0, len(weak_rows) + len(strong_rows))),
        "proxy_ws_mean_iou": float(matched_ious.mean()) if len(matched_ious) else 0.0,
        "proxy_ws_iou_gap": float(1.0 - matched_ious.mean()) if len(matched_ious) else 1.0,
        "proxy_ws_center_shift_mean": float(matched_center_shifts.mean()) if len(matched_center_shifts) else 0.0,
        "proxy_ws_mean_score_diff": float(matched_score_diffs.mean()) if len(matched_score_diffs) else 0.0,
        "proxy_ws_class_disagree_count": float(class_disagree_count),
        "proxy_ws_class_disagree_frac": float(class_disagree_count / match_count) if match_count else 0.0,
        "proxy_ws_disagreement_count": disagreement_count,
    }


def zscore(values: list[float] | np.ndarray) -> np.ndarray:
    """Return a stable z-score array, falling back to zeros for constant input.

    This is used in the notebook to combine heterogeneous proxy scores without
    letting one raw numeric scale dominate only because of units.
    """

    values = np.asarray(values, dtype=float)
    std = values.std()
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std
