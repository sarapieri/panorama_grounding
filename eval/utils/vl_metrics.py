# Ours: PanoCaps metrics (gPQ, precision, recall, F1, mIoU). Predicted and ground-truth
# phrase-mask pairs are matched by Hungarian assignment on mask IoU and phrase similarity.
import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .text_similarity import TextSimilarityMetric

def _encode_binmask(binmask: np.ndarray) -> Dict[str, Any]:
    rle = maskUtils.encode(np.asfortranarray(binmask.astype(np.uint8)))
    # pycocotools returns bytes for counts; make it JSON serializable
    rle["counts"] = rle["counts"].decode("ascii")
    return rle

def deduplicate_and_decode_gt_masks(
    labels_data: List[Dict[str, Any]],
    gt_anns: List[Dict[str, Any]],
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Deduplicate gt RLE masks and labels by annotations.
    """
    # build mapping mask_id -> txt_desc
    id_to_texts = defaultdict(list)

    for entry in labels_data:
        text = entry["txt_desc"]
        for mask_id in entry["mask_ids"]:
            if text not in id_to_texts[mask_id]:
                id_to_texts[mask_id].append(text)

    # sort available ids
    sorted_ids = sorted(id_to_texts.keys())
    valid_ids = [mid for mid in sorted_ids if 0 <= mid < len(gt_anns)]

    # ordered text descriptions
    gt_labels = [id_to_texts[mid] for mid in valid_ids]

    # decode masks (skip missing ids)
    gt_masks  = [maskUtils.decode(gt_anns[mid]['segmentation']) for mid in valid_ids]

    return gt_labels, gt_masks

def deduplicate_and_decode_dt_masks(
    labels_data: Union[List[str], List[List[str]]],
    phrase_seg_counts: Optional[List[int]],
    dt_anns: List[Dict[str, Any]],
    metric: TextSimilarityMetric,
    iou_thr: float = 0.9,
    label_sim_thr: float = 0.5,
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Deduplicate predicted RLE masks by IoU and merge labels, but only if labels match too.
    """

    # expand labels per SEG count
    if phrase_seg_counts is None:
        repeated = list(labels_data)
    else:
        if len(labels_data) != len(phrase_seg_counts):
            raise ValueError(f"Length mismatch: labels={len(labels_data)} vs seg_counts={len(phrase_seg_counts)}")
        # Cap the expanded labels at 50 per image.
        MAX_EXPANDED = 50
        repeated = []
        for label, count in zip(labels_data, phrase_seg_counts):
            if count < 0:
                raise ValueError(f"Negative seg count for label '{label}': {count}")
            if count == 0:
                continue

            remaining = MAX_EXPANDED - len(repeated)
            if remaining <= 0:
                print(
                    "[VL][WARN] Expanded labels reached cap ({}). "
                    "Current expanded length: {}".format(MAX_EXPANDED, len(repeated))
                )
                break

            repeated.extend([label] * min(count, remaining))

    # extract RLEs from annotations
    rles = []
    for ann in dt_anns:
        seg = ann.get("segmentation", ann) if isinstance(ann, dict) else ann
        if isinstance(seg, dict) and ("counts" in seg) and ("size" in seg):
            rles.append(seg)  # compressed RLE as expected by pycocotools
        else:
            raise ValueError("Expected RLE dict at ann['segmentation'] with 'size' and 'counts'.")

    # align counts
    if len(rles) != len(repeated):
        print(
            "[VL][WARN] Predicted masks / labels misalignment: "
            "len(pred_masks)={} vs len(expanded labels)={}".format(
                len(rles), len(repeated)
            )
        )
        min_len = min(len(rles), len(repeated))
        if min_len == 0:
            print("[VL][WARN] One side is empty; returning empty alignment.")
            return [], []
        rles = rles[:min_len]
        repeated = repeated[:min_len]

    # decode all to boolean masks
    decoded_masks = [maskUtils.decode(rle).astype(bool, copy=False) for rle in rles]

    # deduplicate: keep the first mask as representative; merge text labels if masks IoU-match and labels match
    kept_masks: List[np.ndarray] = []
    kept_labels: List[List[str]] = []

    for mask, lbl in zip(decoded_masks, repeated):
        # Ensure lbl becomes a list[str]
        lbl_list = list(lbl) if isinstance(lbl, (list, tuple, set)) else [lbl]
        lbl_list = [str(x) for x in lbl_list if x is not None]

        merged = False
        for i, kmask in enumerate(kept_masks):
            # IoU gate
            if compute_iou(mask, kmask) >= iou_thr:
                # label gate
                if _labels_match(lbl_list, kept_labels[i], metric=metric, label_sim_thr=label_sim_thr):
                    # merge labels (dedupe, keep original order preference)
                    for t in lbl_list:
                        if t not in kept_labels[i]:
                            kept_labels[i].append(t)
                    merged = True
                    break
                # else same region but different label -> do not merge; treat as separate instance

        if not merged:
            kept_masks.append(mask)
            kept_labels.append(lbl_list)

    dt_labels = kept_labels
    pred_masks = kept_masks
    return dt_labels, pred_masks

def build_dedup_views(
    gt_mask_path: str,
    pred_save_path: str,
    gt_cap_path: str,
    cap_pred_save_path: str,
    tmp_dir: str,
    image_ids: List[int],
    metric: TextSimilarityMetric,  # text similarity metric
    iou_thr: float = 0.9,          # for deduping instances
    label_sim_thr: float = 0.5     # for merging label refs
    ) -> Dict[str, str]:
    """
    Creates deduped copies of gt/dt for both masks and captions.
    Returns paths to the deduped files to feed into all evaluators.
    """
    # raw COCO objects
    coco_gt = COCO(gt_mask_path)
    coco_dt = coco_gt.loadRes(pred_save_path)
    coco_cap_gt = COCO(gt_cap_path)
    coco_cap_dt = coco_cap_gt.loadRes(cap_pred_save_path)

    # prepare output containers
    gt_ds = copy.deepcopy(coco_gt.dataset)
    gt_ds["annotations"] = []
    gt_ds["categories"] = [{"id": 1, "name": "object"}]

    dt_anns_out: List[Dict[str, Any]] = []
    cap_gt_out = {"images": copy.deepcopy(coco_cap_gt.dataset.get("images", [])), "annotations": []}
    cap_dt_out: List[Dict[str, Any]] = []

    next_gt_ann_id = 0
    next_dt_ann_id = 0

    for image_id in image_ids:
        # collect raw anns for image
        gt_ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        dt_ann_ids = coco_dt.getAnnIds(imgIds=[image_id])
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        gt_cap_ids = coco_cap_gt.getAnnIds(imgIds=[image_id])
        if not gt_cap_ids:
            continue
        gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ids)[0]

        dt_cap_ids = coco_cap_dt.getAnnIds(imgIds=[image_id])
        if not dt_cap_ids:
            continue
        dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ids)[0]

        # dedup & decode
        gt_label_source = gt_cap_ann.get("label_matched", gt_cap_ann.get("labels", []))
        gt_labels_dedup, gt_masks_decoded = deduplicate_and_decode_gt_masks(gt_label_source, gt_anns)

        if "phrase_seg_counts" in dt_cap_ann:
            dt_labels_dedup, dt_masks_decoded = deduplicate_and_decode_dt_masks(
                dt_cap_ann["labels"], dt_cap_ann["phrase_seg_counts"], dt_anns, metric, iou_thr, label_sim_thr
            )
        else:
            dt_labels_dedup, dt_masks_decoded = deduplicate_and_decode_dt_masks(
                dt_cap_ann["labels"], None, dt_anns, metric, iou_thr, label_sim_thr
            )

        # encode masks back to RLE & (re)build COCO-style records
        # gt masks
        for m in gt_masks_decoded:
            rle = _encode_binmask(m)
            area = float(maskUtils.area(rle))
            bbox = [float(x) for x in maskUtils.toBbox(rle)]
            gt_ds["annotations"].append({
                "id": next_gt_ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
            })
            next_gt_ann_id += 1

        # dt masks
        for m in dt_masks_decoded:
            rle = _encode_binmask(m)
            dt_anns_out.append({
                "id": next_dt_ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "score": 1.0,
            })
            next_dt_ann_id += 1

        # captions: keep text, refresh label metadata + seg counts
        # gt caption
        cap_gt_out["annotations"].append({
            "id": gt_cap_ann["id"],
            "image_id": image_id,
            "caption": gt_cap_ann["caption"],
            "labels": gt_labels_dedup,
        })

        # dt caption prediction
        cap_dt_out.append({
            "image_id": image_id,
            "caption": dt_cap_ann["caption"],
            "labels": dt_labels_dedup,
        })

    # write files
    out_gt_mask = os.path.join(tmp_dir, "gt_mask_dedup.json")
    out_dt_mask = os.path.join(tmp_dir, "dt_mask_dedup.json")
    out_gt_cap  = os.path.join(tmp_dir, "gt_caption_dedup.json")
    out_dt_cap  = os.path.join(tmp_dir, "dt_caption_dedup.json")

    with open(out_gt_mask, "w", encoding="utf-8") as f:
        json.dump(gt_ds, f)
    with open(out_dt_mask, "w", encoding="utf-8") as f:
        json.dump(dt_anns_out, f)
    with open(out_gt_cap, "w", encoding="utf-8") as f:
        json.dump(cap_gt_out, f)
    with open(out_dt_cap, "w", encoding="utf-8") as f:
        json.dump(cap_dt_out, f)

    return {
        "gt_mask": out_gt_mask,
        "dt_mask": out_dt_mask,
        "gt_cap": out_gt_cap,
        "dt_cap": out_dt_cap,
    }

def _labels_match(
    labels_a: Iterable[str],
    labels_b: Iterable[str],
    metric: TextSimilarityMetric,
    label_sim_thr: float = 0.5,
    ) -> bool:
    """
    Returns True if there is any label in A that matches any label in B.
    Matching is:
      - exact string match (case-insensitive), or
      - similarity >= label_sim_thr if `metric` is provided and has `compute_similarity(a, b)`.
    """
    # normalize to lowercase strings
    a = [str(x).strip().lower() for x in labels_a if x is not None]
    b = [str(x).strip().lower() for x in labels_b if x is not None]

    # fast path: any exact overlap
    set_a, set_b = set(a), set(b)
    if set_a & set_b:
        return True

    # similarity path
    if metric is not None and hasattr(metric, "compute_similarity"):
        for la in a:
            for lb in b:
                sim = metric.compute_similarity(la, lb)
                if sim is not None and sim >= label_sim_thr:
                    return True
    return False

def evaluate_with_mapping(
    coco_gt: COCO,
    coco_cap_gt: COCO,
    coco_dt: COCO,
    coco_cap_dt: COCO,
    image_ids: List[int],
    metric: TextSimilarityMetric,
    iou_threshold: float = 0.5,
    text_sim_threshold: float = 0.5,
    ) -> Dict[str, float]:
    """
    Precision, Recall, F1 and grounded panoptic quality (gPQ) over matched phrase-mask pairs.
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    actual_positives = 0
    pq_sum = 0  # sum of IoU x text similarity over matched pairs

    for image_id in tqdm(image_ids, desc="Evaluating VL metrics"):
        try:
            # load gt & dt mask annotations
            matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
            gt_ann_ids = [ann['id'] for ann in matching_anns]
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
            dt_ann_ids = [ann['id'] for ann in matching_anns]
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            # load gt & dt caption annotations
            matching_anns = [ann for ann in coco_cap_gt.anns.values() if str(ann['image_id']) == str(image_id)]
            gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]

            matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == image_id]
            dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

            # prepare gt
            gt_labels = gt_cap_ann['labels']
            gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns]
            dt_labels = dt_cap_ann['labels']
            dt_masks = [maskUtils.decode(ann['segmentation']) for ann in dt_anns]

            actual_positives += len(gt_labels)

            # find best matching pairs
            best_matches, matched_ious, matched_text_sims = find_best_matches(gt_masks, gt_labels, dt_masks, dt_labels, metric,
                                                                              iou_threshold, text_sim_threshold)

            num_matches = len(best_matches)
            true_positives += num_matches
            false_positives += len(dt_labels) - num_matches  # unmatched predictions
            false_negatives += len(gt_labels) - num_matches  # unmatched GTs

            # gPQ numerator
            pq_sum += sum(iou * text_sim for iou, text_sim in zip(matched_ious, matched_text_sims))

        except Exception as e:
            print("[VL][ERROR] Error processing image {}: {}".format(image_id, e))

    # compute metrics
    precision = (
        true_positives / float(true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / float(actual_positives)
        if actual_positives > 0
        else 0.0
    )
    pq_den = true_positives + 0.5 * (false_positives + false_negatives)
    pq = pq_sum / float(pq_den) if pq_den > 0 else 0.0
    f1_score = (
        2.0 * precision * recall / float(precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(
        "[VL] Recall: {:.3f}, Precision: {:.3f}, F1-score: {:.3f}, gPQ: {:.3f}".format(
            recall, precision, f1_score, pq
        )
    )

    iou_suffix = "@{}".format(iou_threshold)
    return {
        "Recall{}".format(iou_suffix): recall,
        "Precision{}".format(iou_suffix): precision,
        "F1{}".format(iou_suffix): f1_score,
        "gPQ{}".format(iou_suffix): pq,
    }

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    union_sum = np.sum(union)
    if union_sum == 0:
        return 0.0
    return float(np.sum(intersection) / union_sum)

def compute_iou_matrix(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    ) -> np.ndarray:
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    return iou_matrix

def compute_miou(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    ) -> float:
    """
    Compute mean IoU (mIoU) between predicted masks and ground truth masks
    using the Hungarian algorithm.
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return 0.0

    iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    paired_iou = [iou_matrix[i, j] for i, j in zip(row_indices, col_indices)]
    return float(np.sum(paired_iou) / max(len(pred_masks), len(gt_masks)))

def evaluate_mask_miou(
    coco_gt: COCO,
    coco_dt: COCO,
    image_ids: List[int],
    ) -> float:
    # load predictions
    mious = []
    for image_id in tqdm(image_ids):
        # ground truth masks
        matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
        ann_ids = [ann['id'] for ann in matching_anns]
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns if 'segmentation' in ann]

        # predicted masks
        matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
        dt_ann_ids = [ann['id'] for ann in matching_anns]
        pred_anns = coco_dt.loadAnns(dt_ann_ids)
        pred_masks = [maskUtils.decode(ann['segmentation']) for ann in pred_anns if 'segmentation' in ann]

        # compute and save the mIoU
        mious.append(compute_miou(pred_masks, gt_masks))

    # report mean IoU across all images
    mean_miou = np.mean(mious) if mious else 0.0

    print("[mIoU] Mean IoU across all images: {:.3f}".format(mean_miou))
    return mean_miou

def find_best_matches(
    gt_masks: List[np.ndarray],
    gt_labels: Union[List[List[str]], List[str]],
    dt_masks: List[np.ndarray],
    dt_labels: Union[List[List[str]], List[str]],
    metric: TextSimilarityMetric,
    iou_threshold: float,
    text_sim_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
    """
    Finds best matching GT-DT pairs based on IoU and text similarity.
    """
    best_matches = []
    matched_ious = []
    matched_sims = []

    if len(gt_labels) == 0 or len(dt_labels) == 0:
        return [], [], []

    # compute pairwise IoU
    ious = compute_iou_matrix(gt_masks, dt_masks)

    # compute text similarity matrix
    text_sims = metric.compute_similarity_matrix(gt_labels, dt_labels)
    # metric.pretty_print_similarity_matrix(text_sims, gt_labels, dt_labels)

    # create a cost matrix
    cost_matrix = -(ious + text_sims) / 2

    # solve the assignment problem using the Hungarian algorithm
    gt_indices, dt_indices = linear_sum_assignment(cost_matrix)

    # process matches
    for gt_idx, dt_idx in zip(gt_indices, dt_indices):
        iou_value = ious[gt_idx, dt_idx]
        text_sim_value = text_sims[gt_idx, dt_idx]

        # apply thresholds
        if iou_value >= iou_threshold and text_sim_value >= text_sim_threshold:
            best_matches.append((gt_idx, dt_idx))
            matched_ious.append(iou_value)
            matched_sims.append(text_sim_value)

    return best_matches, matched_ious, matched_sims