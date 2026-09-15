"""GroundingSuite-Eval metrics (gIoU per stratum, box Acc@IoU) over the predictions.jsonl written by
eval/groundingsuite_eval.py; launched by scripts/eval_gseval.sh.
"""
import argparse
import collections
import json
import os
import re

import numpy as np
from pycocotools import mask as mask_utils

# class_id -> stratum of the GroundingSuite paper (matched through the per-class counts of
# GroundingSuite-Eval.jsonl); the raw class_id is printed as well.
STRATA = {1: 'stuff', 2: 'part', 3: 'multi-object', 4: 'single-object'}


def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# Copied from GroundingSuite's evaluate_grounding.py (do not modify):
# https://github.com/hustvl/GroundingSuite/blob/main/evaluate_grounding.py
def rle_to_mask(rle):
    if not rle or 'counts' not in rle or 'size' not in rle:
        return None
    try:
        return mask_utils.decode(rle)
    except Exception:
        return None


def calculate_mask_iou(mask1, mask2):
    if mask1 is None or mask2 is None:
        return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def calculate_box_iou(box1, box2):
    x_min = max(box1[0], box2[0]); y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2]); y_max = min(box1[3], box2[3])
    if x_max < x_min or y_max < y_min:
        return 0.0
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def score(gt_data, pred_data, mode, iou_threshold):
    """As GroundingEvaluator.evaluate() in the official scorer: match predictions to GT by idx
    (fallback image_path); mask mode -> per-class gIoU (mean IoU); box mode -> per-class Acc@IoU.
    Returns (overall, {class_id: score}, {class_id: count}, n_scored)."""
    pred_map = {p.get('idx'): p for p in pred_data if p.get('idx') is not None}
    if not pred_map:
        pred_map = {p.get('image_path'): p for p in pred_data if p.get('image_path')}

    class_ious = collections.defaultdict(list)      # mask mode
    class_total = collections.defaultdict(int)      # box mode
    class_correct = collections.defaultdict(int)    # box mode
    n = 0
    for gt in gt_data:
        cid = gt.get('class_id', 0)
        pred = pred_map.get(gt.get('idx'))
        if pred is None:
            pred = pred_map.get(gt.get('image_path'))
        if pred is None:
            continue
        if mode == 'mask':
            gt_seg = gt.get('segmentation')
            pred_seg = pred.get('segmentation') or pred.get('predicted_segmentation')
            if not gt_seg or not pred_seg:
                continue
            iou = calculate_mask_iou(rle_to_mask(gt_seg), rle_to_mask(pred_seg))
            class_ious[cid].append(iou)
        else:  # box
            gt_box = gt.get('box')
            pred_box = pred.get('box') or pred.get('predicted_box')
            if gt_box is None or pred_box is None:
                continue
            class_total[cid] += 1
            if calculate_box_iou(gt_box, pred_box) >= iou_threshold:
                class_correct[cid] += 1
        n += 1

    if mode == 'mask':
        per_class = {c: (sum(v) / len(v) if v else 0.0) for c, v in class_ious.items()}
        counts = {c: len(v) for c, v in class_ious.items()}
        all_ious = [i for v in class_ious.values() for i in v]
        overall = sum(all_ious) / len(all_ious) if all_ious else 0.0
    else:
        per_class = {c: (class_correct[c] / class_total[c] if class_total[c] else 0.0)
                     for c in class_total}
        counts = dict(class_total)
        total_correct = sum(class_correct.values())
        overall = total_correct / n if n else 0.0
    return overall, per_class, counts, n


def _print_table(title, overall, per_class, counts, metric_name):
    print(f"\n===== {title}  (0-100) =====")
    print(f"{'stratum':<16}{'class_id':>9}  {metric_name:>8}  {'n':>6}")
    print("-" * 44)
    for cid in sorted(STRATA):
        s = per_class.get(cid)
        s = f"{s:.2f}" if s is not None else " n/a "
        print(f"{STRATA[cid]:<16}{cid:>9}  {s:>8}  {counts.get(cid, 0):>6}")
    print("-" * 44)
    print(f"{'AGGREGATE (all)':<16}{'':>9}  {overall:>8.2f}  {sum(counts.values()):>6}")


def write_viz(gt_data, pred_data, out_dir):
    """Join predictions and GT by idx into predictions_viz.jsonl, the visualizer schema also
    written by refcoco_eval.py."""
    pred_map = {p.get('idx'): p for p in pred_data if p.get('idx') is not None}
    path = os.path.join(out_dir, 'predictions_viz.jsonl')
    n = 0
    with open(path, 'w', encoding='utf-8') as f:
        for gt in gt_data:
            idx = gt.get('idx')
            p = pred_map.get(idx)
            if p is None:
                continue
            pred_seg = p.get('predicted_segmentation') or p.get('segmentation')
            pred_masks = [pred_seg] if pred_seg else []
            pred_text = p.get('pred_text', '') or ''
            caption = gt.get('caption', '') or gt.get('label', '')
            image_rel = gt.get('image_path', '')
            cid = gt.get('class_id')
            f.write(json.dumps({
                'uid': str(idx),
                'image_id': str(idx),
                'image_file': os.path.basename(image_rel),
                'image_path': image_rel,
                'query_index': 0,
                'ann_id': None,
                'caption': caption,
                'gt_mask': gt.get('segmentation'),
                'phrase_seg_counts': [len(pred_masks)],
                'pred_masks': pred_masks,
                'query': caption,
                'pred_text': pred_text,
                'phrases': [s.strip() for s in re.findall(r'<p>(.*?)</p>', pred_text, flags=re.DOTALL)],
                'sample_index': idx,
                'class_id': cid,
                'stratum': STRATA.get(cid),
            }, ensure_ascii=False) + '\n')
            n += 1
    print(f"[metrics_groundingsuite] wrote {n} viz rows -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description='Score PANORAMA on GroundingSuite-Eval')
    p.add_argument('--pred-file', required=True, help='predictions.jsonl from groundingsuite_eval.py')
    p.add_argument('--gseval-root', default=os.environ.get('PANORAMA_GSEVAL_ROOT'),
                   help='Root with GroundingSuite-Eval.jsonl (default $PANORAMA_GSEVAL_ROOT).')
    p.add_argument('--gt-file', default=None,
                   help='Override; default <gseval-root>/GroundingSuite-Eval.jsonl.')
    p.add_argument('--output-dir', default=None, help='default = dir of --pred-file.')
    p.add_argument('--iou-threshold', type=float, default=0.5, help='box-mode Acc@IoU threshold.')
    args = p.parse_args()
    if args.gt_file is None:
        if not args.gseval_root:
            p.error('need --gseval-root (or $PANORAMA_GSEVAL_ROOT) or an explicit --gt-file')
        args.gt_file = os.path.join(args.gseval_root, 'GroundingSuite-Eval.jsonl')
    return args


def main():
    args = parse_args()
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.pred_file))
    os.makedirs(out_dir, exist_ok=True)

    gt_data = read_jsonl(args.gt_file)
    pred_data = read_jsonl(args.pred_file)
    if len(gt_data) != len(pred_data):
        print(f"[gseval][WARN] {len(pred_data)} predictions vs {len(gt_data)} GT items")

    # gIoU (mask) is the primary metric of the benchmark; Acc@IoU (box) is secondary.
    giou, giou_by_class, giou_counts, n_mask = score(gt_data, pred_data, 'mask', args.iou_threshold)
    acc, acc_by_class, acc_counts, n_box = score(gt_data, pred_data, 'box', args.iou_threshold)
    # Reported on the 0-100 scale, as in the GroundingSuite paper.
    giou *= 100; giou_by_class = {c: v * 100 for c, v in giou_by_class.items()}
    acc *= 100;  acc_by_class = {c: v * 100 for c, v in acc_by_class.items()}

    _print_table('GSEval gIoU (mask, primary)', giou, giou_by_class, giou_counts, 'gIoU')
    _print_table(f'GSEval Acc@{args.iou_threshold} (box)', acc, acc_by_class, acc_counts, 'Acc')

    summary = {
        'primary_metric': 'gIoU (mask)',
        'scale': '0-100 (as in the GroundingSuite paper)',
        'mask': {
            'overall_giou': giou,
            'per_stratum': {STRATA[c]: giou_by_class.get(c) for c in sorted(STRATA)},
            'per_class_id': {str(c): giou_by_class.get(c) for c in sorted(giou_by_class)},
            'counts': {STRATA.get(c, str(c)): giou_counts.get(c, 0) for c in sorted(giou_counts)},
            'n_scored': n_mask,
        },
        'box': {
            'overall_acc': acc, 'iou_threshold': args.iou_threshold,
            'per_stratum': {STRATA[c]: acc_by_class.get(c) for c in sorted(STRATA)},
            'n_scored': n_box,
        },
    }
    _p = os.path.join(out_dir, 'metrics_groundingsuite.json')
    with open(_p, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[metrics_groundingsuite] wrote {_p}")

    # Visualizer file next to the metrics, for the same viewer refcoco uses.
    write_viz(gt_data, pred_data, out_dir)


if __name__ == '__main__':
    main()
