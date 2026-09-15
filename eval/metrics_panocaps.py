"""PanoCaps metrics (gPQ, precision, recall, F1, AP50, mIoU, CAPTURE) over the per-image jsons
written by eval/panocaps_eval.py; launched by scripts/eval_panocaps.sh.
"""
import argparse
import json
import os
import pickle
import shutil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .utils import TextSimilarityMetric, build_dedup_views, evaluate_mask_miou, evaluate_with_mapping


_CAPTURE_EVAL = None


def _get_capture():
    """CAPTURE singleton: its scene-graph parser and sentence embedder are loaded once, not per
    split. Imported lazily so this module still works without capture_metric."""
    global _CAPTURE_EVAL
    if _CAPTURE_EVAL is None:
        from capture_metric.capture import CAPTURE
        _CAPTURE_EVAL = CAPTURE()
    return _CAPTURE_EVAL


def parse_args():
    parser = argparse.ArgumentParser(description="PanoCaps metrics.")
    parser.add_argument(
        "--split", type=str, required=True, nargs="+",
        help="One or more splits (GT filename keys, e.g. val test); reads <split>_mask[s].json "
             "and <split>_caption.json. Each split is scored separately.",
    )
    parser.add_argument(
        "--prediction_dir_path", type=str, required=True,
        help="Folder with the per-image json predictions of eval/panocaps_eval.py.",
    )
    parser.add_argument(
        "--gt_dir_path", type=str, required=True,
        help="Folder with the PanoCaps annotations.",
    )
    parser.add_argument(
        "--pooled", action="store_true",
        help="With several splits, also score their union as one 'full' set "
             "(the pooled metric over all images, not the mean of the per-split scores).",
    )
    parser.add_argument(
        "--pooled-only", action="store_true",
        help="Score only the pooled union of the given splits; implies --pooled.",
    )
    parser.add_argument(
        "--no-capture", action="store_true",
        help="Skip the CAPTURE caption metric, whose scene-graph parser dominates the runtime.",
    )
    parser.add_argument(
        "--metrics-name", type=str, default="metrics.json",
        help="Filename of the metrics summary written into --prediction_dir_path.",
    )
    args = parser.parse_args()
    if args.pooled_only:
        args.pooled = True
    return args


def evaluate_split(split, prediction_dir_path, gt_dir_path, no_capture=False):
    # GT mask file: <split>_mask.json, or <split>_masks.json.
    gt_mask_path = f"{gt_dir_path}/{split}_mask.json"
    if not os.path.isfile(gt_mask_path):
        _plural = f"{gt_dir_path}/{split}_masks.json"
        if os.path.isfile(_plural):
            gt_mask_path = _plural
    gt_cap_path = f"{gt_dir_path}/{split}_caption.json"

    print(f"Starting evaluation on '{split}' split.")
    results = {}

    # load all image IDs from GT captions
    all_images_ids = []
    with open(gt_cap_path, "r", encoding="utf-8") as f:
        contents = json.load(f)
        for image in contents["images"]:
            all_images_ids.append(image["id"])

    # Per-split temp dir for the intermediate COCO-format dumps; removed at the end.
    tmp_dir_path = f'tmp/{os.environ.get("SLURM_JOB_ID", "nojid")}_{split}'
    os.makedirs(tmp_dir_path, exist_ok=True)

    # build prediction files in COCO-compatible formats
    pred_save_path = f"{tmp_dir_path}/mask_pred_tmp_save.json"
    cap_pred_save_path = f"{tmp_dir_path}/cap_pred_tmp_save.json"

    coco_pred_file = []
    caption_pred_dict = {}

    for image_id in all_images_ids:
        prediction_path = f"{prediction_dir_path}/{image_id}.json"
        if not os.path.isfile(prediction_path):
            raise FileNotFoundError(
                f"Prediction file not found for image_id={image_id}: {prediction_path}"
            )

        with open(prediction_path, "r", encoding="utf-8") as f:
            pred = json.load(f)

        # Prediction format written by eval/panocaps_eval.py.
        caption_pred_dict[image_id] = {
            "caption": pred["caption"],
            "labels": pred["phrases"],
        }
        if "phrase_seg_counts" in pred:
            caption_pred_dict[image_id]["phrase_seg_counts"] = pred["phrase_seg_counts"]

        for rle_mask in pred["pred_masks"]:
            coco_pred_file.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle_mask,
                    "score": 1.0,
                }
            )

    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(coco_pred_file, f)

    # prepare caption predictions in COCO-like format
    coco_cap_pred_file = []
    for image_id, values in caption_pred_dict.items():
        entry = {
            "image_id": image_id,
            "caption": values["caption"],
            "labels": values["labels"],
        }
        if "phrase_seg_counts" in values:
            entry["phrase_seg_counts"] = values["phrase_seg_counts"]
        coco_cap_pred_file.append(entry)

    with open(cap_pred_save_path, "w", encoding="utf-8") as f:
        json.dump(coco_cap_pred_file, f)

    # load GT captions
    with open(gt_cap_path, "r", encoding="utf-8") as f:
        coco_cap = json.load(f)

    image_ids = [img["id"] for img in coco_cap.get("images", [])]
    metric = TextSimilarityMetric()

    paths = build_dedup_views(
        gt_mask_path=str(gt_mask_path),
        pred_save_path=str(pred_save_path),
        gt_cap_path=str(gt_cap_path),
        cap_pred_save_path=str(cap_pred_save_path),
        tmp_dir=str(tmp_dir_path),
        image_ids=image_ids,
        metric=metric,
        iou_thr=0.9,
        label_sim_thr=0.5,
    )

    coco_gt = COCO(paths["gt_mask"])
    coco_dt = coco_gt.loadRes(paths["dt_mask"])

    coco_cap_gt = COCO(paths["gt_cap"])
    coco_cap_dt = coco_cap_gt.loadRes(paths["dt_cap"])

    # ------------------------------------------------------------------
    # 1) Evaluate Mask Prediction (COCOEval)
    # ------------------------------------------------------------------
    print("\033[92mEvaluating AP50\033[0m")
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    results["AP50"] = coco_eval.stats[1]

    # ------------------------------------------------------------------
    # 2) Evaluate CAPTURE (https://github.com/foundation-multimodal-models/CAPTURE): parses each
    #    caption into a scene graph and soft-matches objects, attributes and relations. A missing
    #    package or model logs and skips, never failing the eval.
    # ------------------------------------------------------------------
    if no_capture:
        print("\033[93mSkipping CAPTURE (--no-capture)\033[0m")
    else:
        print("\033[92mEvaluating CAPTURE\033[0m")
        try:
            # One GT and one predicted caption per image, as {id: [caption]} with str keys so
            # int/str image ids match across refs and candidates.
            cap_ids = coco_cap_dt.getImgIds()
            gts_cap = {str(i): [a["caption"] for a in coco_cap_gt.imgToAnns.get(i, [])] for i in cap_ids}
            res_cap = {str(i): [a["caption"] for a in coco_cap_dt.imgToAnns.get(i, [])] for i in cap_ids}
            # keep only images with both a non-empty GT and a predicted caption
            ids = [k for k in gts_cap if gts_cap[k] and gts_cap[k][0] and res_cap.get(k) and res_cap[k][0]]
            if ids:
                # The GT parse is the same for every model scored on this split and is slow, so
                # it is cached once per split (PANORAMA_CAPTURE_GT_CACHE relocates the cache,
                # default: beside the GT files). Delete the .pkl if the GT changes.
                _cache_dir = os.environ.get("PANORAMA_CAPTURE_GT_CACHE", gt_dir_path)
                _cache = os.path.join(_cache_dir, f"capture_gt_parsed_{split}.pkl")
                _prev_gt = None
                if os.path.isfile(_cache):
                    try:
                        with open(_cache, "rb") as _cf:
                            _cached = pickle.load(_cf)
                        if set(_cached) >= set(ids):
                            _prev_gt = {k: _cached[k] for k in ids}
                            print(f"[CAPTURE] reusing cached GT parse for '{split}' "
                                  f"({len(_prev_gt)} samples) from {_cache}")
                        else:
                            print(f"[CAPTURE] cache at {_cache} misses "
                                  f"{len(set(ids) - set(_cached))} ids -- reparsing GT")
                    except Exception as _e:
                        print(f"[CAPTURE][WARN] could not read GT cache: {_e}")
                _out = _get_capture().compute_score(
                    {k: gts_cap[k] for k in ids}, {k: res_cap[k] for k in ids},
                    prev_gt_parsed=_prev_gt, return_parse_results=(_prev_gt is None))
                if _prev_gt is None and isinstance(_out, tuple) and len(_out) == 3:
                    try:
                        os.makedirs(_cache_dir, exist_ok=True)
                        with open(_cache, "wb") as _cf:
                            pickle.dump({r["sample_key"]: r["gt_parsed"] for r in _out[2]}, _cf)
                        print(f"[CAPTURE] wrote GT parse cache -> {_cache}")
                    except Exception as _e:
                        print(f"[CAPTURE][WARN] could not write GT cache: {_e}")
                capture_score = _out
                if isinstance(capture_score, (tuple, list)):   # repo returns (mean, per_sample)
                    capture_score = capture_score[0]
                results["CAPTURE"] = float(capture_score)
                # CAPTURE is a plain mean over samples, so the count is recorded to allow a
                # count-weighted mean over several splits.
                results["CAPTURE_n"] = len(ids)
                print(f"CAPTURE: {results['CAPTURE']:.3f} (n={len(ids)})")
            else:
                print("[metrics_panocaps] no overlapping gt/pred captions -> skipping CAPTURE")
        except ImportError:
            print("[metrics_panocaps] capture_metric not installed -> skipping CAPTURE "
                  "(pip install capture_metric)")
        except Exception as e:
            print(f"[metrics_panocaps][WARN] CAPTURE failed -> skipping: {e}")

    # ------------------------------------------------------------------
    # 3) Evaluate mIoU
    # ------------------------------------------------------------------
    print("\033[92mEvaluating mIoU\033[0m")
    mean_miou = evaluate_mask_miou(coco_gt, coco_dt, image_ids)
    results["Mean IoU (mIoU)"] = mean_miou

    # ------------------------------------------------------------------
    # 4) Evaluate VL Metrics
    # ------------------------------------------------------------------
    print("\033[92mEvaluating VL Metrics\033[0m")
    metric = TextSimilarityMetric()
    results_vl = evaluate_with_mapping(
        coco_gt,
        coco_cap_gt,
        coco_dt,
        coco_cap_dt,
        image_ids,
        metric,
        iou_threshold=0.5,
        text_sim_threshold=0.5,
    )
    results.update(results_vl)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"===== Results [{split}] =====")
    for metric_name, score in results.items():
        print(f"{metric_name}: {score:.3f}")

    # Clean up tmp dir
    try:
        shutil.rmtree(tmp_dir_path)
    except Exception as e:
        print(f"[WARN] could not remove tmp dir {tmp_dir_path}: {e}")

    return {k: float(v) for k, v in results.items()}


def _merge_pooled_gt(splits, gt_dir_path, out_dir):
    """Concatenate the per-split GT (mask and caption COCO jsons) into one 'pooled' GT pair under
    out_dir, so evaluate_split('pooled', preds, out_dir) scores the union in one pass.

    Image ids are unique across splits (predictions are saved per image id in one folder), so
    they are kept; annotation ids restart per split and are re-numbered sequentially.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _resolve_mask(s):
        p = f"{gt_dir_path}/{s}_mask.json"
        return p if os.path.isfile(p) else f"{gt_dir_path}/{s}_masks.json"

    def _merge(paths):
        merged, ann_id = None, 1
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            for a in d.get("annotations", []):
                a["id"] = ann_id
                ann_id += 1
            if merged is None:
                merged = d
            else:
                merged["images"].extend(d["images"])
                merged["annotations"].extend(d.get("annotations", []))
        return merged

    merged_mask = _merge([_resolve_mask(s) for s in splits])
    merged_cap = _merge([f"{gt_dir_path}/{s}_caption.json" for s in splits])
    with open(f"{out_dir}/pooled_mask.json", "w", encoding="utf-8") as f:
        json.dump(merged_mask, f)
    with open(f"{out_dir}/pooled_caption.json", "w", encoding="utf-8") as f:
        json.dump(merged_cap, f)
    n_imgs = len(merged_cap.get("images", []))
    print(f"[pooled] merged {len(splits)} splits -> {n_imgs} images into {out_dir}")
    return out_dir


def _dump_metrics(out, prediction_dir_path, note="", name="metrics.json"):
    """Write the metrics summary after each split, so a killed job keeps the finished splits."""
    try:
        _p = os.path.join(prediction_dir_path, name)
        with open(_p, "w", encoding="utf-8") as _f:
            json.dump(out, _f, indent=2)
        print(f"[metrics_panocaps] wrote {_p}{note}")
    except Exception as e:
        print(f"[metrics_panocaps][WARN] could not write metrics.json: {e}")


def main():
    args = parse_args()
    splits = args.split

    per_split = {}
    if args.pooled_only:
        print("[metrics_panocaps] --pooled-only: skipping per-split evaluation")
    else:
        for split in splits:
            per_split[split] = evaluate_split(
                split, args.prediction_dir_path, args.gt_dir_path,
                no_capture=args.no_capture
            )
            _dump_metrics({"per_split": per_split}, args.prediction_dir_path,
                          f" (after '{split}')", name=args.metrics_name)

    out = {"per_split": per_split}
    # "full" = the pooled metric over the union of all splits (--pooled), not the mean of the
    # per-split scores: merge the GTs and score the union in one pass.
    if args.pooled and len(splits) > 1:
        tmp_gt_dir = f'tmp/{os.environ.get("SLURM_JOB_ID", "nojid")}_pooled_gt'
        _merge_pooled_gt(splits, args.gt_dir_path, tmp_gt_dir)
        print(f"===== FULL (pooled union of: {', '.join(splits)}) =====")
        out["full"] = evaluate_split("pooled", args.prediction_dir_path, tmp_gt_dir,
                                     no_capture=args.no_capture)
        # The merged GT is scratch too (evaluate_split already removed its own tmp dir).
        try:
            shutil.rmtree(tmp_gt_dir)
        except Exception as e:
            print(f"[WARN] could not remove pooled GT dir {tmp_gt_dir}: {e}")

    _dump_metrics(out, args.prediction_dir_path, name=args.metrics_name)


if __name__ == "__main__":
    main()