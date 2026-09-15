"""GCG metrics on GranD-f (AP50, caption metrics, mIoU, recall), following GLaMM's evaluation script.

Reads the per-image jsons written by eval/gcg_eval.py; launched by scripts/eval_gcg.sh.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from pycocoevalcap.eval import COCOEvalCap
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def parse_args():
    parser = argparse.ArgumentParser(description="GCG metrics")
    parser.add_argument("--split", required=True, help="evaluation split: 'val' or 'test'")
    parser.add_argument("--prediction_dir_path", required=True,
                        help="folder with the per-image json predictions of eval/gcg_eval.py")
    parser.add_argument("--gt_dir_path", default=None,
                        help="folder with the GranD-f evaluation annotations "
                             "(default: <data_root>/glamm_data/annotations/gcg_val_test/)")
    parser.add_argument("--data_root", default="./data",
                        help="root folder holding glamm_data/ (PANORAMA_DATA_ROOT)")
    return parser.parse_args()


# Load pre-trained model tokenizer and model for evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

    return sentence_embedding



def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    denom = np.sum(union)
    if denom == 0:
        return 0.0
    iou = np.sum(intersection) / denom

    return iou


def compute_miou(pred_masks, gt_masks):
    # Computing mIoU between predicted masks and ground truth masks
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0


def evaluate_mask_miou(coco_gt, image_ids, pred_save_path):
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_save_path)

    mious = []
    for image_id in tqdm(image_ids):
        # Getting ground truth masks
        matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
        ann_ids = [ann['id'] for ann in matching_anns]

        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns if 'segmentation' in ann]

        # Getting predicted masks
        matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
        dt_ann_ids = [ann['id'] for ann in matching_anns]
        pred_anns = coco_dt.loadAnns(dt_ann_ids)
        pred_masks = [maskUtils.decode(ann['segmentation']) for ann in pred_anns if 'segmentation' in ann]

        # Compute and save the mIoU for the current image
        mious.append(compute_miou(pred_masks, gt_masks))

    # Report mean IoU across all images
    mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0

    print(f"Mean IoU (mIoU) across all images: {mean_miou:.3f}")
    return mean_miou


def compute_iou_matrix(pred_masks, gt_masks):
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return iou_matrix


def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)

    return cosine_similarity([emb1], [emb2])[0, 0]


def find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold, vectorizer=None):
    best_matches = []

    # Compute pair - wise IoU
    pred_masks = [maskUtils.decode(ann['segmentation']) for ann in dt_anns]
    gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns]
    ious = compute_iou_matrix(gt_masks, pred_masks)

    text_sims = np.zeros((len(gt_labels), len(dt_labels)))

    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(dt_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # Find one-to-one matches satisfying both IoU and text similarity thresholds
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break  # No admissible pair found

        best_matches.append(max_iou_idx)

        # Remove selected annotations from consideration
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]


def evaluate_recall_with_mapping(coco_gt, coco_cap_gt, image_ids, pred_save_path, cap_pred_save_path, iou_threshold=0.5,
                                 text_sim_threshold=0.5):
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}

    coco_dt = coco_gt.loadRes(pred_save_path)

    coco_cap_dt = coco_cap_gt.loadRes(cap_pred_save_path)

    true_positives = 0
    actual_positives = 0

    for image_id in tqdm(image_ids):
        try:
            matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
            gt_ann_ids = [ann['id'] for ann in matching_anns]
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
            dt_ann_ids = [ann['id'] for ann in matching_anns]
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            matching_anns = [ann for ann in coco_cap_gt.anns.values() if ann['image_id'] == image_id]
            gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]

            matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == image_id]
            dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

            gt_labels = gt_cap_ann['labels']
            dt_labels = dt_cap_ann['labels']

            actual_positives += len(gt_labels)

            # Find best matching pairs
            best_matches = find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold)

            true_positives += len(best_matches)
        except Exception as e:
            print(e)

    recall = true_positives / actual_positives if actual_positives > 0 else 0

    print(f"Recall: {recall:.3f}")
    return recall


def main():
    args = parse_args()

    # Set the correct split
    split = args.split
    assert split == "val" or split == "test"  # GCG Evaluation has only val and test splits
    results = {}  # collected metrics -> metrics_<split>.json
    gt_dir_path = args.gt_dir_path
    if gt_dir_path is None:
        gt_dir_path = os.path.join(args.data_root, 'glamm_data/annotations/gcg_val_test/')

    gt_mask_path = f"{gt_dir_path}/{split}_gcg_coco_mask_gt.json"
    gt_cap_path = f"{gt_dir_path}/{split}_gcg_coco_caption_gt.json"

    print(f"Starting evalution on {split} split.")

    # Get the image names of the split
    all_images_ids = []
    with open(gt_cap_path, 'r') as f:
        contents = json.load(f)
        for image in contents['images']:
            all_images_ids.append(image['id'])

    # Scratch dir for the intermediate COCO-format dumps, keyed on the job id and the run dir
    # (grandparent of <run>/evals/gcg) so parallel evals never share it; removed at the end.
    _run = os.path.basename(os.path.dirname(os.path.dirname(args.prediction_dir_path))) or "run"
    tmp_dir_path = f'tmp/{os.environ.get("SLURM_JOB_ID", "nojid")}_{_run}_gcg_{split}'
    os.makedirs(tmp_dir_path, exist_ok=True)

    # Create predictions
    pred_save_path = f"{tmp_dir_path}/mask_pred_tmp_save.json"
    cap_pred_save_path = f"{tmp_dir_path}/cap_pred_tmp_save.json"
    coco_pred_file = []
    caption_pred_dict = {}
    for image_id in all_images_ids:
        prediction_path = f"{args.prediction_dir_path}/{image_id}.json"
        if not os.path.isfile(prediction_path):
            raise FileNotFoundError(
                f"Missing GCG prediction for image '{image_id}': {prediction_path}. "
                f"Run the prediction stage first or check --prediction_dir_path.")
        with open(prediction_path, 'r') as f:
            pred = json.load(f)
            bu = pred
            key = list(pred.keys())[0]
            pred = pred[key]
            try:
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            except Exception as e:
                pred = bu
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            for rle_mask in pred['pred_masks']:
                coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})

    # Save gcg_coco_predictions
    with open(pred_save_path, 'w') as f:
        json.dump(coco_pred_file, f)

    # Prepare the CAPTION predictions in COCO format
    cap_image_ids = []
    coco_cap_pred_file = []
    for image_id, values in caption_pred_dict.items():
        cap_image_ids.append(image_id)
        coco_cap_pred_file.append({"image_id": image_id, "caption": values['caption'], "labels": values['labels']})

    # Save gcg_caption_coco_predictions
    with open(cap_pred_save_path, 'w') as f:
        json.dump(coco_cap_pred_file, f)

    # -------------------------------
    # 1. Mask AP
    coco_gt = COCO(gt_mask_path)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    coco_dt = coco_gt.loadRes(pred_save_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    try:  # AP already printed by summarize(); capture must not block later metrics
        results['AP50'] = float(coco_eval.stats[1])
    except Exception:
        pass

    # -------------------------------
    # 2. Caption quality
    # The GT captions object is built outside the try: step 4 reuses it.
    coco_cap_gt = COCO(gt_cap_path)
    if 'info' not in coco_cap_gt.dataset:
        coco_cap_gt.dataset['info'] = {}
    try:
        coco_cap_result = coco_cap_gt.loadRes(cap_pred_save_path)
        coco_eval = COCOEvalCap(coco_cap_gt, coco_cap_result)
        coco_eval.params['image_id'] = coco_cap_result.getImgIds()
        coco_eval.evaluate()
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
            results[metric] = float(score)
    except Exception as e:
        print(f'[metrics_gcg][WARN] caption eval failed: {e}')

    # -------------------------------
    # 3. Mask mIoU
    coco_gt = COCO(gt_mask_path)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    results['Mean IoU (mIoU)'] = float(
        evaluate_mask_miou(coco_gt, all_images_ids, pred_save_path))

    # -------------------------------
    # 4. Recall
    results['Recall'] = float(
        evaluate_recall_with_mapping(coco_gt, coco_cap_gt, all_images_ids, pred_save_path, cap_pred_save_path,
                                     iou_threshold=0.5, text_sim_threshold=0.5))

    # machine-readable metrics summary (never fails the eval)
    try:
        _p = f"{args.prediction_dir_path}/metrics_{split}.json"
        with open(_p, "w") as _f:
            json.dump(results, _f, indent=2)
        print(f"[metrics_gcg] wrote {_p}")
    except Exception as e:
        print(f"[metrics_gcg][WARN] could not write metrics_{split}.json: {e}")

    # Clean up the tmp dir last (every metric above reads from it); a failure above leaves it in place.
    try:
        shutil.rmtree(tmp_dir_path)
    except Exception as e:
        print(f"[metrics_gcg][WARN] could not remove tmp dir {tmp_dir_path}: {e}")


if __name__ == "__main__":
    main()
