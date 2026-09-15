"""GroundingSuite-Eval inference.

Writes one json per expression (union mask as RLE, box, class id) under <save_dir>/preds/, then
merges them into predictions.jsonl for eval/metrics_groundingsuite.py; launched by
scripts/eval_gseval.sh.
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.utils import logging as hf_logging

from .utils import (_init_dist_pytorch, get_dist_info, get_rank,
                    collect_results_cpu, find_seg_indices, mask_to_coco_rle)

# Silence the per-sample generation warnings of transformers.
hf_logging.set_verbosity_error()


def parse_args():
    p = argparse.ArgumentParser(description='GroundingSuite-Eval (PANORAMA)')
    p.add_argument('model_path', help='HF model dir (converted PANORAMA checkpoint).')
    p.add_argument('--gseval-root', default=os.environ.get('PANORAMA_GSEVAL_ROOT'),
                   help='Root with GroundingSuite-Eval.jsonl + unlabeled2017/ (default $PANORAMA_GSEVAL_ROOT).')
    p.add_argument('--gt-file', default=None,
                   help='Override; default <gseval-root>/GroundingSuite-Eval.jsonl.')
    p.add_argument('--image-dir', default=None,
                   help='Override; default <gseval-root> (the dir that CONTAINS unlabeled2017/).')
    p.add_argument('--save-dir', default=None,
                   help='Output dir; default <model_dir>/evals/groundingsuite.')
    p.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = p.parse_args()
    if args.gt_file is None:
        if not args.gseval_root:
            p.error('need --gseval-root (or $PANORAMA_GSEVAL_ROOT) or an explicit --gt-file')
        args.gt_file = os.path.join(args.gseval_root, 'GroundingSuite-Eval.jsonl')
    if args.image_dir is None:
        if not args.gseval_root:
            p.error('need --gseval-root (or $PANORAMA_GSEVAL_ROOT) or an explicit --image-dir')
        args.image_dir = args.gseval_root
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class GSEvalDataset(torch.utils.data.Dataset):
    """Reads GroundingSuite-Eval.jsonl; yields one item per expression in the format
    predict_forward consumes ('image' PIL + 'text' prompt) plus bookkeeping (idx/class_id/...)."""

    # PANORAMA's RES prompt (matches eval/utils/res_dataset.py::get_questions).
    PROMPT = "<image>\n Please segment {} in this image."

    def __init__(self, gt_file, image_dir):
        self.image_dir = image_dir
        self.items = []
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        image_rel = d['image_path']
        image = Image.open(os.path.join(self.image_dir, image_rel)).convert('RGB')
        w, h = image.size
        return {
            'image': image,
            'text': self.PROMPT.format(d['caption']),   # the caption field is the expression
            'idx': d['idx'],
            'image_path': image_rel,                    # relative path
            'class_id': d['class_id'],
            'ori_size': (h, w),
        }


def _union_prediction(pred, ori_size):
    """Reproduce refcoco_eval's read-out: union the [SEG]-selected masks (else all masks) into one
    (1,H,W) blob; empty -> zero mask. Returns (rle_dict, box[x1,y1,x2,y2])."""
    pred_mask = pred['prediction_masks']          # list of (1,H,W)
    pred_text = pred['prediction']
    cleaned = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()
    seg_idx = find_seg_indices(cleaned)
    if len(seg_idx) == 0:
        seg_idx = find_seg_indices(pred_text)

    final = None
    if len(pred_mask) > 0:
        if len(seg_idx) > 0:
            selected = [pred_mask[j] for j in seg_idx if j < len(pred_mask)]
        else:
            selected = pred_mask                  # no [SEG] -> union every predicted mask
        if len(selected) > 0:
            final = selected[0].copy()
            for m in selected[1:]:
                final = final | m

    if final is None:
        h, w = ori_size
        zero = np.zeros((1, h, w), dtype=np.uint8)
        return mask_to_coco_rle(zero)[0], [0, 0, 0, 0]

    final = final.astype(np.uint8)
    try:
        box = torchvision.ops.masks_to_boxes(torch.from_numpy(final)).cpu().numpy().tolist()[0]
    except Exception:
        box = [0, 0, 0, 0]
    return mask_to_coco_rle(final)[0], box


def main():
    args = parse_args()

    if args.launcher == 'pytorch':
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.normpath(args.model_path)), 'evals', 'groundingsuite')
    pred_dir = os.path.join(save_dir, 'preds')           # per-item {idx}.json (resume by exists)
    os.makedirs(pred_dir, exist_ok=True)

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=os.environ.get('PANORAMA_ATTN_IMPL', 'flash_attention_2'),
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    dataset = GSEvalDataset(args.gt_file, args.image_dir)
    if rank == 0:
        print(f"[gseval] {len(dataset)} items | save_dir={save_dir}", flush=True)

    # Strided disjoint shard per rank: DistributedSampler would pad the dataset and make two ranks
    # write the same {idx}.json.
    sampler = list(range(rank, len(dataset), world_size)) if world_size > 1 else None
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=1, num_workers=8,
        pin_memory=False, collate_fn=lambda x: x[0])

    results = []
    for data in tqdm.tqdm(dataloader, disable=(rank != 0)):
        idx = int(data['idx'])
        out_path = os.path.join(pred_dir, f'{idx}.json')
        if os.path.exists(out_path):                     # resume
            continue
        with torch.no_grad():
            pred = model.predict_forward(
                image=data['image'], text=data['text'], tokenizer=tokenizer, processor=processor)
        rle, box = _union_prediction(pred, data['ori_size'])
        prediction = {
            'idx': idx,
            'image_path': data['image_path'],
            'predicted_box': box,
            'predicted_segmentation': rle,
            'class_id': data['class_id'],
            'pred_text': pred.get('prediction', ''),   # for the visualizer; the scorer ignores extra keys
        }
        # Atomic write, so resume and the merge never see a partially written file.
        tmp_path = f'{out_path}.tmp.{rank}'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f)
        os.replace(tmp_path, out_path)
        results.append(idx)

    # Barrier across ranks before the merge; the gathered list is unused, the per-item jsons are
    # the source of truth.
    job_id = (os.getenv('SLURM_JOB_ID') or os.getenv('PBS_JOBID') or os.getenv('LSB_JOBID')
              or os.getenv('JOB_ID') or str(os.getpid()))
    collect_results_cpu(results, len(dataset), tmpdir=f'./eval_tmp_{job_id}')

    # rank 0 merges the per-item jsons into one predictions.jsonl for eval/metrics_groundingsuite.py.
    if get_rank() == 0:
        merged = os.path.join(save_dir, 'predictions.jsonl')
        files = sorted(glob.glob(os.path.join(pred_dir, '*.json')),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        n = 0
        with open(merged, 'w', encoding='utf-8') as out:
            for fp in files:
                try:
                    with open(fp, encoding='utf-8') as f:
                        obj = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue                             # skip a crashed partial file
                out.write(json.dumps(obj, ensure_ascii=False) + '\n')
                n += 1
        print(f"[gseval] merged {n} predictions -> {merged}", flush=True)
        if n != len(dataset):
            print(f"[gseval][WARN] {n} predictions but {len(dataset)} GT items "
                  f"(missing images or crashed items).", flush=True)


if __name__ == '__main__':
    main()
