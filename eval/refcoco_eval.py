"""RefCOCO/+/g and gRefCOCO inference and metrics.

Per-rank prediction jsonl under <model_dir>/evals/refcoco/preds/ (resumable), metrics printed and
written as json; launched by scripts/eval_refcoco.sh and scripts/eval_gres.sh.
"""
import argparse
import copy
import datetime
import glob
import json
import os
import re

import numpy as np
import torch
import tqdm
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.utils import logging as hf_logging

from .utils import (_init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu,
                    find_seg_indices, mask_to_coco_rle)
from .utils.res_dataset import RESDataset, DATASETS_ATTRIBUTES

# Silence the per-sample generation warnings of transformers.
hf_logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser(description='Referring-expression segmentation eval')
    parser.add_argument('model_path', help='converted HF checkpoint directory')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip samples already saved in the per-rank pred files and append new ones.')

    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--data_root', default='./data',
                        help='root folder holding glamm_data/ and ref_seg/ (PANORAMA_DATA_ROOT)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class _IndexedDataset(torch.utils.data.Dataset):
    """Wrap a dataset so each item carries its global index (for save/resume keying)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        d = self.ds[i]
        d['sample_index'] = int(i)
        return d


def main():
    args = parse_args()

    image_folder = os.path.join(args.data_root, 'glamm_data/images/coco2014/train2014/')
    data_path = os.path.join(args.data_root, 'ref_seg/')
    if args.dataset == 'grefcoco':
        # gRefCOCO annotations may live outside ref_seg/ (PANORAMA_GREFCOCO_ROOT = <parent>/grefcoco);
        # G_REFER appends /grefcoco itself, so pass the parent.
        _gref = os.environ.get('PANORAMA_GREFCOCO_ROOT')
        if _gref:
            data_path = os.path.dirname(_gref.rstrip('/'))

    if args.launcher != 'none':
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=os.environ.get('PANORAMA_ATTN_IMPL', 'flash_attention_2'),
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    # Qwen3-VL processor (tokenizer and image processor).
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=args.dataset,
        data_path=data_path,
        split=args.split,
    )

    # Save/resume: one prediction jsonl per (dataset, split, rank) next to the model.
    resume = args.resume
    pred_dir = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), 'evals', 'refcoco', 'preds')
    os.makedirs(pred_dir, exist_ok=True)
    rank_path = os.path.join(pred_dir, f'preds_{args.dataset}_{args.split}_rank{rank}.jsonl')

    # Resume: load every rank's file into one cache keyed by the global sample index (robust to
    # a changed world size). A line counts as done only if it parses and has one prediction per query.
    cache = {}
    if resume:
        for p in glob.glob(os.path.join(pred_dir, f'preds_{args.dataset}_{args.split}_rank*.jsonl')):
            with open(p, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # crashed partial trailing line
                    if 'sample_index' not in e or 'prediction_masks' not in e:
                        continue
                    if len(e['prediction_masks']) != len(e.get('texts', e['prediction_masks'])):
                        continue  # incomplete: not every query was written
                    cache[int(e['sample_index'])] = e
        if rank == 0:
            print(f"[refcoco_eval][resume] {len(cache)} cached samples for {args.dataset}/{args.split}")

    # Expose the global dataset index on each item to key the cache and the saves on it.
    indexed = _IndexedDataset(dataset)

    sampler = torch.utils.data.DistributedSampler(
        indexed,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        indexed,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )

    rank_file = open(rank_path, 'a' if resume else 'w', encoding='utf-8')

    results = []
    for data_batch in tqdm.tqdm(dataloader):
        sample_index = int(data_batch.pop('sample_index'))
        if sample_index in cache:
            e = cache[sample_index]
            results.append({'img_id': e['img_id'], 'gt_masks': e['gt_masks'],
                            'prediction_masks': e['prediction_masks']})
            continue
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_coco_rle(prediction['gt_masks'].cpu().numpy())
        texts = data_batch['text']
        img_metas = {'img_id': data_batch['img_id'],
                     'image_path': data_batch['image_path']}
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['image_path'], data_batch['text']
        pred_masks = []
        pred_texts = []
        pred_n_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            with torch.no_grad():
                pred = model.predict_forward(**_data_batch, tokenizer=tokenizer, processor=processor)
            pred_mask = pred['prediction_masks']
            pred_text = pred['prediction']
            pred_texts.append(pred_text)

            # whether the prediction is empty
            if len(pred_mask) == 0:
                pred_masks.append(None)
                pred_n_masks.append(None)
                continue
            else:
                # List (1, h, w) -> (n, h, w)
                pred_n_masks.append(np.concatenate(pred_mask, axis=0))

                cleaned_pred_text = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()

                answer_seg_idx = find_seg_indices(cleaned_pred_text)

                if len(answer_seg_idx) > 0:
                    selected_masks = [pred_mask[idx] for idx in answer_seg_idx if idx < len(pred_mask)]

                    if len(selected_masks) > 0:
                        final_mask = selected_masks[0]
                        for mask in selected_masks[1:]:
                            final_mask = final_mask | mask
                        _ret_mask = mask_to_coco_rle(final_mask)
                        pred_masks.append(_ret_mask)
                    else:
                        print("No valid masks for answer [SEG] tokens")
                        pred_masks.append(None)
                else:
                    # No [SEG] in the answer text: RefCOCO expects a single mask for the one
                    # expression, so union all predicted phrase masks.
                    print(f"No [SEG] token in answer -> union all {len(pred_mask)} predicted phrase mask(s) (no-SEG path)")
                    final_mask = pred_mask[0]
                    for mask in pred_mask[1:]:
                        final_mask = final_mask | mask
                    pred_masks.append(mask_to_coco_rle(final_mask))

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)
        try:
            _iid = prediction['img_id']
            _iid = _iid.item() if hasattr(_iid, 'item') else _iid
            rank_file.write(json.dumps({
                'sample_index': sample_index, 'img_id': _iid,
                'image_path': str(img_metas['image_path']),
                'gt_masks': prediction['gt_masks'], 'prediction_masks': pred_masks,
                'texts': list(texts), 'pred_texts': pred_texts,
            }, ensure_ascii=False) + '\n')
            rank_file.flush()
        except Exception as _e:
            print(f"[refcoco_eval][WARN] could not save sample {sample_index}: {_e}")
    rank_file.close()

    # Per-job temp dir for collect_results_cpu, so concurrent evals do not share part files;
    # dataset and split are appended because one job runs several sequential collects.
    job_id = (os.getenv('SLURM_JOB_ID') or os.getenv('PBS_JOBID') or os.getenv('LSB_JOBID')
              or os.getenv('JOB_ID') or str(os.getpid()))
    tmpdir = f'./eval_tmp_{job_id}_{args.dataset}_{args.split}'
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results)
        print(metric)
        # Machine-readable metrics summary under <model_dir>/evals/refcoco/ (never fails the eval).
        try:
            _save = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), 'evals', 'refcoco')
            os.makedirs(_save, exist_ok=True)
            _data = dict(metric) if isinstance(metric, dict) else {"raw": str(metric)}
            _data = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in _data.items()}
            _p = os.path.join(_save, f"metrics_{args.dataset}_{args.split}.json")
            with open(_p, "w") as _f:
                json.dump(_data, _f, indent=2)
            print(f"[refcoco_eval] wrote {_p}")
        except Exception as e:
            print(f"[refcoco_eval][WARN] could not write metrics json: {e}")
        # Merge the rank files into one predictions jsonl per (dataset, split), one line per
        # expression in the visualizer schema, then delete the rank files (a crashed eval never
        # reaches this block, so its caches survive for --resume).
        try:
            _seen, _rows = set(), []
            _rank_files = sorted(glob.glob(os.path.join(
                pred_dir, f'preds_{args.dataset}_{args.split}_rank*.jsonl')))
            for _rp in _rank_files:
                with open(_rp, encoding='utf-8') as _f:
                    for _line in _f:
                        _line = _line.strip()
                        if not _line:
                            continue
                        try:
                            _e = json.loads(_line)
                        except json.JSONDecodeError:
                            continue  # crashed partial trailing line
                        _si = _e.get('sample_index')
                        if _si is None or _si in _seen:
                            continue
                        _seen.add(_si)
                        _rows.append(_e)
            _rows.sort(key=lambda r: r['sample_index'])

            def _strip_special(s):
                # Visualizer text: remove chat/image tags and collapse whitespace; the
                # <p>/</p>/[SEG] tags are kept.
                for _pat in (r'<\|im_end\|>', r'<\|im_start\|>', r'<image>', r'</image>'):
                    s = re.sub(_pat, ' ', s)
                return ' '.join(s.split()).strip()

            _mp = os.path.join(pred_dir, f'preds_{args.dataset}_{args.split}.jsonl')
            _n = 0
            with open(_mp, 'w', encoding='utf-8') as _f:
                for _r in _rows:
                    _texts = _r.get('texts', [])
                    _gts = _r.get('gt_masks', [])
                    _pms = _r.get('prediction_masks', [])
                    _pts = _r.get('pred_texts', [])
                    _ipath = _r.get('image_path', '')
                    for _i, _q in enumerate(_texts):
                        _pt = _pts[_i] if _i < len(_pts) else ''
                        _pm = _pms[_i] if _i < len(_pms) else None
                        # prediction_masks[i] is a list of rle dicts (one per (1,H,W) union mask);
                        # normalize so pred_masks is always a flat list, as the visualizer expects.
                        if _pm is None:
                            _pm = []
                        elif isinstance(_pm, dict):
                            _pm = [_pm]
                        else:
                            _pm = list(_pm)
                        # Visualizer schema + provenance extras (query/pred_text/phrases/
                        # sample_index); the extra keys are ignored by the viewer.
                        _f.write(json.dumps({
                            'uid': f"{_r['img_id']}:{_i}",
                            'image_id': str(_r['img_id']),
                            'image_file': os.path.basename(_ipath),
                            'query_index': _i,
                            'ann_id': None,
                            'caption': _strip_special(f"{_q}\n{_pt}"),
                            'gt_mask': _gts[_i] if _i < len(_gts) else None,
                            'phrase_seg_counts': [len(_pm)],
                            'pred_masks': _pm,
                            'query': _q,
                            'pred_text': _pt,
                            'phrases': [s.strip() for s in
                                        re.findall(r'<p>(.*?)</p>', _pt, flags=re.DOTALL)],
                            'sample_index': _r['sample_index'],
                        }, ensure_ascii=False) + '\n')
                        _n += 1
            print(f"[refcoco_eval] merged {_n} expressions ({len(_rows)} images) -> {_mp}")
            for _rp in _rank_files:
                os.remove(_rp)
            print(f"[refcoco_eval] removed {len(_rank_files)} rank cache files")
        except Exception as e:
            print(f"[refcoco_eval][WARN] could not merge prediction jsonls: {e}")


if __name__ == '__main__':
    main()
