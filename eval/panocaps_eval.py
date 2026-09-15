"""PanoCaps inference.

Writes one json per image (caption, phrases, per-phrase mask counts, one RLE mask per instance).
Metrics are computed by eval/metrics_panocaps.py; launched by scripts/eval_panocaps.sh.
"""
import argparse
import datetime
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import torch
import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.utils import logging as hf_logging

from data.common import PANOCAPS_QUESTIONS
from .utils import (_init_dist_pytorch, get_dist_info, get_rank, barrier,
                    collect_results_cpu, mask_to_rle_pytorch, coco_encode_rle)

# Silence the per-sample generation warnings of transformers.
hf_logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="PanoCaps inference.")
    parser.add_argument(
        "model_path",
        type=str,
        help="Converted HF checkpoint directory.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing PanoCaps images."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./pano_caps_preds/",
        help="Directory to save PanoCaps predictions as JSON files."
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
        help="Distributed launcher: 'none' or 'pytorch' (torchrun).",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed inference (set automatically by launcher)."
    )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class PanoCapsInferenceDataset:
    def __init__(
        self,
        image_folder,
        save_dir=None,
        ):
        self.image_folder = image_folder
        self.images = sorted(os.listdir(image_folder))

        # filter out images that already have predictions saved
        if save_dir is not None:
            self.save_dir = save_dir
            existing = {Path(f).stem for f in os.listdir(save_dir)}
            self.images = [
                img for img in self.images
                if Path(img).stem not in existing
            ]

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        # PanoCaps eval prompt (one of the training templates); PANORAMA_PANOCAPS_EVAL_PROMPT
        # overrides it (empty = default).
        return os.environ.get('PANORAMA_PANOCAPS_EVAL_PROMPT', '').strip() or PANOCAPS_QUESTIONS[2]

    def __getitem__(self, index):
        data_dict = {}
        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file

        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = "<image>\n" + questions

        data_dict['img_id'] = image_file

        return data_dict


def _eval_nccl_timeout():
    """NCCL collective timeout for eval (default 20 min, PANORAMA_EVAL_NCCL_TIMEOUT_MIN overrides):
    a slow rank whose prediction has hundreds of masks must not trip the watchdog."""
    return datetime.timedelta(minutes=int(os.environ.get("PANORAMA_EVAL_NCCL_TIMEOUT_MIN", "20")))


def main():
    args = parse_args()

    if args.launcher == "none":
        rank, world_size = 0, 1
    elif args.launcher == "pytorch":
        _init_dist_pytorch("nccl", timeout=_eval_nccl_timeout())
        rank, world_size = get_dist_info()
    else:
        raise ValueError(f"Unsupported launcher: {args.launcher}")

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

    if get_rank() == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    barrier()  # make sure all ranks see it (no-op when not distributed)

    dataset = PanoCapsInferenceDataset(
        image_folder=args.image_dir,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    start = per_rank_samples * rank
    end = min(n_samples, per_rank_samples * (rank + 1))
    per_rank_ids = range(start, end)

    with torch.no_grad():
        for idx in tqdm.tqdm(per_rank_ids):
            data_batch = dataset[idx]

            image_file = data_batch.pop('image_file')
            img_id = data_batch.pop('img_id')

            w, h = data_batch['image'].size

            pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer, processor=processor)

            # The model returns the flat per-instance masks plus the number of instances selected
            # per phrase, so phrase_seg_counts comes from the model's selection, not from the
            # [SEG] count.
            inst_masks = pred_dict['prediction_instance_masks']
            phrase_seg_counts = pred_dict['prediction_phrase_seg_counts']
            if len(inst_masks) == 0:
                pred_masks = torch.zeros((0, h, w), dtype=torch.bool)
            else:
                pred_masks = torch.stack(
                    [torch.tensor(mask) for mask in inst_masks],
                    dim=0,
                )[:, 0]

            process_and_save_output(
                args.save_dir,
                image_file,
                pred_dict['prediction'],
                pred_masks,
                phrase_seg_counts=phrase_seg_counts,
            )
            results.append(pred_dict['prediction'])

    job_id = (
        os.getenv("SLURM_JOB_ID")
        or os.getenv("PBS_JOBID")
        or os.getenv("LSB_JOBID")
        or os.getenv("JOB_ID")
        or str(os.getpid())
    )
    # Call collect unconditionally: collect_results_cpu runs collective barriers,
    # so skipping it on a rank with empty results would deadlock the other ranks.
    collect_results_cpu(results, len(dataset), tmpdir=f"./eval_tmp_{job_id}")


def process_and_save_output(output_dir, image_name, text_output, pred_masks, phrase_seg_counts=None):
    os.makedirs(output_dir, exist_ok=True)

    text = text_output.replace("\n", " ")

    # strict tokens
    p_re   = re.compile(r"<p>(.*?)</p>", re.DOTALL | re.IGNORECASE)  # exact <p>...</p>
    seg_re = re.compile(r"\[SEG\]")                                  # exact [SEG]

    # collect only correct phrases: <p>..</p> followed by >=1 [SEG] before next <p>
    phrases = []
    segments = []

    p_matches = list(p_re.finditer(text))
    has_seg = bool(seg_re.search(text))
    if not has_seg:
        # No [SEG] in the output: every <p>...</p> is a phrase (the [SEG] gate below would drop
        # them all); `segments` is rebuilt from the model's phrase_seg_counts just below.
        phrases = [" ".join(m.group(1).split()).strip() for m in p_matches]
    else:
        # Keep a phrase only if at least one [SEG] follows it (before the next <p>).
        for i, m in enumerate(p_matches):
            start = m.end()
            end   = p_matches[i+1].start() if i+1 < len(p_matches) else len(text)
            window = text[start:end]
            nseg = len(seg_re.findall(window))
            if nseg >= 1:
                phrase = " ".join(m.group(1).split()).strip()
                phrases.append(phrase)
                phrase_idx = len(phrases) - 1
                for j in range(nseg):
                    segments.append({"phrase_idx": phrase_idx})

    # Instance-level: the model supplies per-phrase instance counts; rebuild `segments` from
    # them (aligned with the parsed phrases). Masks pair in order below.
    if phrase_seg_counts is not None:
        n = min(len(phrases), len(phrase_seg_counts))
        phrases = phrases[:n]
        segments = []
        for i in range(n):
            for _ in range(int(phrase_seg_counts[i])):
                segments.append({"phrase_idx": i})

    # drop tags and [SEG], collapse whitespace
    cleaned_str = re.sub(r"<.*?>", "", text).replace("[SEG]", "")
    cleaned_str = " ".join(cleaned_str.split()).strip()

    # masks to RLE
    pred_masks_tensor = pred_masks.detach().cpu()
    if pred_masks_tensor.dtype not in (torch.uint8, torch.bool):
        pred_masks_tensor = (pred_masks_tensor > 0).to(torch.uint8)

    uncompressed = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks_all = [coco_encode_rle(m) for m in uncompressed]

    # pair masks to segments strictly in order; drop tails on either side
    keep_k = min(len(segments), len(rle_masks_all))
    kept_masks = rle_masks_all[:keep_k]
    kept_segments = segments[:keep_k]

    # recompute kept per-phrase counts
    counts = Counter(s["phrase_idx"] for s in kept_segments)
    kept_phrases, kept_seg_counts = [], []
    for i, ph in enumerate(phrases):
        c = counts.get(i, 0)
        if c > 0:
            kept_phrases.append(ph)
            kept_seg_counts.append(c)

    # save
    stem = Path(image_name).stem
    result = {
        "image_id": stem,
        "caption": cleaned_str,
        "phrases": kept_phrases,                # only correct ones with masks kept
        "phrase_seg_counts": kept_seg_counts,
        "pred_masks": kept_masks,               # COCO RLE, length == sum(kept_seg_counts)
    }
    out_path = Path(output_dir) / f"{stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    main()