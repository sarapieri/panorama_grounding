"""GCG inference on GranD-f (val and test images).

Writes one json per image with the caption, its <p> phrases and one RLE mask per [SEG].
Metrics are computed by eval/metrics_gcg.py; launched by scripts/eval_gcg.sh.
"""
import argparse
import datetime
import json
import math
import os
import re
import shutil
from multiprocessing import Pool

import torch
import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers.utils import logging as hf_logging

from .utils import (_init_dist_pytorch, get_dist_info, collect_results_cpu,
                    mask_to_rle_pytorch, coco_encode_rle)

# Silence the per-sample generation warnings of transformers.
hf_logging.set_verbosity_error()

# GranD-f evaluation prompt (the benchmark protocol; the training templates ask for a detailed
# description instead).
GCG_EVAL_QUESTION = ("Could you please give me a brief description of the image? Please respond "
                     "with interleaved segmentation masks for the corresponding parts of the answer.")


def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument('--save_dir', default='./gcg_pred/',
                        help='output folder for the per-image json predictions')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--data_root', default='./data',
                        help='root folder holding glamm_data/ (PANORAMA_DATA_ROOT)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class GCGInferenceDataset:
    def __init__(self,
                 image_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder

        self.images = os.listdir(image_folder)

        if save_dir is not None:
            # skip images that already have a prediction file (resume)
            self.save_dir = save_dir
            done = {f[:-5] for f in os.listdir(self.save_dir)}
            self.images = [item for item in self.images if item[:-4] not in done]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data_dict = {}
        questions = GCG_EVAL_QUESTION
        image_file = self.images[index]
        data_dict['image_file'] = image_file

        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = "<image>\n" + questions

        data_dict['img_id'] = image_file
        return data_dict

def main():
    args = parse_args()

    image_folder = os.path.join(args.data_root, 'glamm_data/images/grandf/val_test/')

    if args.launcher != 'none':
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # Use a portion of CPU cores for multiprocessing pool
    # to avoid overwhelming the system, especially in a multi-rank setup.
    num_workers = max(1, os.cpu_count() // (world_size * 2))
    pool = Pool(processes=num_workers)

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

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Save to dir {args.save_dir}")
    dataset = GCGInferenceDataset(
        image_folder=image_folder,
        save_dir=args.save_dir,
    )

    # Per-job temp dir for collect_results_cpu, so concurrent evals do not share part files.
    job_id = (os.getenv('SLURM_JOB_ID') or os.getenv('PBS_JOBID') or os.getenv('LSB_JOBID')
              or os.getenv('JOB_ID') or str(os.getpid()))
    gcg_tmpdir = f'./eval_tmp_{job_id}_gcg'
    if rank == 0:
        # Clean up tmp dir from previous runs
        if os.path.exists(gcg_tmpdir):
            shutil.rmtree(gcg_tmpdir)

    if len(dataset) == 0:
        if rank == 0:
            print("All images have been processed. Skipping inference.")
        pool.close()
        pool.join()
        return

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size

        with torch.no_grad():
            pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer, processor=processor)
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            masks = [torch.from_numpy(m) for m in pred_dict['prediction_masks']]
            prediction['prediction_masks'] = torch.stack(masks, dim=0)[:, 0]

        # Asynchronously process and save the output
        pool.apply_async(process_and_save_output, args=(
            args.save_dir,
            prediction['image_file'],
            pred_dict['prediction'],
            prediction['prediction_masks'].cpu()  # Pass tensor on CPU
        ))
        results.append(pred_dict['prediction'])

    # Wait for all file saving tasks to complete
    pool.close()
    pool.join()

    # Call collect unconditionally: collect_results_cpu runs collective barriers,
    # so skipping it on a rank with empty results would deadlock the other ranks.
    collect_results_cpu(results, len(dataset), tmpdir=gcg_tmpdir)


def process_and_save_output(output_dir, image_name, text_output, pred_masks_tensor):
    os.makedirs(output_dir, exist_ok=True)

    text_output = text_output.replace("\n", "").replace("  ", " ")

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)


if __name__ == '__main__':
    main()
