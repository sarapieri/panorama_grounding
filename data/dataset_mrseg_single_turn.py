# MRSeg-Referring-Expressions: our single-turn referring expressions, the multi-granularity
# segmentation bucket of the mixture.
# Ours: released at https://huggingface.co/datasets/Panorama-grounding/MRSeg-Referring-Expressions
# Source: the MR-Seg data of SegLLM (Wang et al., ICLR 2025, https://github.com/berkeley-hipie/segllm).
# Loader: per-source accepted/*.jsonl shards (PANORAMA_MRSEG_EXTRACT), each row an expression
# with its mask inline as COCO RLE and an image reference (image roots below) -> one
# '<p> expression </p> [SEG]' turn per expression, one mask.
import glob
import json
import os

import numpy as np
import torch
from pycocotools import mask as mask_utils

from .dataset_refseg import PanoramaRefSeg

EXTRACT_ROOT_ENV = 'PANORAMA_MRSEG_EXTRACT'

# Image roots, overridden through the environment (.env).
COCO2017 = os.environ.get('PANORAMA_COCO_TRAIN2017', '/path/to/COCO/train2017')
ADE20K_IMG = os.environ.get('PANORAMA_ADE20K_IMAGES', '/path/to/ADEChallengeData2016/images/training')
PASCAL_IMG = os.environ.get('PANORAMA_PASCAL_IMAGES', '/path/to/VOCdevkit/VOC2010/JPEGImages')
VG_IMG = os.environ.get('PANORAMA_VG_IMAGES', '/path/to/VisualGenome')

# per-source map: img_rel prefix -> image directory
SOURCES = {
    'mr_paco':    {'coco/train2017': COCO2017},
    'mr_lvis':    {'coco/train2017': COCO2017},
    'attributes': {'coco/train2017': COCO2017},
    'cocostuff':  {'coco/train2017': COCO2017},
    'ade20k':     {'ade20k/images/training': ADE20K_IMG},
    'mr_pascal':  {'pascal': PASCAL_IMG},
    'mr_vg':      {'VG_100K': f'{VG_IMG}/VG_100K', 'VG_100K_2': f'{VG_IMG}/VG_100K_2'},
}


def resolve_image(img_map, img_rel):
    """Map a sample's relative image path to a file on disk via the source's prefix map."""
    for pref, root in img_map.items():
        if img_rel.startswith(pref + '/'):
            return os.path.join(root, img_rel[len(pref) + 1:])
    # fallback: look the basename up under any mapped root
    for root in img_map.values():
        cand = os.path.join(root, os.path.basename(img_rel))
        if os.path.exists(cand):
            return cand
    return os.path.join(next(iter(img_map.values())), os.path.basename(img_rel))


class PanoramaMRSegSingleTurn(PanoramaRefSeg):
    """One MRSeg source as single-turn, single-target RES samples."""

    def __init__(self,
                 source: str,
                 extract_root: str = None,
                 num_classes_per_sample: int = 1,
                 **kwargs):
        assert source in SOURCES, f'unknown MRSeg source {source!r}'
        self.source = source
        self.img_map = SOURCES[source]
        self.extract_root = extract_root or os.environ.get(
            EXTRACT_ROOT_ENV, './data/mrseg_extract')
        # data_root must be a real dir and split_file='' so the parent skips its refs pickle;
        # load_data_list below provides the samples.
        super().__init__(data_root=self.extract_root, ann_file='', split='train',
                         split_file='', num_classes_per_sample=num_classes_per_sample, **kwargs)

    def load_data_list(self):
        files = sorted(glob.glob(f'{self.extract_root}/accepted/{self.source}_shard*.jsonl'))
        if not files:
            raise ValueError(f'No extracted files for source={self.source} '
                             f'under {self.extract_root}/accepted/')
        data_list = []
        for fp in files:
            with open(fp) as f:
                for line in f:
                    r = json.loads(line)
                    data_list.append({
                        'img_rel': r['img_rel'],
                        'img_path': resolve_image(self.img_map, r['img_rel']),
                        'phrase': r['expression'],
                        'rle': r['mask'],                    # {'size': [h, w], 'counts': str}
                    })
        if not data_list:
            raise ValueError(f'Extracted files for {self.source} are empty: {files}')
        return data_list

    def _parse_annotations(self, ann_info):
        image = self._read_image(ann_info['img_path'])
        if image is None:
            return None
        rle = ann_info['rle']
        try:
            m = mask_utils.decode({'size': rle['size'],
                                   'counts': rle['counts'].encode('ascii')})
        except Exception as e:                  # a single bad row must not kill the epoch
            print(f'[MRSeg][{self.source}] RLE decode failed ({ann_info["img_rel"]}): {e}')
            return None
        m = (np.asarray(m) > 0).astype(np.uint8)
        if int(m.sum()) == 0:
            return None
        height, width = m.shape[:2]
        ann_info.update({
            'masks': torch.from_numpy(m)[None],                      # (1, H, W)
            'seg_group_ids': torch.zeros(1, dtype=torch.long),       # single-target set of 1
            'conversations': self._build_seg_conversation([ann_info['phrase']], [False]),
            'image': ann_info['img_path'],
            'height': height,
            'width': width,
        })
        return ann_info
