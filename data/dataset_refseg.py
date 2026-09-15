# Referring expression segmentation training sets.
# Sources: RefCOCO and RefCOCO+ (Yu et al., ECCV 2016), RefCOCOg (Mao et al., CVPR 2016),
#          gRefCOCO (Liu et al., CVPR 2023, https://github.com/henghuiding/gRefCOCO),
#          PhraseCut (Wu et al., CVPR 2020, https://github.com/ChenyunWu/PhraseCutDataset).
# Loader: RefCOCO/+/g through the vendored MMDetection RefCocoDataset;
#         gRefCOCO and PhraseCut follow their official APIs. Answers are '<p> phrase </p> [SEG]',
#         with a phrase's mask set grouped through seg_group_ids.
import collections
import json
import os
import random
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils

from third_parts.mmdet.datasets.refcoco import RefCocoDataset
from .base import PanoramaBaseDataset
from .common import SEG_QUESTIONS, ANSWER_LIST, GRES_QUESTION


class PanoramaRefSeg(RefCocoDataset, PanoramaBaseDataset):
    """RefCOCO/+/g loader: several expressions per image, one turn each."""

    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='train2014/'),
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 arch_type: Literal['qwen'] = 'qwen',
                 preprocessor=None,
                 repeats: float = 1.0,
                 name: str = 'PanoramaRefSeg',
                 **kwargs):

        # Initialize RefCocoDataset
        RefCocoDataset.__init__(self,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )

        # Initialize PanoramaBaseDataset with common functionality
        PanoramaBaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )

        self.begin_str = '<image>\n'
        self.num_classes_per_sample = num_classes_per_sample

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']
        index = np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True)
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            # RefCOCO/+/g masks are polygon lists
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst["mask"]:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)
        ann_info.update({
            'masks': masks,
            'conversations': self._build_seg_conversation(phrases),
            'image': image_path
        })
        return ann_info

    def _build_seg_conversation(self, phrases, is_no_target=None):
        """One question/answer turn per phrase, the answer being '<p> phrase </p> [SEG]' (no mask
        for gRefCOCO expressions without a target)."""
        conversation = []
        _gq = getattr(self, 'gres_question', None)   # gRefCOCO prompt template, else None
        for i, phrase in enumerate(phrases):
            question = (_gq or random.choice(SEG_QUESTIONS)).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            if is_no_target is not None and is_no_target[i]:
                answer = getattr(self, 'no_target_answer', 'No target.')
            else:
                answer = random.choice(ANSWER_LIST).replace('{phrase}', phrase)
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
        return conversation

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']
        if 'seg_group_ids' in data_dict:   # per-instance-set grouping (gRefCOCO/PhraseCut), like PanoCaps
            out_data_dict['seg_group_ids'] = data_dict['seg_group_ids']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self._read_image(image_file)
            if image is None:
                return None

            # Process image using base class method
            image_data = self._process_single_image(image)
            out_data_dict.update(image_data)

            # Create image token string and get input/labels
            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
        else:
            return None   # every RES sample carries an image; refetch otherwise
        return out_data_dict

    def real_len(self):
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    # __len__ and __getitem__ are re-declared so they take precedence over RefCocoDataset's in
    # the method resolution order; keep them identical to PanoramaBaseDataset.
    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)

    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]

        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data

        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")


class PanoramaGRefSeg(PanoramaRefSeg):
    """gRefCOCO (GRES) loader, following the official grefer.py conventions.

    Reads `grefs(unc).json` instead of the RefCOCO refs pickle. One ref is one phrase whose
    ground truth is a per-instance set: one mask per `ann_id`, grouped to the phrase via
    seg_group_ids like PanoCaps. Per-annotation masks may be polygons or RLE (the gRefCOCO
    instances.json mixes both). Expressions without a target get no mask.

    Images are COCO train2014, outside the gRefCOCO folder:
        data_root   = folder holding grefs(unc).json and instances.json
        data_prefix = dict(img_path=<absolute COCO train2014 dir>)
    The image path must be absolute so the parent's prefix joining leaves it untouched.

    Ref selection: include_singles (single-target refs), include_multi (multi-target refs),
    include_no_target (no-target refs).
    """

    def __init__(self,
                 data_root,
                 grefs_file: str = 'grefs(unc).json',
                 ann_file: str = 'instances.json',
                 include_singles: bool = False,
                 include_no_target: bool = False,
                 include_multi: bool = True,
                 no_target_answer: str = 'No target.',
                 split: str = 'train',
                 **kwargs):
        # Set before super().__init__: load_data_list() runs during it.
        self.grefs_file = grefs_file
        self.include_singles = include_singles
        self.include_no_target = include_no_target
        self.include_multi = include_multi
        self.no_target_answer = no_target_answer
        # gRefCOCO uses its fixed template (common.GRES_QUESTION) instead of the random
        # SEG_QUESTIONS; the gRefCOCO eval uses the same template.
        self.gres_question = GRES_QUESTION
        # split_file='' so the parent skips its refs pickle; load_data_list provides the samples.
        super().__init__(data_root=data_root, ann_file=ann_file, split=split,
                         split_file='', **kwargs)

    @staticmethod
    def _ref_ann_ids(ref):
        a = ref['ann_id']
        return a if isinstance(a, list) else [a]

    @classmethod
    def _is_no_target(cls, ref):
        ids = cls._ref_ann_ids(ref)
        return bool(ref.get('no_target')) or ids == [-1] or all(x == -1 for x in ids)

    def load_data_list(self):
        grefs = json.load(open(os.path.join(self.data_root, self.grefs_file)))
        inst = json.load(open(os.path.join(self.data_root, self.ann_file)))
        anns = {a['id']: a for a in inst['annotations']}
        imgs = {im['id']: im for im in inst['images']}
        img_prefix = self.data_prefix['img_path']

        by_img = collections.defaultdict(list)
        for ref in grefs:
            if ref.get('split') != self.split:
                continue
            no_target = self._is_no_target(ref)
            ids = [] if no_target else self._ref_ann_ids(ref)
            if no_target:
                if not self.include_no_target:
                    continue
            elif len(ids) == 1:
                if not self.include_singles:
                    continue
            else:  # multi-target
                if not self.include_multi:
                    continue
            segs = [anns[a]['segmentation'] for a in ids if a != -1 and a in anns]
            by_img[ref['image_id']].append({
                'segs': segs,
                'no_target': no_target,
                'sents': [s['sent'] for s in ref['sentences']],
            })

        data_list = []
        for image_id, ref_list in by_img.items():
            im = imgs[image_id]
            data_list.append({
                'img_path': os.path.join(img_prefix, im['file_name']),
                'img_id': image_id,
                'height': im['height'],
                'width': im['width'],
                'instances': ref_list,
            })
        if not data_list:
            raise ValueError(
                f'No gRefCOCO samples in split "{self.split}" '
                f'(include_singles={self.include_singles}, '
                f'include_no_target={self.include_no_target}).')
        return data_list

    @staticmethod
    def _seg_to_mask(seg, height, width):
        """Decode one annotation's segmentation (polygon or RLE) -> (H,W) uint8."""
        if isinstance(seg, list):
            if not seg:
                return np.zeros((height, width), dtype=np.uint8)
            if isinstance(seg[0], list):            # polygon(s)
                m = mask_utils.decode(mask_utils.merge(
                    mask_utils.frPyObjects(seg, height, width)))
            elif isinstance(seg[0], dict):          # list of RLE dicts
                rle = seg
                for r in rle:
                    if not isinstance(r['counts'], bytes):
                        r['counts'] = r['counts'].encode()
                m = mask_utils.decode(rle)
                if m.ndim == 3:
                    m = m.sum(axis=2)
            else:                                   # single flat polygon
                m = mask_utils.decode(mask_utils.merge(
                    mask_utils.frPyObjects([seg], height, width)))
        elif isinstance(seg, dict):                 # single RLE
            rle = seg
            if isinstance(rle['counts'], list):
                rle = mask_utils.frPyObjects(rle, height, width)
            elif not isinstance(rle['counts'], bytes):
                rle = dict(rle)
                rle['counts'] = rle['counts'].encode()
            m = mask_utils.decode(rle)
        else:
            return np.zeros((height, width), dtype=np.uint8)
        return (m > 0).astype(np.uint8)

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        instances = ann_info['instances']
        idxs = np.random.choice(len(instances), self.num_classes_per_sample, replace=True)
        phrases, all_masks, group_ids, is_no_target = [], [], [], []
        g = 0   # group index over phrases that emit a [SEG]
        for i in idxs:
            ref = instances[i]
            if ref.get('no_target'):
                # Expression without a target: no mask, and g does not advance, so
                # seg_group_ids stays aligned to the [SEG] order.
                phrase = random.choice(ref['sents']).lower()
                if phrase and phrase[-1] == '.':
                    phrase = phrase[:-1]
                phrases.append(phrase)
                is_no_target.append(True)
                continue
            # One mask per ann_id (each unions that annotation's own polygons/RLE), grouped to
            # the phrase via seg_group_ids.
            inst_masks = [m for m in (self._seg_to_mask(seg, height, width) for seg in ref['segs'])
                          if m.sum() > 0]
            if not inst_masks:                       # positive ref that decoded empty -> skip phrase
                continue
            phrase = random.choice(ref['sents']).lower()
            if phrase and phrase[-1] == '.':
                phrase = phrase[:-1]
            phrases.append(phrase)
            is_no_target.append(False)
            for m in inst_masks:
                all_masks.append(m)
                group_ids.append(g)
            g += 1
        if not phrases:
            return None

        if all_masks:
            masks = torch.stack([torch.from_numpy(m) for m in all_masks], dim=0)
        else:                                        # no target in this image: empty mask set
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        grp_t = torch.as_tensor(group_ids, dtype=torch.long)
        ann_info.update({
            'masks': masks,
            'seg_group_ids': grp_t,
            'conversations': self._build_seg_conversation(phrases, is_no_target),
            'image': image_path,
        })
        return ann_info


class PanoramaPhraseCut(PanoramaRefSeg):
    """PhraseCut (VGPhraseCut_v0) loader, following the official PhraseCut API.

    Open-vocabulary phrase to region on Visual Genome images. One phrase's ground truth is a
    per-instance set: one mask per instance (that instance's polygons unioned), grouped to the
    phrase via seg_group_ids like PanoCaps. Masks are rasterized with PIL, mirroring PhraseCut's
    data_transfer.polygons_to_mask, so they match the benchmark's own definition.

    Images live in two Visual Genome folders (VG_100K, VG_100K_2); each image_id is resolved
    through a filename index built once at init.

    Config:
        data_root       = PhraseCut folder (holds VGPhraseCut_v0/refer_*.json)
        vg_images_root  = Visual Genome root containing VG_100K/ and VG_100K_2/
        refer_file      = 'VGPhraseCut_v0/refer_train.json'
    """

    def __init__(self,
                 data_root,
                 vg_images_root,
                 refer_file: str = 'VGPhraseCut_v0/refer_train.json',
                 **kwargs):
        # Set before super().__init__: load_data_list() runs during it.
        self.vg_images_root = vg_images_root
        self.refer_file = refer_file
        # split_file='' so the parent skips its refs pickle; load_data_list provides the samples.
        super().__init__(data_root=data_root, ann_file=refer_file, split_file='', **kwargs)

    def load_data_list(self):
        tasks = json.load(open(os.path.join(self.data_root, self.refer_file)))
        # filename -> full path across the two VG folders
        img_index = {}
        for sub in ('VG_100K', 'VG_100K_2'):
            d = os.path.join(self.vg_images_root, sub)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    img_index[fn] = os.path.join(d, fn)

        by_img, miss = collections.defaultdict(list), 0
        for t in tasks:
            polys = t.get('Polygons') or []
            if not polys:
                continue
            path = img_index.get('%s.jpg' % t['image_id'])
            if path is None:
                miss += 1
                continue
            # polys = list[instance] -> list[polygon]; one mask per instance
            by_img[(t['image_id'], path)].append({'phrase': t['phrase'], 'insts': polys})

        data_list = [{'img_path': path, 'img_id': image_id, 'instances': refs}
                     for (image_id, path), refs in by_img.items()]
        if miss:
            print(f'[PanoramaPhraseCut] {miss} tasks skipped (image not in VG_100K/VG_100K_2)')
        if not data_list:
            raise ValueError('No PhraseCut samples (check refer_file / vg_images_root).')
        return data_list

    @staticmethod
    def _polygons_to_mask(polygons, height, width):
        """PIL rasterization of one instance's polygon list -> (H,W) uint8 union of its parts
        (mirrors PhraseCutDataset/utils/data_transfer.py:polygons_to_mask)."""
        m = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            p = [(int(x), int(y)) for x, y in polygon]
            canvas = Image.new('L', (width, height), 0)
            ImageDraw.Draw(canvas).polygon(p, outline=1, fill=1)
            m |= np.asarray(canvas, dtype=np.uint8)
        return m

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        instances = ann_info['instances']
        idxs = np.random.choice(len(instances), self.num_classes_per_sample, replace=True)
        phrases, all_masks, group_ids = [], [], []
        g = 0
        for i in idxs:
            ref = instances[i]
            # One mask per instance, grouped to the phrase via seg_group_ids.
            inst_masks = [m for m in (self._polygons_to_mask(polys, height, width)
                                      for polys in ref['insts']) if m.sum() > 0]
            if not inst_masks:
                continue
            phrase = ref['phrase'].lower()
            if phrase and phrase[-1] == '.':
                phrase = phrase[:-1]
            phrases.append(phrase)
            for m in inst_masks:
                all_masks.append(m)
                group_ids.append(g)
            g += 1
        if not all_masks:
            return None

        masks = torch.stack([torch.from_numpy(m) for m in all_masks], dim=0)
        grp_t = torch.as_tensor(group_ids, dtype=torch.long)
        ann_info.update({
            'masks': masks,
            'seg_group_ids': grp_t,
            'conversations': self._build_seg_conversation(phrases),
            'image': image_path,
        })
        return ann_info
