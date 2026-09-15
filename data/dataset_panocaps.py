# PanoCaps: our human-annotated panoptic grounded captioning benchmark, train split.
# Ours: released at https://huggingface.co/datasets/Panorama-grounding/PanoCaps
# Loader: COCO-format caption + instance-mask json -> one '<p> phrase </p> [SEG]' per phrase,
# with the phrase's mask set grouped through seg_group_ids.
import copy
import os
import re
import random
from typing import Literal, Dict, List, Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from .base import PanoramaBaseDataset
from .common import PANOCAPS_QUESTIONS


def panocaps_tag_caption(caption: str) -> str:
    """Convert PanoCaps inline tags into the interleaved grounded answer.

    Tags look like '<1,2:some text>' or '<0:other text>', where the numbers are
    mask ids and the text is the phrase. Each tag becomes '<p> text </p> [SEG]':
    exactly one [SEG] per phrase, which grounds to its *set* of masks (carried
    separately via seg_group_ids). A tag whose ids are not plain comma-separated
    digits (e.g. a range '1-3') is left untouched.
    """
    tag_re = re.compile(r'<\s*([0-9,\s]+)\s*:\s*([^>]+?)\s*>')

    def repl(m: re.Match) -> str:
        ids_raw = m.group(1)
        text = m.group(2)
        tokens = [tok.strip() for tok in ids_raw.split(',') if tok.strip()]
        if not all(tok.isdigit() for tok in tokens):
            return m.group(0)
        return f"<p> {text} </p> [SEG]"

    return tag_re.sub(repl, caption)


class PanoCapsDataset(PanoramaBaseDataset):
    """PanoCaps train split: tagged captions + instance masks -> grounded captioning samples."""

    def __init__(self,
                 data_path: str,
                 image_folder: str,
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 2048,
                 special_tokens=None,
                 arch_type: Literal['qwen'] = 'qwen',
                 preprocessor=None,
                 extra_image_processor=None,
                 repeats: float = 1.0,
                 name: str = 'PanoCapsDataset',
                 **kwargs):
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
            **kwargs,
        )

        self.data_path = data_path
        self.image_folder = image_folder
        self.question_templates = PANOCAPS_QUESTIONS
        self.begin_str = '<image>\n'

        self.data_list = self._load_annotations()
        if not self.data_list:
            raise RuntimeError(f"No PanoCaps samples found for '{self.data_path}'.")

    def _load_annotations(self) -> List[Dict]:
        """Load paired caption + GT instance-mask COCO annotations."""
        caption_json = self.data_path + '_caption.json'
        instance_json = self.data_path + '_mask.json'
        assert os.path.isfile(caption_json) and os.path.isfile(instance_json), \
            f"Invalid PanoCaps paths:\n  captions: {caption_json}\n  instances: {instance_json}"

        print(f"Loading PanoCaps captions: {caption_json}\n  instances: {instance_json}")
        coco_caps = COCO(caption_json)
        coco_inst = COCO(instance_json)

        data_list = []
        for img_id in coco_caps.getImgIds():
            cap_ann = coco_caps.loadAnns(coco_caps.getAnnIds(imgIds=[img_id]))[0]
            caption = cap_ann["caption_ann"]
            # flattened mask ids, in the order phrases appear in the caption
            gt_mask_order = [int(i) for d in cap_ann["label_matched"] for i in d["mask_ids"]]
            # phrase index per mask (parallel to gt_mask_order): groups a phrase's masks
            seg_group_ids = [j for j, d in enumerate(cap_ann["label_matched"]) for _ in d["mask_ids"]]

            inst_anns = coco_inst.loadAnns(coco_inst.getAnnIds(imgIds=[img_id]))
            masks = [a["segmentation"] for a in inst_anns] if inst_anns else None

            file_name = coco_inst.loadImgs([img_id])[0]["file_name"]

            data_list.append({
                "caption": caption,
                "gt_mask_order": gt_mask_order,
                "seg_group_ids": seg_group_ids,
                "masks": masks,
                "file_name": file_name,
            })
        return data_list

    def real_len(self) -> int:
        return len(self.data_list)

    def decode_mask(self, object_masks, ori_height, ori_width):
        """Decode per-instance segmentations (RLE dict or polygons) into a
        (N, H, W) uint8 tensor of {0,1} masks, or None if empty."""
        if not object_masks:
            return None
        binary_masks = []
        for segm in object_masks:
            # RLE dict
            if isinstance(segm, dict) and "counts" in segm:
                m = mask_utils.decode(segm).astype(np.uint8)
                m = (m > 0).astype(np.uint8)
                binary_masks.append(m)
            # polygon list (one instance may have several polygons -> merge)
            elif isinstance(segm, list):
                rles = mask_utils.frPyObjects(segm, ori_height, ori_width)
                m = mask_utils.decode(rles)
                if m.ndim == 3:
                    m = np.any(m, axis=2).astype(np.uint8)
                else:
                    m = (m > 0).astype(np.uint8)
                binary_masks.append(m)
            else:
                raise ValueError(f"Unsupported segmentation format for one instance: {type(segm)}")
        if not binary_masks:
            return None
        masks = np.stack(binary_masks, axis=0).astype(np.uint8)
        return torch.from_numpy(masks)

    def _create_panocaps_conversation(self, caption: str) -> List[Dict]:
        """Build the interleaved grounded-captioning conversation."""
        question = random.choice(self.question_templates).strip()
        question = self.begin_str + question
        answer = panocaps_tag_caption(caption)
        return [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': answer},
        ]

    def prepare_data(self, index: int) -> Dict[str, Any]:
        data_dict = copy.deepcopy(self.data_list[index])

        # read image
        image_path = os.path.join(self.image_folder, data_dict['file_name'])
        image = self._read_image(image_path)
        if image is None:
            return None
        ori_width, ori_height = image.size

        # decode GT masks, then reorder them to match the [SEG] order of the caption
        decoded_masks = self.decode_mask(
            data_dict['masks'], ori_height=ori_height, ori_width=ori_width
        )
        if decoded_masks is None or decoded_masks.shape[0] == 0:
            return None

        gt_order = np.asarray(data_dict['gt_mask_order'], dtype=int)
        in_range = gt_order[(gt_order >= 0) & (gt_order < decoded_masks.shape[0])]
        if in_range.size < gt_order.size:
            print(f"[PanoCaps][WARN] {data_dict['file_name']} mask-order out of range: "
                  f"{gt_order.tolist()} (num masks: {decoded_masks.shape[0]})")
            return None

        masks = decoded_masks.index_select(0, torch.as_tensor(in_range, dtype=torch.long))

        # Build the conversation (one '<p> phrase </p> [SEG]' per phrase); masks stay flat,
        # grouped by phrase index (a phrase may ground to several masks, e.g.
        # '<1,2:the two cats>'). Skip the sample if the phrase count and the grouping disagree.
        conversation = self._create_panocaps_conversation(data_dict['caption'])
        n_tags = conversation[1]['value'].count('</p>')
        grp = np.asarray(data_dict['seg_group_ids'], dtype=int)
        n_phrases = (int(grp.max()) + 1) if grp.size else 0
        if grp.size != masks.shape[0] or n_tags != n_phrases:
            print(f"[PanoCaps][WARN] {data_dict['file_name']} grouped phrase mismatch: "
                  f"{n_tags} <p> tags vs {n_phrases} phrases ({grp.size} grp ids vs {masks.shape[0]} masks)")
            return None
        out_data_dict = {'masks': masks,
                         'seg_group_ids': torch.as_tensor(grp, dtype=torch.long)}

        # process image (pixel_values, image_grid_thw, g_pixel_values, num_image_tokens)
        image_data = self._process_single_image(image)
        out_data_dict.update(image_data)

        # encode the conversation
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(conversation, image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)

        return out_data_dict
